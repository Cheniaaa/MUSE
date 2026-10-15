"""仅消费descriptor的包级selection解析。"""
from copy import deepcopy
from decimal import Decimal
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from pydantic import Field, ValidationError

from muse_sdk.config import ConfigError, DataModel, check_json, resolve_config
from muse_sdk.descriptor import PackageDescriptor, iter_selections, parse_descriptor_json, _reference
from muse_sdk.selection import ResolvedPackageSelection, ResolvedSelection
from muse_sdk.experiment import (
    ActionSlice, ActionSpec, AlgorithmInputs, BoundSelection, BoundTask, CatalogSnapshot, EntityBindingPlan,
    CostConstraint, EntityModelRef, ExperimentDefinition, ExperimentRequest, ResourceRef, ValidationReport,
)
from muse_sdk.descriptor import GroupBinding, InputField, RoleBinding, TensorSpec, TrainingSchedule


class SelectionOverride(DataModel):
    enabled: bool | None = None
    config: dict = Field(default_factory=dict)
    weight: float | None = None


def _overrides(request: dict, allowed: dict, path=()) -> dict:
    if not isinstance(request, dict):
        raise ConfigError('.'.join(path) or 'term_config', 'Expected object')
    output = {}
    for key, value in request.items():
        current = (*path, key)
        if key not in allowed:
            raise ConfigError('.'.join(current), 'Unknown selection identity')
        if allowed[key] is None:
            override = SelectionOverride.model_validate(value)
            for field in ('enabled', 'weight'):
                if field in override.model_fields_set and getattr(override, field) is None:
                    raise ConfigError('.'.join((*current, field)), 'Explicit null is not allowed')
            if 'role_id' in override.config:
                raise ConfigError('.'.join(current), 'role_id cannot be overridden')
            output[current] = override
        else:
            output.update(_overrides(value, allowed[key], current))
    return output


def resolve_package_selections(descriptor: PackageDescriptor, term_config: dict,
                               builtin_catalog: PackageDescriptor | None = None) -> ResolvedPackageSelection:
    """descriptor来自build_descriptor或parse_descriptor_json；只处理本次覆盖。"""
    check_json(term_config, 'term_config')
    allowed = {'roles': {role: {} for role in descriptor.composition.agent_roles},
               'critic_inputs': {group: {} for group in descriptor.composition.critic_inputs}, 'world': {}}
    for owner, kind, selections in iter_selections(descriptor.composition):
        node = allowed
        for part in (*owner, kind):
            node = node.setdefault(part, {})
        node.update({selection.selection_id: None for selection in selections})
    overrides = _overrides(term_config, allowed)
    output, inputs = [], []
    reasons = {}
    costs = set()
    for owner, kind, selections in iter_selections(descriptor.composition):
        for selection in selections:
            identity = (*owner, kind, selection.selection_id)
            term = _reference(selection.term, descriptor.terms, '.'.join(identity),
                              getattr(builtin_catalog, 'terms', None))
            override = overrides.get(identity, SelectionOverride())
            enabled = selection.default_enabled if override.enabled is None else override.enabled
            if selection.required and not enabled:
                raise ConfigError('.'.join(identity), 'Required selection cannot be disabled')
            if kind != 'reward' and 'weight' in override.model_fields_set:
                raise ConfigError('.'.join(identity), 'Only rewards accept weight')
            weight = None
            if kind == 'reward':
                weight = override.weight if override.weight is not None else (selection.weight if selection.weight is not None else 1.0)
            try:
                config = resolve_config(term.config_schema, selection.config, override.config, require_complete=enabled)
            except ConfigError as exc:
                for error in exc.errors:
                    error['field'] = '.'.join((*identity, 'config', error['field']))
                raise
            role = owner[1] if owner[0] == 'roles' else selection.role_id
            if enabled and kind == 'cost':
                cost_key = (*owner, term.cost_id)
                if cost_key in costs:
                    raise ConfigError('.'.join(identity), f'Duplicate enabled cost_id {term.cost_id}')
                costs.add(cost_key)
            if kind == 'termination':
                reasons[len(reasons) + 1] = ('world', 'termination', selection.selection_id)
            if enabled:
                inputs.extend({'selection_identity': list(identity), 'binding_role_id': role,
                               'field': item.model_dump(mode='json')} for item in term.input_fields)
            output.append(ResolvedSelection(
                owner_path=owner, term_kind=kind, selection_id=selection.selection_id,
                term=selection.term, enabled=enabled, config=config, weight=weight, binding_role_id=role,
            ))
    return ResolvedPackageSelection(package_hash=descriptor.package_hash, selections=tuple(output),
                                    termination_reasons=reasons, required_inputs=tuple(deepcopy(inputs)))


def _resource(snapshot, kind, identity, revision):
    for resource in snapshot.resources:
        if (resource.kind, resource.resource_id, resource.revision) == (kind, identity, revision):
            return resource
    raise ConfigError(kind + '_id', 'Unknown resource revision')


def _ref(record):
    return ResourceRef(resource_id=record.resource_id, revision=record.revision, sha256=record.sha256)


def _tensor(spec, batch, slots, group_slots=None, axes=None):
    axes = {**(axes or {}), 'B': batch, 'N_role': slots, 'N_group': group_slots}
    shape = []
    for dim in spec.shape:
        value = axes.get(dim) if isinstance(dim, str) else dim
        if value is None or value <= 0:
            raise ConfigError('input.shape', f'Unresolved or invalid dimension: {dim}')
        shape.append(value)
    return spec.model_copy(update={'shape': shape}, deep=True)


def _public_fields(batch, bindings, action_specs, groups, sensors):
    """普通飞行env公开缓存的单一CPU契约；T4按这些字段供给。"""
    fields = {}

    def add(path, shape, units, frame, phase, dtype='float32'):
        fields[tuple(path)] = InputField(path=list(path), phase=phase, tensor=TensorSpec(
            shape=shape, dtype=dtype, units=units, frame=frame, rate='decision'))

    for role, binding in bindings.items():
        prefix = [batch, binding.role_slot_count]
        add(['role_positions_world', role], [*prefix, 3], 'm', 'world_z_up', 'post_physics')
        add(['physics_outputs', 'velocity_world', role], [*prefix, 3], 'm/s', 'world_z_up', 'post_physics')
        add(['physics_outputs', 'orientation_wxyz', role], [*prefix, 4], '1', 'world_z_up', 'post_physics')
        add(['physics_outputs', 'angular_velocity_body', role], [*prefix, 3], 'rad/s', 'body_flu', 'post_physics')
        add(['physics_outputs', 'previous_action', role], action_specs[role].shape,
            'mixed', 'mixed', 'post_physics')
        add(['commands', 'target_world', role], [*prefix, 3], 'm', 'world_z_up', 'next_command')
        add(['act_commands', 'target_world', role], [*prefix, 3], 'm', 'world_z_up', 'action')
        add(['physics_outputs', 'crashed', role], prefix, '1', 'none', 'post_physics', 'bool')
        for name in ('present', 'alive', 'controllable'):
            add(['role_masks', role, name], prefix, '1', 'none', 'post_physics', 'bool')
    add(['physics_outputs', 'episode_decision'], [batch], '1', 'none', 'post_physics', 'int64')
    for group, profile in sensors.items():
        prefix = [batch, groups[group].count]
        for signal, output in profile['outputs'].items():
            path = ['sensor_outputs', group, signal]
            add([*path, 'data'], [*prefix, output['width']], output['units'], output['frame'], 'post_physics')
            add([*path, 'valid'], prefix, '1', 'none', 'post_physics', 'bool')
            add([*path, 'age'], prefix, 's', 'none', 'post_physics')
            add([*path, 'sample_tick'], prefix, '1', 'none', 'post_physics', 'int64')
    return fields


def _bind_input(item, role, batch, bindings, fields, kind, identity, axes=None):
    if kind == 'algorithm' and item.path[0] == 'sensor_outputs':
        raise ConfigError(identity + '.input_fields', 'Raw sensor inputs must be read through an observation Term')
    if kind == 'task' and (item.path[0] == 'observations' or item.phase in ('action', 'post_observation')):
        raise ConfigError(identity + '.input_fields', 'Task inputs must be available before Task update and at construction')
    group = bindings[role].group_id if role in bindings else None
    path = [role if isinstance(part, RoleBinding) else group if isinstance(part, GroupBinding) else part
            for part in item.path]
    if any(part is None for part in path):
        raise ConfigError(identity + '.input_fields', 'A concrete role binding is required')
    slots = bindings[role].role_slot_count if role in bindings else None
    actual = fields.get(tuple(path))
    if actual is None:
        raise ConfigError(identity + '.input_fields', f'Unavailable public field: {path}')
    group_slots = actual.tensor.shape[1] if path[0] == 'sensor_outputs' else None
    dimensions = dict(axes or {})
    if path[:2] == ['network', 'services'] and path[-1] in ('remaining_bits', 'served_bits', 'valid'):
        dimensions['N_service_links'] = actual.tensor.shape[1]
    expected = _tensor(item.tensor, batch, slots, group_slots, axes=dimensions)
    if expected != actual.tensor or item.phase != actual.phase:
        raise ConfigError(identity + '.input_fields', f'Input shape/units/frame/phase mismatch: {path}')
    if kind == 'observation' and item.phase in ('action', 'post_observation'):
        raise ConfigError(identity + '.input_fields', 'Observation must also be valid at reset/shape probe')
    if kind in ('reward', 'termination', 'cost') and path[:2] == ['commands', 'target_world']:
        raise ConfigError(identity + '.input_fields', 'Transition scoring must bind act_commands')
    return actual.model_copy(deep=True)


def _resolve_costs(raw, package_selection, descriptor, builtin_catalog, roles):
    """将启用cost与显式约束按owner冻结；空配置保持普通实验语义。"""
    if not isinstance(raw, dict):
        raise ConfigError('cost_config', 'Expected object')
    config_roles = raw.get('roles', {})
    config_world = raw.get('world', {})
    if set(raw) - {'roles', 'world'} or not isinstance(config_roles, dict) or not isinstance(config_world, dict):
        raise ConfigError('cost_config', 'Expected roles and world objects only')
    if set(config_roles) - set(roles):
        raise ConfigError('cost_config.roles', 'Unknown role')
    if any(not isinstance(value, dict) for value in (*config_roles.values(), config_world)):
        raise ConfigError('cost_config', 'Each owner must contain cost objects')

    resolved = {'roles': {}, 'world': {}}
    consumed = set()
    for selection in package_selection.selections:
        if not selection.enabled or selection.term_kind != 'cost':
            continue
        term = _reference(selection.term, descriptor.terms, 'term', builtin_catalog.terms)
        owner_kind = 'roles' if selection.owner_path[0] == 'roles' else 'world'
        owner = selection.owner_path[1] if owner_kind == 'roles' else None
        target = config_roles.get(owner, {}) if owner is not None else config_world
        cost_id = term.cost_id
        identity = '.'.join((*selection.owner_path, 'cost', selection.selection_id))
        if cost_id not in target:
            raise ConfigError('cost_config.' + (f'roles.{owner}' if owner is not None else 'world'),
                              f'Missing constraint for enabled cost_id {cost_id}')
        value = target[cost_id]
        try:
            constraint = CostConstraint.model_validate(value)
        except ValidationError as exc:
            error = ConfigError(f'cost_config.{"roles." + owner if owner is not None else "world"}.{cost_id}',
                                'Invalid cost constraint')
            error.errors = [
                {'field': f'cost_config.{"roles." + owner if owner is not None else "world"}.{cost_id}.'
                          f'{".".join(map(str, item["loc"]))}', 'message': item['msg']}
                for item in exc.errors()
            ]
            raise error
        target_output = resolved['world'] if owner is None else resolved['roles'].setdefault(owner, {})
        target_output[cost_id] = {
            **constraint.model_dump(), 'cost_id': cost_id,
            'scope': 'role' if owner is not None else 'world',
            'units': term.output_spec.units, 'statistic': 'episode_discounted_sum',
            'boundary': 'world_or_learning_done' if owner is not None else 'world',
        }
        consumed.add(('roles', owner, cost_id) if owner is not None else ('world', cost_id))

    for owner, values in config_roles.items():
        for cost_id in values:
            if ('roles', owner, cost_id) not in consumed:
                raise ConfigError(f'cost_config.roles.{owner}.{cost_id}', 'No enabled cost uses this constraint')
    for cost_id in config_world:
        if ('world', cost_id) not in consumed:
            raise ConfigError(f'cost_config.world.{cost_id}', 'No enabled cost uses this constraint')
    return resolved


def resolve_experiment(request: dict, catalog_snapshot: CatalogSnapshot) -> ValidationReport:
    """真实请求边界：只消费CPU descriptor，返回完整普通飞行Experiment。"""
    try:
        check_json(request)
        data = ExperimentRequest.model_validate(request)
        experiment = _resolve_flight(data, catalog_snapshot)
        return ValidationReport(valid=True, resolved_experiment=experiment, errors=[], warnings=[
            {'field': 'physics', 'message': 'CPU resolution only; GPU compile and rollout are not validated'},
        ])
    except (ConfigError, ValidationError) as exc:
        errors = exc.errors if isinstance(exc, ConfigError) else [
            {'field': '.'.join(map(str, e['loc'])), 'message': e['msg']} for e in exc.errors()]
        return ValidationReport(valid=False, resolved_experiment=None, errors=errors, warnings=[])


def _resolve_flight(data, snapshot):
    import math
    environment = _resource(snapshot, 'environment', data.environment_id, data.environment_revision)
    package = _resource(snapshot, 'training_package', data.training_package_id, data.training_package_revision)
    builtins = snapshot.builtins
    descriptor = parse_descriptor_json(json.dumps(package.metadata), builtins.descriptor)
    composition = descriptor.composition
    if descriptor.package_hash != package.sha256:
        raise ConfigError('training_package', 'Descriptor does not match registered content hash')
    if data.task_id != composition.task:
        raise ConfigError('task_id', 'Select the package composition Task')
    task, task_config = None, {}
    if data.task_id is None:
        if data.task_config or composition.task_config:
            raise ConfigError('task_config', 'Task config requires a selected Task')
    else:
        task = _reference(data.task_id, descriptor.tasks, 'task_id', builtins.descriptor.tasks)
        task_config = resolve_config(task.config_schema, composition.task_config, data.task_config,
                                     require_complete=True)
        if task.role_status_bindings:
            raise ConfigError('task.role_status_bindings', 'M2.5 physical fallback is not enabled; role_status bindings must be empty')
    if data.algorithm_id != composition.algorithm:
        raise ConfigError('algorithm_id', 'Select the package composition Algorithm')
    algorithm = _reference(data.algorithm_id, descriptor.algorithms, 'algorithm_id', builtins.descriptor.algorithms)
    algorithm_config = resolve_config(algorithm.config_schema, composition.algorithm_config,
                                      data.algorithm_config, require_complete=True)
    training_schedule = None
    if algorithm.training_schedule is not None:
        field = algorithm.training_schedule.collection_steps
        value = algorithm_config[field]
        if type(value) is not int or value <= 0:
            raise ConfigError('algorithm_config.' + field, 'Expected positive integer collection_steps')
        training_schedule = TrainingSchedule(collection_steps=value)
    batch = data.topology.world_count
    if any(world < 0 or world >= batch for world in data.preview.world_ids):
        raise ConfigError('preview.world_ids', 'World is outside topology')
    groups, models, controllers, sensors = {}, {}, {}, {}
    for index, group in enumerate(data.scenario.entity_groups):
        field = f'scenario.entity_groups[{index}]'
        if group.group_id in groups:
            raise ConfigError(field + '.group_id', 'Duplicate group')
        groups[group.group_id] = group
        model = builtins.entity_models.get(group.model_id)
        if model is None or model.revision != group.model_revision:
            raise ConfigError(field + '.model_id', 'Unknown entity model revision')
        models[group.model_id] = EntityModelRef(**_ref(model).model_dump(),
            physics_asset=model.metadata['physics_asset'], root_body=model.metadata['root_body'],
            support_geoms=model.metadata['support_geoms'], radio_devices=model.metadata.get('radio_devices', {}))
        if group.controller_id not in model.metadata['controller_ids'] or group.controller_id not in builtins.controllers:
            raise ConfigError(field + '.controller_id', 'Unknown or incompatible controller')
        if group.sensor_profile_id not in model.metadata['sensor_profile_ids'] or group.sensor_profile_id not in builtins.sensors:
            raise ConfigError(field + '.sensor_profile_id', 'Unknown or incompatible sensor profile')
        controllers[group.group_id] = deepcopy(builtins.controllers[group.controller_id])
        profile = deepcopy(builtins.sensors[group.sensor_profile_id])
        try:
            config = resolve_config(profile['config_schema'], {}, group.sensor_config, require_complete=True)
        except ConfigError as exc:
            for error in exc.errors:
                error['field'] = field + '.sensor_config.' + error['field']
            raise
        periods = {}
        for rate_key in dict.fromkeys(output['rate_key'] for output in profile['outputs'].values()):
            rate_field = field + '.sensor_config.' + rate_key
            try:
                period = 1. / config[rate_key] / data.scenario.scene.physics_dt
            except OverflowError as exc:
                raise ConfigError(rate_field, 'Rate must align to positive integer physics ticks') from exc
            if not math.isfinite(period):
                raise ConfigError(rate_field, 'Expected finite positive integer physics ticks')
            period_ticks = round(period)
            if period_ticks < 1 or not math.isclose(period, period_ticks, rel_tol=1e-9, abs_tol=1e-9):
                raise ConfigError(rate_field, 'Rate must align to positive integer physics ticks')
            if period_ticks > 2**63 - 1:
                raise ConfigError(rate_field, 'Physics tick period must fit int64')
            config[rate_key] = 1. / (period_ticks * data.scenario.scene.physics_dt)
            periods[rate_key] = period_ticks
        for output in profile['outputs'].values():
            output['period_ticks'] = periods[output['rate_key']]
            # 先规范化，再比较十进制声明时间，避免float乘除法绕过严格下界。
            if ('minimum_period_s' in output
                    and Decimal(str(data.scenario.scene.physics_dt)) * output['period_ticks']
                    <= Decimal(str(output['minimum_period_s']))):
                raise ConfigError(field + '.sensor_config.' + output['rate_key'],
                                  f'Canonical period must be greater than {output["minimum_period_s"]} seconds')
        group.sensor_config = config
        profile['config'] = deepcopy(config)
        sensors[group.group_id] = profile
        placements = data.scenario.placements.get(group.placement_id)
        if placements is None or len(placements) != group.count:
            raise ConfigError(field + '.placement_id', 'Exactly one explicit pose per group slot is required')
        for pose in placements:
            if not math.isclose(sum(x * x for x in pose.orientation_wxyz), 1., abs_tol=1e-6):
                raise ConfigError(field + '.placement_id', 'orientation_wxyz must be a unit quaternion')
    if set(data.role_config) != set(composition.agent_roles):
        raise ConfigError('role_config', 'Bindings must match all declared roles')
    if set(data.scenario.command_targets_world) != set(composition.agent_roles):
        raise ConfigError('scenario.command_targets_world', 'Command targets must match declared roles')
    bindings, actions, owned = {}, {}, set()
    motion_interfaces = {
        'body_velocity': (['vx_b', 'vy_b', 'vz_b', 'yaw_rate'], ['m/s', 'm/s', 'm/s', 'rad/s'],
                          ['body_flu', 'body_flu', 'body_flu', 'world_heading']),
        'diff_drive': (['v_forward', 'yaw_rate'], ['m/s', 'rad/s'], ['body_flu', 'world_heading']),
    }
    for role, declared in composition.agent_roles.items():
        config = data.role_config[role]
        group = groups.get(config.group_id)
        if group is None:
            raise ConfigError('role_config.' + role, 'Unknown group')
        meta = builtins.entity_models[group.model_id].metadata
        if declared.entity_kind != meta['entity_kind']:
            raise ConfigError('role_config.' + role, 'Entity kind mismatch')
        interface = declared.action_interfaces.get('motion')
        if (set(declared.action_interfaces) - {'motion', 'radio'} or interface not in motion_interfaces
                or {'motion': interface} != meta['action_interfaces']):
            raise ConfigError('role_config.' + role, 'Declared motion interface does not match the entity model')
        if interface not in algorithm.action_requirements:
            raise ConfigError('algorithm_id', f'Algorithm does not support {role} motion interface {interface}')
        if 'radio' in declared.action_interfaces and (declared.action_interfaces['radio'] != 'power_channel'
                or 'power_channel' not in algorithm.action_requirements):
            raise ConfigError('algorithm_id', f'Algorithm does not support {role} radio power_channel interface')
        if any(slot < 0 or slot >= group.count for slot in config.slots) or len(set(config.slots)) != len(config.slots):
            raise ConfigError('role_config.' + role, 'Duplicate or out-of-range group slot')
        for slot in config.slots:
            key = (group.group_id, slot)
            if key in owned:
                raise ConfigError('role_config.' + role, 'Multiple action owners for one group slot')
            owned.add(key)
        targets = data.scenario.command_targets_world[role]
        if len(targets) != len(config.slots) or any(len(target) != 3 for target in targets):
            raise ConfigError('scenario.command_targets_world.' + role, 'Expected one xyz target per role slot')
        bindings[role] = EntityBindingPlan(role_id=role, group_id=group.group_id, model_id=group.model_id,
            entity_kind=declared.entity_kind, role_slot_count=len(config.slots), role_slot_to_group_slot=config.slots,
            action_interfaces=declared.action_interfaces, controller_id=group.controller_id,
            sensor_profile_id=group.sensor_profile_id)
        motion = controllers[group.group_id]['motion']
        motion_fields, units, frames = motion_interfaces[interface]
        dimension = len(motion_fields)
        if (motion['fields'] != motion_fields or motion['units'] != units or motion['frames'] != frames
                or len(motion['minimum']) != dimension or len(motion['maximum']) != dimension
                or any(lo >= hi for lo, hi in zip(motion['minimum'], motion['maximum']))):
            raise ConfigError('action_specs.' + role, f'{interface} dimension/units/frame/bounds mismatch')
        actions[role] = ActionSpec(interface=interface, dtype='float32',
                                  shape=[batch, len(config.slots), dimension], **motion)
    if len(owned) != sum(group.count for group in groups.values()
                         if builtins.entity_models[group.model_id].metadata['action_interfaces']):
        raise ConfigError('role_config', 'Every actionable entity needs an explicit action owner')
    package_selection = resolve_package_selections(descriptor, data.term_config, builtins.descriptor)
    costs = _resolve_costs(data.cost_config, package_selection, descriptor, builtins.descriptor,
                           composition.agent_roles)
    from muse_sdk.geometry import EnvironmentRF, GeometryAssetSummary, geometry_input_fields
    from muse_sdk.network import network_axes, network_public_fields
    from .geometry_resolution import resolve_endpoints, resolve_geometry
    from .network_resolution import resolve_network
    geometry, endpoints = None, None
    if data.scenario.radio_geometry is not None:
        endpoints = resolve_endpoints(data.scenario.radio_geometry, groups=groups, entity_models=models)
        rf = environment.metadata['rf_geometry']
        summary = environment.metadata.get('geometry_summary')
        geometry = resolve_geometry(data.scenario.radio_geometry,
            environment_rf=EnvironmentRF.model_validate(rf) if rf is not None else None,
            summary=GeometryAssetSummary.model_validate(summary) if summary is not None else None,
            groups=groups, entity_models=models, endpoints=endpoints)
    required_inputs = [(item, 'task:' + data.task_id) for item in task.input_fields] if task is not None else []
    required_inputs.extend((item, 'algorithm:' + data.algorithm_id) for item in algorithm.observation_requirements)
    for raw in package_selection.required_inputs:
        required_inputs.append((InputField.model_validate(raw['field']), '.'.join(raw['selection_identity'])))
    role_requirements = {role: [] for role in bindings}
    for role, declared in composition.agent_roles.items():
        for requirement in declared.input_requirements:
            if not isinstance(requirement, dict):
                raise ConfigError('roles.' + role + '.input_requirements', 'Use structured field requirements')
            item = InputField.model_validate(requirement)
            role_requirements[role].append(item)
            required_inputs.append((item, 'role:' + role))
    network, model_bindings, network_activation = resolve_network(data, descriptor=descriptor, builtins=builtins,
        bindings=bindings, endpoints=endpoints, geometry=geometry, required_inputs=required_inputs)
    axes = {} if network is None else network_axes(network, geometry=geometry, batch_size=batch)
    if network is not None:
        from muse_sdk.experiment import RadioActionSpec
        controls = {control.role_id: control for control in network.role_controls}
        for role, action in actions.items():
            if role in controls:
                budget = network.endpoint_limits[controls[role].endpoint_indices[0]]
                action.radio = RadioActionSpec(power_minimum_w=0., power_maximum_w=budget.max_power_w,
                    channel_ids=[network.resources[f].channel_id for f in budget.resource_indices])
    layout, offset = [], 0
    for role, action in actions.items():
        for field, dimension in [('motion', action.shape[-1]), *([('radio', 2)] if action.radio is not None else [])]:
            width = action.shape[1] * dimension
            layout.append(ActionSlice(role_id=role, field=field, start=offset, stop=offset + width,
                shape=[batch, action.shape[1], dimension]))
            offset += width
    fields = _public_fields(batch, bindings, actions, groups, sensors)
    activation = {'physics': ['scenario.entity_groups'], 'commands': ['scenario.command_targets_world']}
    activation.update(network_activation)
    if network is not None:
        fields.update((tuple(item.path), item) for item in network_public_fields(network, model_bindings, batch_size=batch))
        if geometry is not None:
            fields.update((tuple(item.path), item) for item in geometry_input_fields(geometry.layout, batch_size=batch, phase='post_physics'))
    task_spec = None
    if task is not None:
        if any(role not in bindings for role in task.learning_done_roles):
            raise ConfigError('task.learning_done_roles', 'Unknown learning_done role')
        task_inputs = []
        for item in task.input_fields:
            # Task 无隐式 role owner，具体路径才能决定 N_role。
            role = None
            if item.path[0] in ('role_positions_world', 'role_masks'):
                role = item.path[1]
            elif item.path[0] in ('physics_outputs', 'commands') and len(item.path) > 2:
                role = item.path[2]
            role = role if isinstance(role, str) and role in bindings else None
            task_inputs.append(_bind_input(item, role, batch, bindings, fields, 'task', 'task', axes=axes))
        published = {}
        for name, spec in task.published_fields.items():
            if spec.shape[:1] != ['B']:
                raise ConfigError('task.published_fields.' + name, 'Task shared fields require world shape [B, ...]')
            published[name] = _tensor(spec, batch, None, axes=axes)
        # 先校验全部 inputs，再发布，防止 Task 自引用尚未构造的字段。
        for name, spec in published.items():
            fields[('task', name)] = InputField(path=['task', name], phase='post_task', tensor=spec)
        task_spec = BoundTask(input_fields=task_inputs, published_fields=published,
                               learning_done_roles=list(task.learning_done_roles))
        for item in task_inputs:
            if item.path[0] == 'sensor_outputs':
                consumers = activation.setdefault('sensor:' + item.path[1], [])
                if 'task:' + data.task_id not in consumers:
                    consumers.append('task:' + data.task_id)
    # 全部observation先于reward/cost/termination，不让role声明顺序变成执行依赖。
    observation_roles = {}
    for selection in package_selection.selections:
        if not selection.enabled or selection.term_kind != 'observation':
            continue
        term = _reference(selection.term, descriptor.terms, 'term', builtins.descriptor.terms)
        role = selection.binding_role_id
        output = _tensor(term.output_spec, batch, bindings[role].role_slot_count if role else None, axes=axes)
        path = (['observations', 'actors', selection.owner_path[1], selection.selection_id]
                if selection.owner_path[0] == 'roles' else
                ['observations', 'critic_inputs', selection.owner_path[1], selection.selection_id])
        fields[tuple(path)] = InputField(path=path, phase='post_observation', tensor=output)
        observation_roles[tuple(path)] = role
    selections = []
    for group in groups.values():
        if group.controller_id != 'none':
            activation['controller:' + group.group_id] = ['role action motion']
    for selection in package_selection.selections:
        term = _reference(selection.term, descriptor.terms, 'term', builtins.descriptor.terms)
        identity = '.'.join((*selection.owner_path, selection.term_kind, selection.selection_id))
        role = selection.binding_role_id
        if role is None and 'controlled_role' in selection.config:
            role = selection.config['controlled_role']
        inputs, output = [], None
        if selection.enabled:
            if role is not None and role not in bindings:
                raise ConfigError(identity, 'Unknown controlled role')
            if (selection.term_kind == 'reward' and training_schedule is not None
                    and term.scope not in algorithm.reward_inputs):
                raise ConfigError(identity, f'Algorithm does not consume {term.scope} reward')
            if selection.term_kind == 'cost' and term.scope not in algorithm.cost_inputs:
                raise ConfigError(identity, f'Algorithm does not consume {term.scope} cost')
            inputs = [_bind_input(item, role, batch, bindings, fields, selection.term_kind, identity, axes=axes)
                       for item in term.input_fields]
            for item in inputs:
                if item.path[0] == 'sensor_outputs':
                    consumers = activation.setdefault('sensor:' + item.path[1], [])
                    if identity not in consumers:
                        consumers.append(identity)
            output = _tensor(term.output_spec, batch, bindings[role].role_slot_count if role else None, axes=axes)
            activation.setdefault(selection.term_kind, []).append(identity)
        selections.append(BoundSelection(**selection.model_dump(), input_fields=inputs, output_spec=output,
                                         time_out=term.time_out, cost_id=term.cost_id))
    if (training_schedule is not None and set(algorithm.reward_inputs) == {'world'}
            and not any(s.enabled and s.term_kind == 'reward' and s.owner_path == ('world',) for s in selections)):
        raise ConfigError('world.reward', 'World-only learning requires an enabled world reward')
    role_inputs = {role: [_bind_input(item, role, batch, bindings, fields, 'algorithm', role, axes=axes)
                         for item in requirements] for role, requirements in role_requirements.items()}
    algorithm_observations = [_bind_input(item, None, batch, bindings, fields, 'algorithm', 'algorithm', axes=axes)
                              for item in algorithm.observation_requirements]
    actor_inputs, critic_inputs = {}, {}
    if algorithm.actor_observation == 'all_local':
        for role in bindings:
            local = [item for path, item in fields.items() if path[:3] == ('observations', 'actors', role)]
            if not local:
                raise ConfigError('algorithm.actor_observation.' + role, 'Algorithm requires local actor observations')
            actor_inputs[role] = local
    else:
        for role, items in algorithm.actor_observation.items():
            if role not in bindings:
                raise ConfigError('algorithm.actor_observation', 'Unknown role')
            actor_inputs[role] = []
            for item in items:
                bound = _bind_input(item, role, batch, bindings, fields, 'algorithm', 'algorithm', axes=axes)
                if bound.path[:3] != ['observations', 'actors', role]:
                    raise ConfigError('algorithm.actor_observation', 'Actor must only read its own observations')
                actor_inputs[role].append(bound)
    if algorithm.critic_inputs == 'all_centralized':
        for path, item in fields.items():
            if path[:2] == ('observations', 'critic_inputs'):
                critic_inputs.setdefault(path[2], []).append(item)
        if not critic_inputs:
            raise ConfigError('algorithm.critic_inputs', 'Algorithm requires centralized critic observations')
    else:
        for group, items in algorithm.critic_inputs.items():
            critic_inputs[group] = []
            for item in items:
                if item.path[:2] != ['observations', 'critic_inputs']:
                    raise ConfigError('algorithm.critic_inputs', 'Critic inputs must read observations.critic_inputs')
                # 显式 critic 无 role owner；具体 observation selection 才提供真实绑定。
                if any(isinstance(part, (RoleBinding, GroupBinding)) for part in item.path):
                    raise ConfigError('algorithm.critic_inputs.input_fields', 'A concrete role binding is required')
                role = observation_roles.get(tuple(item.path))
                critic_inputs[group].append(_bind_input(item, role, batch,
                                                        bindings, fields, 'algorithm', 'algorithm.critic_inputs', axes=axes))
    algorithm_inputs = AlgorithmInputs(observations=algorithm_observations, actors=actor_inputs, critics=critic_inputs)
    return ExperimentDefinition(environment=_ref(environment), geometry=geometry, network=network,
        environment_scene=environment.metadata['scene'], environment_metadata=deepcopy(environment.metadata.get('metadata', {})),
        training_package=_ref(package),
        entity_models=models, platform_revision=builtins.platform_revision, scenario=data.scenario.model_copy(deep=True),
        bindings=bindings, controllers=controllers, sensors=sensors, task_id=data.task_id, task_config=task_config,
        task_spec=task_spec,
        algorithm_id=data.algorithm_id, algorithm_config=algorithm_config, training_schedule=training_schedule,
        algorithm_inputs=algorithm_inputs, role_inputs=role_inputs, selections=selections,
        termination_reasons=package_selection.termination_reasons,
        action_specs=actions, action_layout=layout, activation=activation, model_bindings=model_bindings, costs=costs,
        topology=data.topology, root_seed=data.root_seed, preview=data.preview)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(key, 'Duplicate JSON key')
        result[key] = value
    return result


def _read_json(path: Path):
    value = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object)
    check_json(value, str(path))
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor', type=Path, required=True)
    parser.add_argument('--term-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ConfigError('output', 'Output already exists')
        # JSON模式负责线格式到typed DTO的转换；不加载Python注册对象。
        descriptor = parse_descriptor_json(json.dumps(_read_json(args.descriptor), allow_nan=False))
        result = resolve_package_selections(descriptor, _read_json(args.term_config))
        fd, name = tempfile.mkstemp(dir=args.output.parent, prefix='.selection-')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                stream.write(result.model_dump_json(indent=2))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, args.output)
        finally:
            Path(name).unlink(missing_ok=True)
        return 0
    except (ValueError, OSError) as exc:
        errors = exc.errors if isinstance(exc, ConfigError) else [{'field': 'input', 'message': str(exc)}]
        print(json.dumps({'errors': errors}, allow_nan=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
