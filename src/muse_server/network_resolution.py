"""纯CPU网络布局、唯一RF/服务所有者和有限模型激活闭包。"""
import math

from jsonschema import Draft202012Validator, ValidationError

from muse_sdk.config import ConfigError, resolve_config
from muse_sdk.descriptor import InputField, _reference
from muse_sdk.models import RADIO_FIELDS
from muse_sdk.network import (BoundModel, CommunicationBinding, ControlBinding, EmitterBinding,
    EndpointBudget, NetworkSpec, ServiceBinding, network_axes, network_model_input_fields)


def _shape(spec, axes, field):
    shape = []
    for dim in spec.shape:
        value = axes.get(dim) if isinstance(dim, str) else dim
        if value is None or value <= 0:
            raise ConfigError(field, f'Unresolved or invalid dimension: {dim}')
        shape.append(value)
    return spec.model_copy(update={'shape': shape}, deep=True)


def resolve_network(request, *, descriptor, composition, builtins, bindings, endpoints, geometry, required_inputs):
    """required_inputs为(InputField, consumer identity)；仅显式消费激活模型。"""
    selection = request.scenario.network
    reasons = {}
    for item, consumer in required_inputs:
        if item.path[:2] == ['network', 'geometry']:
            reasons.setdefault('geometry', []).append(consumer)
        if item.path[0] == 'sensing_outputs':
            reasons.setdefault('sensing', []).append(consumer)
        elif item.path[:2] == ['network', 'radio']:
            reasons.setdefault('radio', []).append(consumer)
        if item.path[0] == 'sensing_outputs' or item.path[0] == 'network' and item.path[1] in (
                'geometry', 'radio', 'resources', 'endpoint_masks', 'allocations', 'simultaneous_transmissions', 'services', 'interval_s'):
            reasons.setdefault('network', []).append(consumer)
    if selection is not None:
        for role in selection.role_controls:
            reasons.setdefault('network', []).append('role:' + role)
        for emitter in selection.background_emitters:
            reasons.setdefault('network', []).append('emitter:' + emitter.emitter_id)
        for service in selection.services:
            consumer = 'service:' + service.service_id
            reasons.setdefault('network', []).append(consumer)
            reasons.setdefault('radio', []).append(consumer)
    for role, binding in bindings.items():
        if binding.action_interfaces.get('radio') == 'power_channel':
            reasons.setdefault('network', []).append('role:' + role)
    if set(request.model_parameters) - set(composition.model_bindings):
        raise ConfigError('model_config', 'Model config requires a declared radio/sensing binding')
    for kind, choice in request.model_choices.items():
        if kind not in composition.model_bindings or choice not in composition.model_bindings[kind].options:
            raise ConfigError('model_choices.' + kind, 'Model choice must be declared by the selected composition')

    def model_descriptor(kind):
        binding = composition.model_bindings.get(kind)
        if binding is None:
            raise ConfigError('model_bindings.' + kind, 'Required model binding is missing')
        reference = request.model_choices.get(kind, binding.default)
        return reference, _reference(reference, getattr(descriptor, kind + '_models'),
            'model_bindings.' + kind, getattr(builtins.descriptor, kind + '_models'))

    if not reasons:
        for kind, override in request.model_parameters.items():
            reference, meta = model_descriptor(kind)
            resolve_config(meta.config_schema, composition.model_bindings[kind].options[reference].config,
                           override, require_complete=False)
        return None, {}, {}
    if selection is None or endpoints is None or not endpoints[0]:
        raise ConfigError('scenario.network', 'Wireless consumers require network and explicit device endpoints')

    points, sets = endpoints
    field = 'scenario.network'
    resources = selection.resources
    ids = [r.resource_id for r in resources]
    channels = [r.channel_id for r in resources]
    if len(set(ids)) != len(ids) or len(set(channels)) != len(channels):
        raise ConfigError(field + '.resources', 'Resource IDs and channel IDs must be unique')
    bands = []
    for resource in resources:
        low = resource.center_frequency_hz - resource.bandwidth_hz / 2
        high = resource.center_frequency_hz + resource.bandwidth_hz / 2
        if low <= 0 or not math.isfinite(high):
            raise ConfigError(field + '.resources', 'Frequency interval must be finite with positive lower edge')
        if any(low < hi and lo < high for lo, hi in bands):
            raise ConfigError(field + '.resources', 'Resource frequency intervals overlap')
        bands.append((low, high))

    def resource_indices(names, path):
        if len(set(names)) != len(names) or any(name not in ids for name in names):
            raise ConfigError(path, 'Expected distinct declared resource IDs')
        return tuple(ids.index(name) for name in names)

    def collection(name, path):
        if name not in sets:
            raise ConfigError(path, 'Unknown endpoint collection: ' + name)
        return sets[name]

    limits = [None] * len(points)
    for name, limit in selection.endpoint_limits.items():
        path = field + '.endpoint_limits.' + name
        indices = collection(name, path)
        budget = EndpointBudget(resource_indices=resource_indices(limit.resource_ids, path), max_power_w=limit.max_power_w)
        for index in indices:
            limits[index] = budget
    if any(limit is None for limit in limits):
        raise ConfigError(field + '.endpoint_limits', 'Every endpoint requires an explicit budget and resource set')
    group_models = {g.group_id: builtins.entity_models[g.model_id] for g in request.scenario.entity_groups}
    can_transmit = [group_models[p.group_id].metadata['radio_devices'][p.device_id]['can_transmit'] for p in points]
    owners, controls, emitters, passive = {}, [], [], []

    def own(indices, owner, path, transmitting):
        for index in indices:
            if index in owners:
                raise ConfigError(path, 'Multiple RF owners for one physical device')
            if transmitting and not can_transmit[index]:
                raise ConfigError(path, 'Receive-only endpoint must use passive RF owner')
            owners[index] = owner

    for role, name in selection.role_controls.items():
        path = field + '.role_controls.' + role
        indices = collection(name, path)
        own(indices, 'role:' + role, path, True)
        binding = bindings.get(role)
        if binding is None:
            raise ConfigError(path, 'Unknown role')
        if [(points[i].group_id, points[i].slot) for i in indices] != [
                (binding.group_id, slot) for slot in binding.role_slot_to_group_slot]:
            raise ConfigError(path, 'Radio endpoints must match role group and ordered slots')
        if any(limits[i] != limits[indices[0]] for i in indices):
            raise ConfigError(path, 'Role slots require a common power budget and ordered resource set')
        controls.append(ControlBinding(role_id=role, endpoint_indices=indices))

    minimum_dwell = min(selection.epoch_max_ticks, request.scenario.scene.control_substeps)
    seen_emitters = set()
    for emitter in selection.background_emitters:
        path = field + '.background_emitters.' + emitter.emitter_id
        if emitter.emitter_id in seen_emitters:
            raise ConfigError(path, 'Duplicate emitter ID')
        seen_emitters.add(emitter.emitter_id)
        indices = tuple(i for name in emitter.endpoints for i in collection(name, path))
        own(indices, 'emitter:' + emitter.emitter_id, path, True)
        config = emitter.config
        if emitter.behavior == 'periodic':
            durations = (config.on_ticks, config.off_ticks)
            if config.phase_ticks >= sum(durations):
                raise ConfigError(path, 'Initial phase must be inside one on/off period')
        elif emitter.behavior in ('sweep', 'markov'):
            durations = (config.dwell_ticks,)
        else:
            durations = ()
        if any(ticks < minimum_dwell for ticks in durations):
            raise ConfigError(path, 'Emitter dwell is shorter than epoch grid resolution')
        if emitter.behavior == 'sweep':
            if config.initial_index >= len(config.channel_ids):
                raise ConfigError(path, 'Initial sweep index is out of range')
            states = [(config.active, config.power_w, [channel] * len(indices)) for channel in config.channel_ids]
        elif emitter.behavior == 'markov':
            states = [(state.active, state.power_w, state.channel_id) for state in config.states]
        elif emitter.behavior == 'fixed':
            states = [(config.active, config.power_w, config.channel_id)]
        else:
            states = [([True] * len(indices), config.power_w, config.channel_id)]
        for active, power, channel in states:
            if any(len(values) != len(indices) for values in (active, power, channel)):
                raise ConfigError(path, 'Emitter vectors must match expanded endpoint count')
            for i, value, channel_id in zip(indices, power, channel):
                if value > limits[i].max_power_w or channel_id not in [channels[f] for f in limits[i].resource_indices]:
                    raise ConfigError(path, 'Emitter state exceeds endpoint power/resource budget')
        emitters.append(EmitterBinding(emitter_id=emitter.emitter_id, endpoint_indices=indices,
            behavior=emitter.behavior, config=config.model_copy(deep=True)))
    for name in selection.passive_endpoints:
        indices = collection(name, field + '.passive_endpoints')
        own(indices, 'none', field + '.passive_endpoints', False)
        passive.extend(indices)
    if len(owners) != len(points):
        raise ConfigError(field, 'Every endpoint needs exactly one explicit RF owner, including passive none')
    for role, binding in bindings.items():
        has_radio = 'radio' in binding.action_interfaces
        if has_radio != (role in selection.role_controls) or has_radio and binding.action_interfaces['radio'] != 'power_channel':
            raise ConfigError(field + '.role_controls', 'Radio action interface and explicit role control must match')

    desired = () if geometry is None else geometry.layout.desired
    paths = () if geometry is None else geometry.layout.paths
    allocations, allocated, seen_links = [], {}, set()
    for allocation in selection.communication_allocations:
        path = field + '.communication_allocations.' + allocation.link_id
        fs = resource_indices(allocation.resource_ids, path)
        ls = tuple(i for i, item in enumerate(desired) if item.link_id == allocation.link_id)
        if not ls or allocation.link_id in seen_links:
            raise ConfigError(path, 'Expected a distinct declared communication link')
        seen_links.add(allocation.link_id)
        for l in ls:
            p = desired[l].path
            tx, rx = paths[p].tx, paths[p].rx
            for f in fs:
                key = (tx, f)
                if key in allocated:
                    raise ConfigError(path, 'Duplicate physical tx/resource allocation to an L slot')
                if f not in limits[tx].resource_indices:
                    raise ConfigError(path, 'Allocation exceeds transmitter resource set')
                allocated[key] = l
            allocations.append(CommunicationBinding(link_index=l, tx=tx, rx=rx, path=p, resource_indices=fs))
    services, consumed, service_ids = [], set(), set()
    for service in selection.services:
        path = field + '.services.' + service.service_id
        fs = resource_indices(service.resource_ids, path)
        ls = tuple(i for i, item in enumerate(desired) if item.link_id == service.link_id)
        if not ls or service.service_id in service_ids:
            raise ConfigError(path, 'Expected declared link and unique service ID')
        service_ids.add(service.service_id)
        for l in ls:
            propagation = paths[desired[l].path]
            for f in fs:
                if allocated.get((propagation.tx, f)) != l:
                    raise ConfigError(path, 'Service must be a communication allocation subset')
                key = (propagation.tx, propagation.rx, f)
                if key in consumed:
                    raise ConfigError(path, 'Duplicate physical service consumer')
                consumed.add(key)
        services.append(ServiceBinding(service_id=service.service_id, link_indices=ls, resource_indices=fs, initial_bits=service.initial_bits))
    d, k = request.scenario.scene.control_substeps, selection.epoch_max_ticks
    spec = NetworkSpec(endpoints=points, endpoint_sets=sets, endpoint_to_point=tuple(range(len(points))),
        resources=tuple(resources), endpoint_limits=tuple(limits), role_controls=tuple(controls),
        background_emitters=tuple(emitters), passive_endpoints=tuple(passive), communication_allocations=tuple(allocations),
        services=tuple(services), epoch_offsets=(*range(0, d, k), d), physics_dt=request.scenario.scene.physics_dt,
        link_count=len(desired), sensing_count=0 if geometry is None else len(geometry.layout.sensing),
        point_count=0 if geometry is None else len(geometry.layout.points), path_count=len(paths))
    axes = network_axes(spec, geometry=geometry, batch_size=request.topology.world_count)
    models = {}

    if 'sensing' in reasons:
        _, sensing = model_descriptor('sensing')
        if any(item.path[:2] == ['network', 'radio'] for item in sensing.input_fields):
            reasons.setdefault('radio', []).append('model:sensing')
    for kind, override in request.model_parameters.items():
        if kind not in reasons:
            reference, meta = model_descriptor(kind)
            resolve_config(meta.config_schema, composition.model_bindings[kind].options[reference].config,
                           override, require_complete=False)
    for kind in ('radio', 'sensing'):
        if kind not in reasons:
            continue
        reference, meta = model_descriptor(kind)
        if any(item.path[:2] == ['network', 'geometry'] for item in meta.input_fields):
            reasons.setdefault('geometry', []).append('model:' + kind)
        if kind == 'radio' and not spec.link_count:
            raise ConfigError('model_bindings.radio', 'Radio model requires a positive L axis')
        if kind == 'radio' and services and 'rate' not in meta.output_spec.output_blocks:
            raise ConfigError('model_bindings.radio', 'Service requires a radio rate output block')
        if kind == 'radio' and meta.frequency_hz_schema is not None:
            validator = Draft202012Validator(meta.frequency_hz_schema)
            for index, resource in enumerate(spec.resources):
                try:
                    validator.validate(resource.center_frequency_hz)
                except ValidationError as exc:
                    raise ConfigError(f'{field}.resources.{index}.center_frequency_hz', exc.message) from exc
        available = {}
        for phase in ('reset', 'post_physics'):
            for item in network_model_input_fields(spec, geometry=geometry,
                    batch_size=request.topology.world_count, phase=phase):
                available[(tuple(item.path), phase)] = item
            if kind == 'sensing' and 'radio' in models:
                for block in models['radio'].descriptor.output_spec.output_blocks:
                    for name, units in RADIO_FIELDS[block].items():
                        item = InputField(path=['network', 'radio', block, name], phase=phase,
                            tensor=dict(shape=[axes['B'], axes['L'], axes['F']], dtype='bool' if name == 'valid' else 'float32',
                                units=units, frame='none', rate='radio_epoch'))
                        available[(tuple(item.path), phase)] = item
        inputs = []
        for item in meta.input_fields:
            actual = available.get((tuple(item.path), item.phase))
            expected = _shape(item.tensor, axes, 'model_bindings.' + kind + '.input_fields')
            if actual is None or expected != actual.tensor:
                raise ConfigError('model_bindings.' + kind + '.input_fields', f'Unavailable or mismatched model input: {item.path}')
            inputs.append(actual.model_copy(deep=True))
        for schema in (meta.state_schema, meta.diagnostics_schema):
            for name, tensor in schema.items():
                _shape(tensor, axes, 'model_bindings.' + kind + '.' + name)
        if kind == 'sensing':
            for name in ('measurement', 'fim'):
                tensor = getattr(meta.output_spec, name)
                if tensor is not None:
                    _shape(tensor, axes, 'model_bindings.sensing.' + name)
        config = resolve_config(meta.config_schema, composition.model_bindings[kind].options[reference].config,
                                request.model_parameters.get(kind, {}), require_complete=True)
        models[kind] = BoundModel(reference=reference, config=config, descriptor=meta.model_copy(deep=True), input_fields=inputs)
    return spec, models, {key: list(dict.fromkeys(value)) for key, value in reasons.items()}
