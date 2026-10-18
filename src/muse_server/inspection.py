"""部分草稿的schema/来源视图；完整运行语义仍交唯一resolver。"""
from copy import deepcopy

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from muse_sdk.config import ConfigError, check_json, resolve_config_layers
from muse_sdk.descriptor import iter_selections, _reference
from muse_sdk.editor import InspectionReport, InspectionSection
from muse_sdk.experiment import CostConstraint, ExperimentRequest, Preview, RoleConfig, ScenarioDefinition, Topology

from .resolver import _package_context, _resolve_flight, _selection_overrides


def inspect_experiment(request: dict, catalog_snapshot) -> InspectionReport:
    check_json(request)
    sections = []
    resolved = None
    invalid_fields = set()
    try:
        data = ExperimentRequest.model_validate(request)
        errors = []
    except ValidationError as exc:
        data = None
        errors = [{'field': '.'.join(map(str, error['loc'])), 'message': error['msg']} for error in exc.errors()]
        invalid_fields = {error['loc'][0] for error in exc.errors() if error['loc']}

    def section(path, schema, layers, *, required=True):
        try:
            effective, origins = resolve_config_layers(schema, layers, require_complete=False)
            field_errors = []
        except ConfigError as exc:
            # HTTP输入结构在下方分区边界检查；此处只接收参数schema错误。
            effective, origins, field_errors = exc.effective, exc.origins, exc.errors
        pending = list(Draft202012Validator(schema).iter_errors(effective)) if required else []
        missing = []
        while pending:
            error = pending.pop()
            pending.extend(error.context)
            if error.validator == 'required':
                for name in error.validator_value:
                    if name not in error.instance:
                        pointer = '/' + '/'.join(str(key).replace('~', '~0').replace('/', '~1')
                                                  for key in (*error.absolute_path, name))
                        if pointer not in missing:
                            missing.append(pointer)
        sections.append(InspectionSection(path=list(path), schema=deepcopy(schema), effective=effective,
            origins=origins, missing=missing, errors=field_errors))
        return sections[-1]

    def object_input(container, key, path):
        value = container.get(key, {})
        if not isinstance(value, dict):
            raise ConfigError('.'.join(map(str, (*path, key))), 'Expected object')
        return value

    # 固定的场景/运行输入使用SDK DTO schema；不拼造缺失Scenario。
    for key, dto in (('scenario', ScenarioDefinition), ('topology', Topology), ('preview', Preview)):
        if key in request and isinstance(request[key], dict):
            section((key,), dto.model_json_schema(), [('scenario', request[key])])
    package_keys = ('training_package_id', 'training_package_revision', 'composition_id')
    if any(key not in request or key in invalid_fields for key in package_keys):
        return InspectionReport(sections=sections, errors=errors, resolved_experiment=None)
    try:
        context = _package_context(catalog_snapshot, request['training_package_id'],
                                    request['training_package_revision'], request['composition_id'])
        package, descriptor, composition = context
        builtins = catalog_snapshot.builtins
        if data is not None:
            try:
                resolved = _resolve_flight(data, catalog_snapshot, context)
            except ConfigError as exc:
                errors.extend(exc.errors)
            except ValidationError as exc:
                errors.extend({'field': '.'.join(map(str, error['loc'])), 'message': error['msg']}
                              for error in exc.errors())
        for kind in ('task', 'algorithm'):
            reference = getattr(composition, kind)
            if reference is not None:
                meta = _reference(reference, getattr(descriptor, kind + 's'), kind,
                                  getattr(builtins.descriptor, kind + 's'))
                key = kind + '_config'
                section((key,), meta.config_schema, [('package', getattr(composition, key)),
                    ('override', object_input(request, key, ()))])

        overrides = _selection_overrides(composition, object_input(request, 'term_config', ()))
        for owner, kind, selections in iter_selections(composition):
            for item in selections:
                identity = (*owner, kind, item.selection_id)
                meta = _reference(item.term, descriptor.terms, '.'.join(identity), builtins.descriptor.terms)
                override = overrides.get(identity)
                enabled = item.default_enabled if override is None or override.enabled is None else override.enabled
                section(('term_config', *identity, 'config'), meta.config_schema,
                    [('package', item.config), ('override', {} if override is None else override.config)], required=enabled)

        choices = object_input(request, 'model_choices', ())
        model_config = object_input(request, 'model_config', ())
        for kind, binding in composition.model_bindings.items():
            if 'model_choices' in invalid_fields:
                continue
            reference = choices.get(kind, binding.default)
            if reference not in binding.options:
                raise ConfigError('model_choices.' + kind, 'Model choice must be declared by the selected composition')
            meta = _reference(reference, getattr(descriptor, kind + '_models'), 'model_choices.' + kind,
                              getattr(builtins.descriptor, kind + '_models'))
            section(('model_config', kind), meta.config_schema,
                [('package', binding.options[reference].config), ('override', object_input(model_config, kind, ('model_config',)))],
                required=resolved is not None and kind in resolved.model_bindings)

        roles = object_input(request, 'role_config', ())
        for role in composition.agent_roles:
            section(('role_config', role), RoleConfig.model_json_schema(),
                    [('override', object_input(roles, role, ('role_config',)))])
        scenario = object_input(request, 'scenario', ())
        groups = scenario.get('entity_groups', [])
        if isinstance(groups, list):
            for index, group in enumerate(groups):
                if not isinstance(group, dict) or group.get('sensor_profile_id') not in builtins.sensors:
                    continue
                profile = builtins.sensors[group['sensor_profile_id']]
                sensor_section = section(('scenario', 'entity_groups', index, 'sensor_config'), profile['config_schema'],
                    [('scenario', object_input(group, 'sensor_config', ('scenario', 'entity_groups', index)))])
                if resolved is not None:
                    sensor_section.effective = deepcopy(resolved.sensors[group['group_id']]['config'])
        constraints = object_input(request, 'cost_config', ())
        for owner, kind, selections in iter_selections(composition):
            if kind != 'cost':
                continue
            values = constraints
            for part in owner:
                values = object_input(values, part, ('cost_config',))
            for item in selections:
                meta = _reference(item.term, descriptor.terms, item.term, builtins.descriptor.terms)
                override = overrides.get((*owner, kind, item.selection_id))
                enabled = item.default_enabled if override is None or override.enabled is None else override.enabled
                section(('cost_config', *owner, meta.cost_id), CostConstraint.model_json_schema(),
                    [('override', object_input(values, meta.cost_id, ('cost_config', *owner)))], required=enabled)
    except (ConfigError, ValidationError) as exc:
        extra = exc.errors if isinstance(exc, ConfigError) else [
            {'field': '.'.join(map(str, error['loc'])), 'message': error['msg']} for error in exc.errors()]
        errors.extend(error for error in extra if error not in errors)
    return InspectionReport(sections=sections, errors=errors, resolved_experiment=resolved)
