"""平台与包共享的CPU descriptor结构和静态组合校验。"""
from copy import deepcopy
import inspect
import re
from typing import Literal

from pydantic import BaseModel, Field, model_serializer, model_validator

from .config import ConfigError, DataModel, check_json, export_config_schema, resolve_config, validate_config_schema
from .registration import TrainingPackageDefinition
from .models import MODEL_INPUT_VIEWS, RADIO_FIELDS, RadioModel, SensingModel


class TensorSpec(DataModel):
    shape: list[int | str]
    dtype: Literal['float32', 'float64', 'int32', 'int64', 'bool']
    units: str
    frame: str
    rate: str | float


class RoleBinding(DataModel):
    binding: Literal['role_id']


class GroupBinding(DataModel):
    binding: Literal['group_id']


class InputField(DataModel):
    path: list[str | RoleBinding | GroupBinding] = Field(min_length=2)
    phase: Literal['reset', 'action', 'post_physics', 'post_task', 'post_observation', 'next_command']
    tensor: TensorSpec

    @model_validator(mode='after')
    def validate_path(self):
        roots = {'entities', 'physics_outputs', 'sensor_outputs', 'network', 'sensing_outputs',
                 'role_actions', 'role_masks', 'act_role_masks', 'act_commands', 'task',
                 'observations', 'role_positions_world', 'commands'}
        if not isinstance(self.path[0], str) or self.path[0] not in roots:
            raise ValueError('Unknown public input root')
        if any(isinstance(part, str) and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', part) for part in self.path):
            raise ValueError('Input path segments must be nonempty identifiers')
        return self


class TrainingSchedule(DataModel):
    collection_steps: int = Field(gt=0)


class TrainingScheduleBinding(DataModel):
    """指向算法配置的顶层整数参数；不接受表达式或隐藏默认值。"""
    collection_steps: str = Field(min_length=1)


class AlgorithmDescriptor(DataModel):
    algorithm_id: str = Field(min_length=1)
    config_schema: dict
    observation_requirements: list[InputField]
    action_requirements: list[str]
    actor_observation: dict[str, list[InputField]] | Literal['all_local']
    critic_inputs: dict[str, list[InputField]] | Literal['all_centralized']
    reward_inputs: list[Literal['role', 'world']]
    cost_inputs: list[Literal['role', 'world']] = Field(default_factory=list)
    act_aux: dict[str, TensorSpec]
    diagnostics: dict[str, TensorSpec]
    training_schedule: TrainingScheduleBinding | None

    @model_validator(mode='after')
    def check_schedule_binding(self):
        if self.training_schedule is not None:
            name = self.training_schedule.collection_steps
            parameter = self.config_schema.get('properties', {}).get(name, {})
            if parameter.get('type') != 'integer':
                raise ValueError('training_schedule.collection_steps must reference an integer config field')
        return self


class TaskDescriptor(DataModel):
    task_id: str = Field(min_length=1)
    config_schema: dict
    input_fields: list[InputField]
    published_fields: dict[str, TensorSpec]
    role_status_bindings: dict[str, list[Literal['alive', 'controllable']]]
    learning_done_roles: list[str]


class RadioOutputSpec(DataModel):
    output_blocks: list[Literal['link', 'interference', 'rate']] = Field(min_length=1)

    @model_validator(mode='after')
    def unique_blocks(self):
        if len(set(self.output_blocks)) != len(self.output_blocks):
            raise ValueError('Duplicate radio output block')
        return self


class MeasurementSpec(TensorSpec):
    time_semantics: Literal['snapshot', 'increment', 'interval_sample']


class SensingOutputSpec(DataModel):
    measurement: MeasurementSpec | None = None
    fim: MeasurementSpec | None = None

    @model_validator(mode='after')
    def require_output(self):
        if self.measurement is None and self.fim is None:
            raise ValueError('Sensing model must declare measurement or fim')
        return self


class ModelDescriptor(DataModel):
    model_id: str = Field(min_length=1)
    config_schema: dict
    input_fields: list[InputField]
    state_scope: Literal['world']
    state_schema: dict[str, TensorSpec]
    diagnostics_schema: dict[str, TensorSpec] = Field(default_factory=dict)
    assumptions: dict

    @model_validator(mode='after')
    def model_contract(self):
        for schema in (self.state_schema, self.diagnostics_schema):
            for spec in schema.values():
                if not spec.shape or spec.shape[0] != 'B':
                    raise ValueError('Model state/diagnostics require leading B axis')
        seen, paths = set(), {}
        for item in self.input_fields:
            if not all(isinstance(part, str) for part in item.path):
                raise ValueError('Model inputs require concrete paths')
            path, phase = tuple(item.path), item.phase
            if phase not in ('reset', 'post_physics') or (path, phase) in seen:
                raise ValueError('Invalid or duplicate model input phase')
            if path in paths and paths[path] != item.tensor:
                raise ValueError('Model input specification differs between phases')
            seen.add((path, phase))
            paths[path] = item.tensor
            if len(path) == 3 and path[0] == 'network' and path[1] in MODEL_INPUT_VIEWS:
                continue
            if path in (('physics_outputs', 'sim_time'), ('network', 'interval_s')):
                expected = TensorSpec(shape=['B'], dtype='float64', units='s', frame='none', rate='physics')
            elif (isinstance(self, SensingDescriptor) and len(path) == 4 and path[:2] == ('network', 'radio')
                  and path[2] in RADIO_FIELDS and path[3] in RADIO_FIELDS[path[2]]):
                expected = TensorSpec(shape=['B', 'L', 'F'], dtype='bool' if path[3] == 'valid' else 'float32',
                    units=RADIO_FIELDS[path[2]][path[3]], frame='none', rate='radio_epoch')
            else:
                raise ValueError('Unsupported model input path')
            if item.tensor != expected:
                raise ValueError('Model input differs from fixed field contract')
        return self


class RadioDescriptor(ModelDescriptor):
    output_spec: RadioOutputSpec
    frequency_hz_schema: dict | None = None

    @model_validator(mode='after')
    def frequency_domain(self):
        if self.frequency_hz_schema is not None:
            validate_config_schema(self.frequency_hz_schema)
            if self.frequency_hz_schema.get('type') != 'number':
                raise ValueError('frequency_hz_schema requires type="number"')
        return self


class SensingDescriptor(ModelDescriptor):
    output_spec: SensingOutputSpec

    @model_validator(mode='after')
    def sensing_fields(self):
        for name in ('measurement', 'fim'):
            spec = getattr(self.output_spec, name)
            if spec is None:
                continue
            if not spec.shape or spec.shape[0] != 'B' or spec.dtype not in ('float32', 'float64'):
                raise ValueError('Sensing output must be floating point with leading B axis')
            if name == 'fim' and (len(spec.shape) < 3 or spec.shape[-1] != spec.shape[-2]
                    or isinstance(spec.shape[-1], int) and spec.shape[-1] <= 0):
                raise ValueError('FIM requires positive square matrix axes')
            if name == 'fim' and spec.time_semantics == 'interval_sample':
                raise ValueError('FIM cannot use interval_sample')
        return self


class TermDescriptor(DataModel):
    term_kind: Literal['observation', 'reward', 'cost', 'termination']
    scope: Literal['role', 'world']
    config_schema: dict
    input_fields: list[InputField]
    output_spec: TensorSpec
    cost_id: str | None = None
    time_out: bool = False

    @model_validator(mode='after')
    def check_semantics(self):
        if (self.term_kind == 'cost') != (self.cost_id is not None):
            raise ValueError('cost_id is required only for cost terms')
        if self.cost_id == '':
            raise ValueError('cost_id cannot be empty')
        if self.term_kind == 'cost' and self.output_spec.dtype != 'float32':
            raise ValueError('Cost terms must return float32')
        if self.time_out and self.term_kind != 'termination':
            raise ValueError('time_out only applies to termination')
        if self.term_kind == 'termination' and (self.scope != 'world' or self.output_spec.dtype != 'bool'):
            raise ValueError('termination must return world bool')
        prefix = ['B', 'N_role'] if self.scope == 'role' else ['B']
        if self.output_spec.shape[:len(prefix)] != prefix:
            raise ValueError('Output shape prefix does not match scope')
        if self.term_kind != 'observation' and self.output_spec.shape != prefix:
            raise ValueError('Scalar contribution required for reward/cost/termination')
        return self


class TermSelection(DataModel):
    selection_id: str = Field(min_length=1)
    term: str = Field(min_length=1)
    required: bool = False
    default_enabled: bool = True
    config: dict = Field(default_factory=dict)
    weight: float | None = None
    role_id: str | None = None
    history_length: int = Field(default=0, ge=0)

    @model_serializer(mode='wrap')
    def serialize_selection(self, handler):
        data = handler(self)
        # 省略未声明字段，不能把absence在JSON边界变成显式null。
        for key in ('weight', 'role_id', 'history_length'):
            if key not in self.model_fields_set:
                data.pop(key, None)
        return data


class RoleComposition(DataModel):
    entity_kind: str
    action_interfaces: dict[str, str]
    input_requirements: list[dict | str] = Field(default_factory=list)
    observation_terms: list[TermSelection] = Field(default_factory=list)
    reward_terms: list[TermSelection] = Field(default_factory=list)
    cost_terms: list[TermSelection] = Field(default_factory=list)


class CriticComposition(DataModel):
    observation_terms: list[TermSelection] = Field(default_factory=list)


class Composition(DataModel):
    task: str | None = None
    algorithm: str | None = None
    task_config: dict = Field(default_factory=dict)
    algorithm_config: dict = Field(default_factory=dict)
    agent_roles: dict[str, RoleComposition] = Field(default_factory=dict)
    critic_inputs: dict[str, CriticComposition] = Field(default_factory=dict)
    world_reward_terms: list[TermSelection] = Field(default_factory=list)
    world_cost_terms: list[TermSelection] = Field(default_factory=list)
    world_termination_terms: list[TermSelection] = Field(default_factory=list)
    model_bindings: dict[str, str] = Field(default_factory=dict)


class PackageDescriptor(DataModel):
    sdk_version: Literal['0.1'] = '0.1'
    package_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    terms: dict[str, TermDescriptor]
    tasks: dict[str, TaskDescriptor]
    algorithms: dict[str, AlgorithmDescriptor]
    radio_models: dict[str, RadioDescriptor]
    sensing_models: dict[str, SensingDescriptor]
    composition: Composition


def parse_descriptor_json(content: str, builtin_catalog: PackageDescriptor | None = None) -> PackageDescriptor:
    """不可信序列化数据的单一入口；返回可直接用于解析的已验证descriptor。"""
    descriptor = PackageDescriptor.model_validate_json(content)
    for entries in (descriptor.terms, descriptor.tasks, descriptor.algorithms,
                    descriptor.radio_models, descriptor.sensing_models):
        for entry in entries.values():
            validate_config_schema(entry.config_schema)
    validate_composition(descriptor, builtin_catalog)
    return descriptor


def iter_selections(composition: Composition):
    for role, config in composition.agent_roles.items():
        for kind in ('observation', 'reward', 'cost'):
            yield ('roles', role), kind, getattr(config, kind + '_terms')
    for group, config in composition.critic_inputs.items():
        yield ('critic_inputs', group), 'observation', config.observation_terms
    for kind in ('reward', 'cost', 'termination'):
        yield ('world',), kind, getattr(composition, 'world_' + kind + '_terms')


def _reference(name: str, entries: dict, field: str, builtin_entries: dict | None = None):
    if name.startswith('builtin:'):
        if builtin_entries is None:
            raise ConfigError(field, 'An explicit platform catalog is required')
        name, entries = name.removeprefix('builtin:'), builtin_entries
    if name not in entries:
        raise ConfigError(field, f'Unknown reference {name}')
    return entries[name]


def validate_composition(descriptor: PackageDescriptor, builtin_catalog: PackageDescriptor | None = None) -> None:
    """只读校验：注册可缺实验必填参数，完整依赖闭包归M1。"""
    composition = descriptor.composition
    for kind, entries in (('task', descriptor.tasks), ('algorithm', descriptor.algorithms)):
        name = getattr(composition, kind)
        if name is not None:
            meta = _reference(name, entries, kind, getattr(builtin_catalog, kind + 's', None))
            resolve_config(meta.config_schema, getattr(composition, kind + '_config'), {}, require_complete=False)
    for kind, name in composition.model_bindings.items():
        if kind not in ('radio', 'sensing'):
            raise ConfigError('model_bindings', f'Unknown model kind {kind}')
        entries = descriptor.radio_models if kind == 'radio' else descriptor.sensing_models
        _reference(name, entries, 'model_bindings.' + kind, getattr(builtin_catalog, kind + '_models', None))
    for owner, kind, selections in iter_selections(composition):
        seen = set()
        for selection in selections:
            field = '.'.join((*owner, kind, selection.selection_id))
            if selection.selection_id in seen:
                raise ConfigError(field, 'Duplicate selection identity')
            seen.add(selection.selection_id)
            term = _reference(selection.term, descriptor.terms, field, getattr(builtin_catalog, 'terms', None))
            if term.term_kind != kind:
                raise ConfigError(field, 'Term kind mismatch')
            if owner[0] == 'roles' and term.scope != 'role' or owner[0] == 'world' and term.scope != 'world':
                raise ConfigError(field, 'Term scope mismatch')
            if selection.required and not selection.default_enabled:
                raise ConfigError(field, 'Required selection must be enabled by default')
            if kind != 'reward' and 'weight' in selection.model_fields_set:
                raise ConfigError(field, 'Only reward selections accept weight')
            if kind != 'observation' and 'history_length' in selection.model_fields_set:
                raise ConfigError(field, 'Only observation selections accept history_length')
            if kind == 'reward' and 'weight' in selection.model_fields_set and selection.weight is None:
                raise ConfigError(field, 'Reward weight cannot be null')
            if 'role_id' in selection.config:
                raise ConfigError(field, 'role_id is a platform binding, not business config')
            needs_binding = owner[0] == 'critic_inputs' and term.scope == 'role'
            if needs_binding:
                if selection.role_id not in composition.agent_roles:
                    raise ConfigError(field, 'Critic role observation requires a declared role_id')
            elif selection.role_id is not None:
                raise ConfigError(field, 'role_id only belongs to role-scoped critic selections')
            resolve_config(term.config_schema, selection.config, {}, require_complete=False)


def _describe(entries: dict[str, type], kind: str) -> dict:
    output = {}
    identity = 'model_id' if kind in ('radio', 'sensing') else kind + '_id'
    for name, implementation in entries.items():
        if not name or name.startswith('builtin:'):
            raise ConfigError(kind, 'Invalid package registration name')
        if kind in ('radio', 'sensing'):
            base = RadioModel if kind == 'radio' else SensingModel
            if not isinstance(implementation, type) or not issubclass(implementation, base) or inspect.isabstract(implementation):
                raise ConfigError(kind + '.' + name, f'Model must be a concrete {base.__name__}')
        if kind == 'algorithm' and not callable(getattr(implementation, 'reset_worlds', None)):
            raise ConfigError('algorithm.' + name, 'Algorithm must provide reset_worlds(mask); '
                              'stateless algorithms may inherit muse_sdk.interfaces.Algorithm')
        meta = dict(implementation.describe())
        if meta.get(identity) != name:
            raise ConfigError(kind + '.' + name, 'describe ID differs from registration')
        config = meta.get('config_schema')
        if not isinstance(config, type) or not issubclass(config, BaseModel):
            raise ConfigError(kind + '.' + name, 'describe config_schema must be a Python config model')
        meta['config_schema'] = export_config_schema(config)
        check_json(meta, kind + '.' + name)
        dto = {'task': TaskDescriptor, 'algorithm': AlgorithmDescriptor,
               'radio': RadioDescriptor, 'sensing': SensingDescriptor}[kind]
        output[name] = dto.model_validate(deepcopy(meta))
    return output


def build_descriptor(definition: TrainingPackageDefinition, composition: dict, package_hash: str,
                     builtin_catalog: PackageDescriptor | None = None) -> PackageDescriptor:
    terms = {}
    for name, registration in definition.terms.items():
        if not name or name.startswith('builtin:') or not callable(registration.implementation):
            raise ConfigError('terms', 'Invalid term name or callable')
        terms[name] = TermDescriptor(
            term_kind=registration.term_kind, scope=registration.scope,
            config_schema=export_config_schema(registration.config_model),
            input_fields=list(registration.input_fields), output_spec=registration.output_spec,
            cost_id=registration.cost_id, time_out=registration.time_out,
        )
    check_json(composition, 'composition')
    descriptor = PackageDescriptor(
        package_hash=package_hash, terms=terms, composition=deepcopy(composition),
        tasks=_describe(definition.tasks, 'task'), algorithms=_describe(definition.algorithms, 'algorithm'),
        radio_models=_describe(definition.radio_models, 'radio'), sensing_models=_describe(definition.sensing_models, 'sensing'),
    )
    validate_composition(descriptor, builtin_catalog)
    return descriptor
