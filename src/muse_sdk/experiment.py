"""普通飞行的CPU请求、资源与冻结运行契约。"""
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from .config import DataModel
from .descriptor import InputField, PackageDescriptor, TensorSpec, TrainingSchedule
from .selection import ResolvedSelection
from .geometry import EnvironmentRF, GeometrySelection, RadioDeviceDefinition, ResolvedGeometry
from .network import BoundModel, NetworkSelection, NetworkSpec
from .run_protocol import CheckpointReference, RunTimeouts


class ResourceRef(DataModel):
    resource_id: str
    revision: int = Field(gt=0)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class ResourceRecord(ResourceRef):
    kind: Literal['environment', 'training_package', 'entity_model']
    path: str
    metadata: dict


class EntityModelRef(ResourceRef):
    physics_asset: str
    root_body: str
    support_geoms: list[str]
    radio_devices: dict[str, RadioDeviceDefinition] = Field(default_factory=dict)


class EnvironmentBundle(DataModel):
    scene: str
    metadata: dict = Field(default_factory=dict)
    rf_geometry: EnvironmentRF | None = None


class Pose(DataModel):
    position: list[float] = Field(min_length=3, max_length=3)
    orientation_wxyz: list[float] = Field(min_length=4, max_length=4)


class EntityGroup(DataModel):
    group_id: str
    model_id: str
    model_revision: int = Field(gt=0)
    count: int = Field(gt=0)
    placement_id: str
    controller_id: str
    sensor_profile_id: str
    sensor_config: dict = Field(default_factory=dict)


class SceneConfig(DataModel):
    physics_dt: float = Field(gt=0)
    control_substeps: int = Field(gt=0)
    integrator: Literal['RK4']
    density: float = Field(ge=0)
    viscosity: float = Field(ge=0)
    precision: Literal['float32']


class ScenarioDefinition(DataModel):
    entity_groups: list[EntityGroup] = Field(min_length=1)
    placements: dict[str, list[Pose]]
    scene: SceneConfig
    command_targets_world: dict[str, list[list[float]]]
    radio_geometry: GeometrySelection | None = None
    network: NetworkSelection | None = None


class RoleConfig(DataModel):
    group_id: str
    slots: list[int] = Field(min_length=1)


class Topology(DataModel):
    world_count: int = Field(gt=0)


class Preview(DataModel):
    world_ids: list[int] = Field(min_length=1, max_length=1)
    rate_hz: float = Field(ge=1, le=30)


class ExperimentRequest(DataModel):
    model_config = ConfigDict(serialize_by_alias=True)

    environment_id: str
    environment_revision: int = Field(gt=0)
    training_package_id: str
    training_package_revision: int = Field(gt=0)
    scenario: ScenarioDefinition
    composition_id: str = Field(min_length=1)
    task_config: dict = Field(default_factory=dict)
    role_config: dict[str, RoleConfig]
    term_config: dict = Field(default_factory=dict)
    cost_config: dict = Field(default_factory=dict)
    model_parameters: dict = Field(default_factory=dict, alias='model_config')
    model_choices: dict[Literal['radio', 'sensing'], str] = Field(default_factory=dict)
    algorithm_config: dict = Field(default_factory=dict)
    topology: Topology
    root_seed: int = Field(ge=0)
    preview: Preview


class RadioActionSpec(DataModel):
    """功率单位为 W；有序 ID 限于 int32/float32 均可精确表示的整数区间。"""
    power_minimum_w: float = Field(ge=0, le=3.4028234663852886e38)
    power_maximum_w: float = Field(gt=0, le=3.4028234663852886e38)
    channel_ids: list[Annotated[int, Field(ge=-(2**24), le=2**24)]] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_radio(self):
        if self.power_minimum_w >= self.power_maximum_w:
            raise ValueError('Radio power minimum must be less than maximum')
        if len(set(self.channel_ids)) != len(self.channel_ids):
            raise ValueError('Radio channel IDs must be unique')
        return self


class ActionSpec(DataModel):
    interface: Literal['body_velocity', 'diff_drive']
    dtype: Literal['float32']
    shape: list[int]
    fields: list[str]
    units: list[str]
    frames: list[str]
    minimum: list[float]
    maximum: list[float]
    radio: RadioActionSpec | None = None


class EntityBindingPlan(DataModel):
    role_id: str
    group_id: str
    model_id: str
    entity_kind: str
    role_slot_count: int
    role_slot_to_group_slot: list[int]
    action_interfaces: dict[str, str]
    controller_id: str
    sensor_profile_id: str
    network_profile_id: None = None


class BoundSelection(ResolvedSelection):
    input_fields: list[InputField]
    output_spec: TensorSpec | None
    time_out: bool
    cost_id: str | None


class ActionSlice(DataModel):
    role_id: str
    field: Literal['motion', 'radio']
    start: int
    stop: int
    shape: list[int]


class AlgorithmInputs(DataModel):
    observations: list[InputField]
    actors: dict[str, list[InputField]]
    critics: dict[str, list[InputField]]


class BoundTask(DataModel):
    """Task 构造及输出边界实际消费的已绑定声明。"""
    input_fields: list[InputField]
    published_fields: dict[str, TensorSpec]
    learning_done_roles: list[str]


class CostConstraint(DataModel):
    discount: float = Field(ge=0, le=1)
    budget: float = Field(ge=0)
    lambda_initial: float = Field(ge=0)
    lambda_learning_rate: float = Field(gt=0)


class ResolvedCost(CostConstraint):
    cost_id: str = Field(min_length=1)
    scope: Literal['role', 'world']
    units: str
    statistic: Literal['episode_discounted_sum']
    boundary: Literal['world_or_learning_done', 'world']


class CostLayout(DataModel):
    roles: dict[str, dict[str, ResolvedCost]]
    world: dict[str, ResolvedCost]

    @model_validator(mode='after')
    def check_owners(self):
        for scope, owners in (('role', self.roles.values()), ('world', (self.world,))):
            boundary = 'world_or_learning_done' if scope == 'role' else 'world'
            for costs in owners:
                for cost_id, cost in costs.items():
                    if cost.cost_id != cost_id or cost.scope != scope or cost.boundary != boundary:
                        raise ValueError('Cost identity/scope/boundary does not match its owner')
        return self


class ExperimentDefinition(DataModel):
    geometry: ResolvedGeometry | None = None
    network: NetworkSpec | None = None
    environment: ResourceRef
    environment_scene: str
    environment_metadata: dict
    training_package: ResourceRef
    entity_models: dict[str, EntityModelRef]
    platform_revision: str
    scenario: ScenarioDefinition
    bindings: dict[str, EntityBindingPlan]
    controllers: dict[str, dict]
    sensors: dict[str, dict]
    composition_id: str
    task_id: str | None
    task_config: dict
    task_spec: BoundTask | None = None
    algorithm_id: str
    algorithm_config: dict
    training_schedule: TrainingSchedule | None
    algorithm_inputs: AlgorithmInputs
    role_inputs: dict[str, list[InputField]]
    selections: list[BoundSelection]
    termination_reasons: dict[int, tuple[str, str, str]]
    action_specs: dict[str, ActionSpec]
    action_layout: list[ActionSlice]
    activation: dict[str, list[str]]
    model_bindings: dict[str, BoundModel]
    costs: CostLayout
    topology: Topology
    root_seed: int
    preview: Preview


class ValidationReport(DataModel):
    valid: bool
    resolved_experiment: ExperimentDefinition | None
    errors: list[dict[str, str]]
    warnings: list[dict[str, str]]


def learning_contract(experiment: ExperimentDefinition) -> dict:
    """保留有序学习语义；仅去掉已知tensor/action布局的运行batch轴。"""
    data = experiment.model_dump(mode='json')

    def feature_spec(spec):
        return {**spec, 'shape': spec['shape'][1:]}

    def inputs(fields):
        return [{**field, 'tensor': field['tensor'] if field['path'][:2] == ['network', 'resources']
                 else feature_spec(field['tensor'])} for field in fields]

    contract = {key: data[key] for key in (
        'platform_revision', 'training_package', 'algorithm_id', 'algorithm_config',
        'training_schedule', 'task_id', 'task_config', 'costs', 'model_bindings', 'activation')}
    contract['bindings'] = list(data['bindings'].items())
    contract['network'] = data['network']
    if data['network'] is not None:
        contract['geometry'] = data['geometry']
    contract['model_bindings'] = {
        kind: {**bound, 'input_fields': inputs(bound['input_fields'])}
        for kind, bound in data['model_bindings'].items()}
    contract['action_specs'] = [(role, feature_spec(spec)) for role, spec in data['action_specs'].items()]
    contract['action_layout'] = [feature_spec(item) for item in data['action_layout']]
    algorithm_inputs = data['algorithm_inputs']
    contract['algorithm_inputs'] = {
        'observations': inputs(algorithm_inputs['observations']),
        **{key: [(role, inputs(fields)) for role, fields in algorithm_inputs[key].items()]
           for key in ('actors', 'critics')},
    }
    contract['role_inputs'] = [(role, inputs(fields)) for role, fields in data['role_inputs'].items()]
    task = data['task_spec']
    contract['task_spec'] = None if task is None else {
        **task, 'input_fields': inputs(task['input_fields']),
        'published_fields': {name: feature_spec(spec) for name, spec in task['published_fields'].items()},
    }
    contract['selections'] = [
        {**item, 'input_fields': inputs(item['input_fields']),
         'output_spec': feature_spec(item['output_spec'])}
        for item in data['selections'] if item['enabled']]
    sensors = [
        {'group_id': group, **{key: profile[key] for key in (
            'sensor_profile_id', 'outputs')}}
        for group, profile in data['sensors'].items() if 'sensor:' + group in data['activation']]
    contract['sensors'] = {} if not sensors else {
        'physics_dt': data['scenario']['scene']['physics_dt'],
        'control_substeps': data['scenario']['scene']['control_substeps'], 'groups': sensors}
    return contract


class RunPlan(DataModel):
    run_id: str
    experiment_id: str
    mode: Literal['TRAIN', 'EVALUATE', 'INTERACTIVE']
    device: str
    platform_revision: str
    artifact_directory: str
    resource_paths: dict[str, str]
    experiment: ExperimentDefinition
    max_decision_steps: int = Field(default=24000, gt=0, description='本次运行的决策步数上限；一次同时推进全部world，不乘world数，不计内部物理子步')
    timeouts: RunTimeouts = Field(default_factory=RunTimeouts)
    checkpoint: CheckpointReference | None = None
    checkpoint_interval_updates: int | None = Field(default=None, gt=0)


class BuiltinMetadata(DataModel):
    platform_revision: str
    descriptor: PackageDescriptor
    entity_models: dict[str, ResourceRecord]
    controllers: dict[str, dict]
    sensors: dict[str, dict]


class CatalogSnapshot(DataModel):
    resources: list[ResourceRecord]
    builtins: BuiltinMetadata
