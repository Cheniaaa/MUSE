"""声明式无线配置、CPU冻结布局和借用快照；不加载执行后端。"""
import math
from typing import Annotated, Any, Literal, NamedTuple

from pydantic import ConfigDict, Field, model_validator

from .config import DataModel
from .descriptor import InputField, RadioDescriptor, SensingDescriptor, TensorSpec
from .geometry import GeometryPoint, geometry_input_fields
from .models import RADIO_FIELDS


Ticks = Annotated[int, Field(gt=0, le=2**63 - 1)]
Power = Annotated[float, Field(ge=0, le=3.4028234663852886e38)]
Channel = Annotated[int, Field(ge=-(2**24), le=2**24)]
Identifier = Annotated[str, Field(pattern=r'^[A-Za-z_][A-Za-z0-9_]*$')]


class ResourceSelection(DataModel):
    resource_id: str = Field(min_length=1)
    channel_id: Channel
    center_frequency_hz: float = Field(gt=0)
    bandwidth_hz: float = Field(gt=0)
    sharing: Literal['exclusive', 'reuse']


class EndpointLimit(DataModel):
    resource_ids: list[str] = Field(min_length=1)
    max_power_w: float = Field(gt=0, le=3.4028234663852886e38)


class EmitterState(DataModel):
    active: list[bool] = Field(min_length=1)
    power_w: list[Power] = Field(min_length=1)
    channel_id: list[Channel] = Field(min_length=1)


class FixedEmitterConfig(EmitterState):
    pass


class PeriodicEmitterConfig(DataModel):
    on_ticks: Ticks
    off_ticks: Ticks
    phase_ticks: int = Field(ge=0)
    power_w: list[Power] = Field(min_length=1)
    channel_id: list[Channel] = Field(min_length=1)


class SweepEmitterConfig(DataModel):
    channel_ids: list[Channel] = Field(min_length=1)
    dwell_ticks: Ticks
    initial_index: int = Field(ge=0)
    active: list[bool] = Field(min_length=1)
    power_w: list[Power] = Field(min_length=1)


class MarkovEmitterConfig(DataModel):
    states: list[EmitterState] = Field(min_length=1)
    transition_matrix: list[list[float]]
    dwell_ticks: Ticks
    initial_state: int | None = Field(default=None, ge=0)
    initial_distribution: list[float] | None = None

    @model_validator(mode='after')
    def validate_process(self):
        n = len(self.states)
        if (self.initial_state is None) == (self.initial_distribution is None):
            raise ValueError('Declare exactly one initial_state or initial_distribution')
        if self.initial_state is not None and self.initial_state >= n:
            raise ValueError('Initial state is out of range')
        rows = self.transition_matrix
        if len(rows) != n:
            raise ValueError('Transition matrix must be square with one row per state')
        if self.initial_distribution is not None:
            rows = [*rows, self.initial_distribution]
        for row in rows:
            if len(row) != n or any(x < 0 for x in row) or not math.isclose(sum(row), 1., rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError('Probabilities must be nonnegative with row sum one')
        return self


class FixedEmitter(DataModel):
    emitter_id: str = Field(min_length=1)
    endpoints: list[str] = Field(min_length=1)
    behavior: Literal['fixed']
    config: FixedEmitterConfig


class PeriodicEmitter(DataModel):
    emitter_id: str = Field(min_length=1)
    endpoints: list[str] = Field(min_length=1)
    behavior: Literal['periodic']
    config: PeriodicEmitterConfig


class SweepEmitter(DataModel):
    emitter_id: str = Field(min_length=1)
    endpoints: list[str] = Field(min_length=1)
    behavior: Literal['sweep']
    config: SweepEmitterConfig


class MarkovEmitter(DataModel):
    emitter_id: str = Field(min_length=1)
    endpoints: list[str] = Field(min_length=1)
    behavior: Literal['markov']
    config: MarkovEmitterConfig


BackgroundEmitter = Annotated[FixedEmitter | PeriodicEmitter | SweepEmitter | MarkovEmitter, Field(discriminator='behavior')]


class CommunicationAllocation(DataModel):
    link_id: str
    resource_ids: list[str] = Field(min_length=1)


class ServiceSelection(CommunicationAllocation):
    service_id: Identifier
    initial_bits: float = Field(ge=0)


class NetworkSelection(DataModel):
    resources: list[ResourceSelection] = Field(min_length=1)
    epoch_max_ticks: Ticks
    endpoint_limits: dict[str, EndpointLimit]
    role_controls: dict[str, str] = Field(default_factory=dict)
    background_emitters: list[BackgroundEmitter] = Field(default_factory=list)
    passive_endpoints: list[str] = Field(default_factory=list)
    communication_allocations: list[CommunicationAllocation] = Field(default_factory=list)
    services: list[ServiceSelection] = Field(default_factory=list)


class EndpointBudget(DataModel):
    model_config = ConfigDict(frozen=True)
    resource_indices: tuple[int, ...]
    max_power_w: float


class ControlBinding(DataModel):
    model_config = ConfigDict(frozen=True)
    role_id: str
    endpoint_indices: tuple[int, ...]


class EmitterBinding(DataModel):
    model_config = ConfigDict(frozen=True)
    emitter_id: str
    endpoint_indices: tuple[int, ...]
    behavior: Literal['fixed', 'periodic', 'sweep', 'markov']
    config: FixedEmitterConfig | PeriodicEmitterConfig | SweepEmitterConfig | MarkovEmitterConfig


class CommunicationBinding(DataModel):
    model_config = ConfigDict(frozen=True)
    link_index: int
    tx: int
    rx: int
    path: int
    resource_indices: tuple[int, ...]


class ServiceBinding(DataModel):
    model_config = ConfigDict(frozen=True)
    service_id: str
    link_indices: tuple[int, ...]
    resource_indices: tuple[int, ...]
    initial_bits: float


class NetworkSpec(DataModel):
    model_config = ConfigDict(frozen=True)
    endpoints: tuple[GeometryPoint, ...]
    endpoint_sets: dict[str, tuple[int, ...]]
    endpoint_to_point: tuple[int, ...]
    resources: tuple[ResourceSelection, ...]
    endpoint_limits: tuple[EndpointBudget, ...]
    role_controls: tuple[ControlBinding, ...]
    background_emitters: tuple[EmitterBinding, ...]
    passive_endpoints: tuple[int, ...]
    communication_allocations: tuple[CommunicationBinding, ...]
    services: tuple[ServiceBinding, ...]
    epoch_offsets: tuple[int, ...]
    physics_dt: float
    link_count: int
    sensing_count: int
    point_count: int
    path_count: int


class BoundModel(DataModel):
    model_config = ConfigDict(frozen=True)
    reference: str
    config: dict
    descriptor: RadioDescriptor | SensingDescriptor
    input_fields: list[InputField]


class SensingFieldView(NamedTuple):
    data: Any
    valid: Any
    interval_start_tick: Any
    interval_end_tick: Any


class ServiceView(NamedTuple):
    remaining_bits: Any
    served_bits: Any
    valid: Any
    interval_start_tick: Any
    interval_end_tick: Any


class WirelessSnapshot(NamedTuple):
    radio: Any
    radio_sample_tick: Any
    radio_interval_s: Any
    services: dict[str, ServiceView]
    sensing: dict[str, SensingFieldView]


def network_model_input_fields(spec, *, geometry, batch_size, phase):
    """模型reset/evaluate共享的具体字段；静态资源只有F轴。"""
    b, e, f, l = batch_size, len(spec.endpoints), len(spec.resources), spec.link_count
    fields = [] if geometry is None else geometry_input_fields(geometry.layout, batch_size=b, phase=phase)
    declarations = [
        (['physics_outputs', 'sim_time'], [b], 'float64', 's', 'physics'),
        (['network', 'interval_s'], [b], 'float64', 's', 'physics'),
        (['network', 'endpoint_masks', 'present'], [b, e], 'bool', '1', 'radio_epoch'),
        (['network', 'resources', 'frequency_hz'], [f], 'float64', 'Hz', 'radio_epoch'),
        (['network', 'resources', 'bandwidth_hz'], [f], 'float64', 'Hz', 'radio_epoch'),
        (['network', 'simultaneous_transmissions', 'active'], [b, e, f], 'bool', '1', 'radio_epoch'),
        (['network', 'simultaneous_transmissions', 'power_w'], [b, e, f], 'float32', 'W', 'radio_epoch')]
    if l:
        declarations += [
            (['network', 'allocations', 'link_active'], [b, l, f], 'bool', '1', 'radio_epoch'),
            (['network', 'allocations', 'link_power_w'], [b, l, f], 'float32', 'W', 'radio_epoch')]
    fields.extend(InputField(path=path, phase=phase, tensor=TensorSpec(shape=shape, dtype=dtype,
        units=units, frame='none', rate=rate)) for path, shape, dtype, units, rate in declarations)
    return fields


def network_axes(spec, *, geometry, batch_size):
    """仅发布已存在的正轴；模型layout与CPU绑定共同消费。"""
    axes = {'B': batch_size, 'E': len(spec.endpoints), 'F': len(spec.resources),
            'L': spec.link_count, 'S': spec.sensing_count,
            'Q': spec.point_count, 'P': spec.path_count}
    if geometry is not None:
        axes.update(Q=len(geometry.layout.points), P=len(geometry.layout.paths))
    return {name: size for name, size in axes.items() if size > 0}


def network_public_fields(spec, model_bindings, *, batch_size):
    """Task/Term公开结果表，不向其重复发布原始感知结果。"""
    fields = network_model_input_fields(spec, geometry=None, batch_size=batch_size, phase='post_physics')
    b, l, f = batch_size, spec.link_count, len(spec.resources)

    def add(path, shape, dtype, units, rate='decision'):
        fields.append(InputField(path=path, phase='post_physics', tensor=TensorSpec(
            shape=shape, dtype=dtype, units=units, frame='none', rate=rate)))

    radio = model_bindings.get('radio')
    if radio is not None:
        for block in radio.descriptor.output_spec.output_blocks:
            for name, units in RADIO_FIELDS[block].items():
                add(['network', 'radio', block, name], [b, l, f], 'bool' if name == 'valid' else 'float32', units, 'radio_epoch')
        for name, dtype, units in (('sample_tick', 'int64', 'tick'), ('sample_time', 'float64', 's'), ('interval_s', 'float64', 's')):
            add(['network', 'radio', name], [b], dtype, units, 'radio_epoch')
    for service in spec.services:
        root = ['network', 'services', service.service_id]
        for name in ('remaining_bits', 'served_bits', 'valid'):
            add([*root, name], [b, len(service.link_indices)], 'bool' if name == 'valid' else 'float64', '1' if name == 'valid' else 'bit')
        for name in ('interval_start_tick', 'interval_end_tick'):
            add([*root, name], [b], 'int64', 'tick')
    sensing = model_bindings.get('sensing')
    if sensing is not None:
        axes = network_axes(spec, geometry=None, batch_size=b)
        for name in ('measurement', 'fim'):
            output = getattr(sensing.descriptor.output_spec, name)
            if output is None:
                continue
            shape = [axes[dim] if isinstance(dim, str) else dim for dim in output.shape]
            root = ['sensing_outputs', name]
            fields.append(InputField(path=[*root, 'data'], phase='post_physics', tensor=output.model_dump(exclude={'time_semantics'}) | {'shape': shape}))
            add([*root, 'valid'], shape[:-2] if name == 'fim' else shape, 'bool', '1', output.rate)
            for tick in ('interval_start_tick', 'interval_end_tick'):
                add([*root, tick], [b], 'int64', 'tick')
    return fields
