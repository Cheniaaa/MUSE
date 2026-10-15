"""资源作者的RF语义、实验关系及平台派生布局；不加载数值后端。"""
import math
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator

from .config import DataModel
from .descriptor import InputField, TensorSpec


class RadioDeviceDefinition(DataModel):
    can_transmit: bool
    can_receive: bool
    body: str = Field(min_length=1)
    position_m: list[float] = Field(min_length=3, max_length=3)
    orientation_wxyz: list[float] = Field(min_length=4, max_length=4)
    assumption: str = Field(min_length=1)

    @field_validator('orientation_wxyz')
    @classmethod
    def unit_quaternion(cls, value):
        if not math.isclose(sum(x*x for x in value), 1., rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError('Expected a unit mount quaternion')
        return value


class EnvironmentRF(DataModel):
    rf_blocking: list[str]


class GeometryAssetFact(DataModel):
    geom_type: str
    body: str
    is_static: bool
    mesh: str | None


class GeometryAssetSummary(DataModel):
    geoms: dict[str, GeometryAssetFact]
    flex_names: list[str]


class TargetSelection(DataModel):
    group_id: str
    slots: list[int] = Field(min_length=1)


class EndpointSelection(TargetSelection):
    device_id: str


class LinkSelection(DataModel):
    link_id: str
    tx: str
    rx: str


class SensingSelection(DataModel):
    sensing_id: str
    tx: str
    target: TargetSelection
    rx: str


class GeometrySelection(DataModel):
    endpoints: dict[str, EndpointSelection]
    links: list[LinkSelection] = Field(default_factory=list)
    sensing: list[SensingSelection] = Field(default_factory=list)
    endpoint_epsilon_m: float = Field(default=1e-4, gt=0, le=3.4028234663852886e38)

    @field_validator('endpoint_epsilon_m')
    @classmethod
    def representable_epsilon(cls, value):
        if struct.unpack('f', struct.pack('f', value))[0] == 0:
            raise ValueError('Endpoint epsilon underflows float32')
        return value


class GeometryPoint(DataModel):
    model_config = ConfigDict(frozen=True)
    group_id: str
    slot: int
    device_id: str | None
    body: str
    position_m: tuple[float, float, float]
    assumption: str


class PropagationPath(DataModel):
    model_config = ConfigDict(frozen=True)
    path_id: str
    tx: int
    rx: int


class DesiredGeometry(DataModel):
    model_config = ConfigDict(frozen=True)
    link_id: str
    path: int


class SensingGeometry(DataModel):
    model_config = ConfigDict(frozen=True)
    sensing_id: str
    outbound: int
    inbound: int
    direct: int | None


class InterfererGeometry(DataModel):
    model_config = ConfigDict(frozen=True)
    tx: int
    path: int


class ConsumerInterference(DataModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal['desired', 'sensing']
    consumer: int
    sources: tuple[InterfererGeometry, ...]


class PropagationPathLayout(DataModel):
    model_config = ConfigDict(frozen=True)
    points: tuple[GeometryPoint, ...]
    paths: tuple[PropagationPath, ...]
    desired: tuple[DesiredGeometry, ...]
    sensing: tuple[SensingGeometry, ...]
    interference: tuple[ConsumerInterference, ...]


class ResolvedGeometry(DataModel):
    model_config = ConfigDict(frozen=True)
    layout: PropagationPathLayout
    endpoint_epsilon_m: float
    blockers: tuple[str, ...]


def geometry_input_fields(layout: PropagationPathLayout, *, batch_size: int, phase: str):
    """同一字段表供宿主和模型声明使用；不提供radio epoch调度。"""
    q, p = len(layout.points), len(layout.paths)
    return [InputField(path=['network', 'geometry', name], phase=phase,
        tensor=TensorSpec(shape=shape, dtype=dtype, units=units, frame=frame, rate='radio_epoch'))
        for name, shape, dtype, units, frame in (
            ('positions_world', [batch_size, q, 3], 'float32', 'm', 'world_z_up'),
            ('distance_m', [batch_size, p], 'float32', 'm', 'none'),
            ('los', [batch_size, p], 'bool', '1', 'none'),
            ('valid', [batch_size, p], 'bool', '1', 'none'),
            ('sample_tick', [batch_size], 'int64', 'tick', 'none'),
            ('sample_time', [batch_size], 'float64', 's', 'none'))]


@dataclass(frozen=True)
class GeometryView:
    """借用只读快照；跨refresh持有数据的消费者自行clone。"""
    positions_world: Any
    distance_m: Any
    los: Any
    valid: Any
    sample_tick: Any
    sample_time: Any
    layout: PropagationPathLayout

    @property
    def mapping(self):
        return MappingProxyType({name: getattr(self, name) for name in (
            'positions_world', 'distance_m', 'los', 'valid', 'sample_tick', 'sample_time')})
