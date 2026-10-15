"""Radio/Sensing模型的轻量计算契约；tensor检查由Runtime拥有。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple


@dataclass(frozen=True)
class ExecutionContext:
    device: str


class ModelResetMask(NamedTuple):
    world_reset: Any


class ModelResult(NamedTuple):
    output: Any
    next_state: Any
    diagnostics: Mapping[str, Any]


class LinkMetrics(NamedTuple):
    path_gain_linear: Any
    received_power_w: Any
    valid: Any


class InterferenceMetrics(NamedTuple):
    interference_power_w: Any
    sinr_linear: Any
    valid: Any


class RateMetrics(NamedTuple):
    rate_bps: Any
    valid: Any


class RadioEvaluationResult(NamedTuple):
    link: LinkMetrics | None
    interference: InterferenceMetrics | None
    rate: RateMetrics | None


class SensingResult(NamedTuple):
    measurement: Any
    fim: Any
    field_valid: Mapping[str, Any]
    time_semantics: Mapping[str, str]


class RadioInputs(NamedTuple):
    geometry: Mapping[str, Any]
    endpoint_masks: Mapping[str, Any]
    resources: Mapping[str, Any]
    allocations: Mapping[str, Any]
    simultaneous_transmissions: Mapping[str, Any]
    sim_time: Any
    interval_s: Any


class SensingInputs(NamedTuple):
    geometry: Mapping[str, Any]
    endpoint_masks: Mapping[str, Any]
    resources: Mapping[str, Any]
    allocations: Mapping[str, Any]
    simultaneous_transmissions: Mapping[str, Any]
    radio_outputs: RadioEvaluationResult | None
    sim_time: Any
    interval_s: Any


# 固定SDK物理字段，不是开放registry；声明与实际返回校验共用单位定义。
RADIO_FIELDS = {
    'link': {'path_gain_linear': '1', 'received_power_w': 'W', 'valid': '1'},
    'interference': {'interference_power_w': 'W', 'sinr_linear': '1', 'valid': '1'},
    'rate': {'rate_bps': 'bit/s', 'valid': '1'},
}
RADIO_BLOCK_TYPES = {'link': LinkMetrics, 'interference': InterferenceMetrics, 'rate': RateMetrics}
MODEL_INPUT_VIEWS = ('geometry', 'endpoint_masks', 'resources', 'allocations', 'simultaneous_transmissions')


class RadioModel(ABC):
    def __init__(self, config, layout, context: ExecutionContext):
        self.config, self.layout, self.context = config, layout, context

    @classmethod
    @abstractmethod
    def describe(cls) -> dict:
        ...

    def initial_state(self, batch_shape):
        return ()

    @abstractmethod
    def reset_state(self, state, reset_mask: ModelResetMask, inputs: RadioInputs, rng) -> ModelResult:
        ...

    @abstractmethod
    def evaluate(self, inputs: RadioInputs, state, rng) -> ModelResult:
        ...


class SensingModel(ABC):
    def __init__(self, config, layout, context: ExecutionContext):
        self.config, self.layout, self.context = config, layout, context

    @classmethod
    @abstractmethod
    def describe(cls) -> dict:
        ...

    def initial_state(self, batch_shape):
        return ()

    @abstractmethod
    def reset_state(self, state, reset_mask: ModelResetMask, inputs: SensingInputs, rng) -> ModelResult:
        ...

    @abstractmethod
    def evaluate(self, inputs: SensingInputs, state, rng) -> ModelResult:
        ...
