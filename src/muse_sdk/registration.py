"""包注册只绑定实现和metadata，不创建执行实例。"""
from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel


@dataclass(frozen=True)
class TermRegistration:
    implementation: Callable
    term_kind: Literal['observation', 'reward', 'cost', 'termination']
    scope: Literal['role', 'world']
    config_model: type[BaseModel]
    input_fields: tuple[dict, ...]
    output_spec: dict
    cost_id: str | None = None
    time_out: bool = False


@dataclass(frozen=True)
class TrainingPackageDefinition:
    terms: dict[str, TermRegistration]
    tasks: dict[str, type]
    algorithms: dict[str, type]
    radio_models: dict[str, type]
    sensing_models: dict[str, type]
