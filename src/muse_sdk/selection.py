"""包级selection结果，尚不包含Scenario/Experiment布局。"""
from typing import Literal

from pydantic import Field

from .config import DataModel


class ResolvedSelection(DataModel):
    owner_path: tuple[str, ...]
    term_kind: Literal['observation', 'reward', 'cost', 'termination']
    selection_id: str
    term: str
    enabled: bool
    config: dict
    weight: float | None
    binding_role_id: str | None


class ResolvedPackageSelection(DataModel):
    package_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    selections: tuple[ResolvedSelection, ...]
    termination_reasons: dict[int, tuple[str, str, str]]
    required_inputs: tuple[dict, ...]
