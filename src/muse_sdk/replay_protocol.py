"""历史回放的HTTP/WS边界；不依赖文件读取或图形模块。"""
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .config import DataModel


class ReplayRequest(DataModel):
    run_id: str
    trajectory_artifact_id: str


class ReplayCommand(DataModel):
    command_seq: int = Field(gt=0)
    kind: Literal['play', 'pause', 'seek', 'step', 'speed', 'world', 'episode']
    world_id: int | None = Field(default=None, ge=0)
    sample_index: int | None = Field(default=None, ge=0)
    episode: int | None = Field(default=None, ge=0)
    delta: Literal[-1, 1] | None = None
    speed: float | None = Field(default=None, gt=0)

    @field_validator('delta', mode='before')
    @classmethod
    def strict_delta(cls, value):
        # Literal比较允许True==1；WS边界仍须保持严格整数语义。
        if value is not None and type(value) is not int:
            raise ValueError('delta must be an integer -1 or +1')
        return value

    @model_validator(mode='after')
    def command_target(self):
        required = {'seek': 'sample_index', 'step': 'delta', 'speed': 'speed',
                    'world': 'world_id', 'episode': 'episode'}.get(self.kind)
        supplied = {name for name in ('world_id', 'sample_index', 'episode', 'delta', 'speed')
                    if getattr(self, name) is not None}
        if supplied != ({required} if required else set()):
            raise ValueError(f'{self.kind} requires only {required or "no target"}')
        return self


class ReplayState(DataModel):
    session_id: str
    status: Literal['preparing', 'ready', 'error', 'closing']
    source: ReplayRequest
    viewer_url: str | None = None
    revision: int = 0
    last_command_seq: int = 0
    playing: bool = False
    speed: float = 1.
    cursor: dict | None = None
    worlds: dict[str, dict] = Field(default_factory=dict)
    episodes: dict[str, list[dict]] = Field(default_factory=dict)
    objects: list[dict] = Field(default_factory=list)
    diagnostics: dict | None = None
    camera_ready: bool = False
    export: dict | None = None
    error: str | None = None
