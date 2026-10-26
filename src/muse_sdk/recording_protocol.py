"""独立诊断轨迹的CPU配置与文件声明。"""
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .config import DataModel


class RecordingConfig(DataModel):
    world_ids: list[Annotated[int, Field(ge=0)]] = Field(min_length=1)
    every_n_decisions: int = Field(gt=0, description='episode决策步采样间隔；结束/reset边界总是保存')
    max_data_bytes: int = Field(gt=0, description='所有选中world的未压缩动态数组字节预算，不含模型和metadata')

    @model_validator(mode='after')
    def unique_worlds(self):
        if len(set(self.world_ids)) != len(self.world_ids):
            raise ValueError('Recording world_ids must be unique')
        return self


class RecordingState(DataModel):
    status: Literal['disabled', 'capturing', 'publishing', 'published', 'failed']
    config: RecordingConfig | None
    row_bytes: int | None = None
    rows_per_world: int | None = None
    static_bytes: int | None = None
    summary: dict | None = None
    artifact_id: str | None = None
    error: str | None = None


class TrajectoryArtifact(DataModel):
    id: str
    run: str
    kind: Literal['trajectory'] = 'trajectory'
    path: str
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    summary: dict


class TrajectoryField(DataModel):
    name: Literal['xpos', 'xmat', 'velocity_world', 'angular_velocity_body', 'present', 'alive',
        'controllable', 'crashed', 'action_motion', 'action_power_w', 'action_channel', 'reward',
        'world_reward', 'role_cost', 'world_cost', 'world_terminated', 'world_truncated',
        'world_end_reason', 'learning_done']
    role_id: str | None
    cost_id: str | None
    group: Literal['float', 'integer', 'boolean']
    dtype: Literal['<f4', '<i4', '<i8', '|b1']
    shape: list[int]
    offset: int


class TrajectoryManifest(DataModel):
    format_revision: Literal[1] = 1
    run_id: str
    experiment_id: str
    recording: RecordingConfig
    physics_dt: float
    decimation: int
    nbody: int = Field(gt=0)
    model_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    fields: list[TrajectoryField]
    summary: dict
    episodes: dict[str, list[dict]]
