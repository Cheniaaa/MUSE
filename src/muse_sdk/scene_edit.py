"""有限环境编辑和冷路径预览的 JSON 边界；不包含运行时 overlay。"""
import math
from typing import Annotated

from pydantic import Field, field_validator, model_validator

from .config import DataModel
from .experiment import EntityModelRef, Pose, ResourceRef


class BoxEdit(DataModel):
    label: str
    pose: Pose
    size_m: list[Annotated[float, Field(gt=0)]] = Field(min_length=3, max_length=3)
    collision: bool
    rf_blocking: bool

    @field_validator('pose')
    @classmethod
    def unit_quaternion(cls, value):
        # Pose 复用现有字段；编辑边界与 resolver 的单位四元数语义一致。
        if not math.isclose(sum(x * x for x in value.orientation_wxyz), 1., abs_tol=1e-6):
            raise ValueError('orientation_wxyz must be a unit quaternion')
        return value


class EnvironmentEdit(DataModel):
    base: ResourceRef
    upserts: dict[str, BoxEdit]
    removed: list[str]

    @model_validator(mode='after')
    def edit_targets(self):
        if self.upserts.keys() & set(self.removed):
            raise ValueError('An object cannot be both updated and removed')
        if len(self.removed) != len(set(self.removed)):
            raise ValueError('Duplicate removed object')
        if any(not name for name in (*self.upserts, *self.removed)):
            raise ValueError('Object IDs must not be empty')
        return self


class EnvironmentObject(DataModel):
    object_id: str
    label: str
    editable: bool
    reason: str | None
    box: BoxEdit | None


class EnvironmentManifest(DataModel):
    base: ResourceRef
    objects: list[EnvironmentObject]


class PreviewEntityRequest(DataModel):
    object_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    slot: int = Field(ge=0)
    model_id: str
    model_revision: int = Field(gt=0)
    pose: Pose
    sensor_profile_id: str | None = None
    sensor_config: dict = Field(default_factory=dict)

    @field_validator('pose')
    @classmethod
    def unit_quaternion(cls, value):
        return BoxEdit.unit_quaternion(value)


class ScenePreviewRequest(DataModel):
    draft_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    environment: ResourceRef
    environment_edit: EnvironmentEdit | None = None
    entities: list[PreviewEntityRequest]

    @model_validator(mode='after')
    def distinct_entities(self):
        if len({entity.object_id for entity in self.entities}) != len(self.entities):
            raise ValueError('Duplicate preview object ID')
        if len({(entity.group_id, entity.slot) for entity in self.entities}) != len(self.entities):
            raise ValueError('Duplicate preview group/slot')
        return self


class PreviewEntity(DataModel):
    object_id: str
    group_id: str
    slot: int
    model: EntityModelRef
    pose: Pose
    sensor_outputs: list[dict]
