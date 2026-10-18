"""草稿和 Run 视口共用的对象/意图；原生编号仅限内部编译快照。"""
from typing import Literal

from .config import DataModel
from .experiment import Pose
from .run_protocol import SceneObject


class ViewIntent(DataModel):
    kind: Literal['select', 'focus', 'reset_camera', 'visibility']
    object_id: str | None = None
    visible: bool | None = None


class EditGesture(DataModel):
    base_revision: int
    object_id: str
    gesture_id: str
    phase: Literal['start', 'update', 'end']
    pose: Pose
