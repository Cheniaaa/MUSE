"""Viser 的公开对象层；渲染、原生拾取及客户端临时手势共享同一帧锁。"""
from dataclasses import dataclass, field
import threading
from uuid import uuid4

import mujoco
import numpy as np
from mjviser.conversions import merge_geoms

from muse_sdk.experiment import Pose
from muse_sdk.view_protocol import EditGesture, SceneObject, ViewIntent
from .scene_data import apply_body_poses, geom_rotations


@dataclass
class _ClientState:
    client: object
    paired: bool
    claim: str | None = None
    storage_read: bool = False
    route: str | None = None
    interactive: bool = False
    roots: dict = field(default_factory=dict)
    visible: dict = field(default_factory=dict)
    selected: str | None = None
    highlight: object = None
    gizmo: object = None
    gesture_id: str | None = None
    delta: tuple | None = None


def _quaternion(matrix):
    value = np.empty(4)
    mujoco.mju_mat2Quat(value, np.asarray(matrix).reshape(9))
    return value


class SceneInteraction:
    """每个 ViserServer 生命周期仅一个实例；Run 不传 revision/editable。"""

    def __init__(self, server, model, objects: list[SceneObject], *, view_id: str | None = None, emit,
                 revision: int | None = None, editable: set[str] = frozenset()):
        self.server, self.model = server, model
        self.view_id, self.emit, self.revision = view_id, emit, revision
        self.editable = editable
        self._view_ids = set()
        self._lock = threading.RLock()
        self._closed = False
        self._frame = None
        self._data = mujoco.MjData(model)
        self._objects = {obj.object_id: obj for obj in objects}
        self._names = {obj.object_id: f'/muse/objects/{index}' for index, obj in enumerate(objects)}
        self._geom_objects = {geom: obj.object_id for obj in objects for geom in obj.geom_ids}
        self._clients = {}
        self._rotations = geom_rotations(model)
        self._meshes, self._rendered = {}, {}
        self._root = server.scene.add_frame('/muse', show_axes=False, visible=False)
        for obj in objects:
            server.scene.add_frame(self._names[obj.object_id], show_axes=False)
            for geom in obj.geom_ids:
                mesh = merge_geoms(model, [geom])
                self._meshes[geom] = mesh
                self._rendered[geom] = server.scene.add_mesh_trimesh(
                    self._names[obj.object_id] + f'/geoms/{geom}', mesh)
        server.on_client_connect(self.connect_client)
        server.on_client_disconnect(self.disconnect_client)

    def update_frame(self, xpos, xmat, frame: dict) -> None:
        """消费调用方的本帧 body world pose；绝不从初态 qpos 做 forward。"""
        with self._lock, self.server.atomic():
            if self._closed:
                return
            first = self._frame is None
            apply_body_poses(self.model, self._data, xpos, xmat, self._rotations)
            self._frame = dict(frame)
            for geom, handle in self._rendered.items():
                body = self.model.geom_bodyid[geom]
                handle.position = self._data.xpos[body]
                handle.wxyz = _quaternion(self._data.xmat[body])
            self._root.visible = True
            if first:
                self._focus(self.server.initial_camera, None, None)
            for state in self._clients.values():
                if first:
                    self._focus(state.client.camera, None, state if state.interactive else None)
                self._refresh_client(state)

    def _bounds(self, object_ids, state):
        points = []
        for object_id in object_ids:
            if state is not None and not state.visible[object_id]:
                continue
            for geom in self._objects[object_id].geom_ids:
                body = self.model.geom_bodyid[geom]
                vertices = self._meshes[geom].vertices @ self._data.xmat[body].reshape(3, 3).T
                vertices = vertices + self._data.xpos[body]
                if state is not None and object_id == state.selected and state.delta is not None:
                    rotation, position = state.delta
                    vertices = vertices @ rotation.T + position
                points.append(vertices)
        if not points:
            return None
        points = np.concatenate(points)
        return points.min(0), points.max(0)

    def _focus(self, camera, object_id, state):
        if self._frame is None:
            return
        ids = [object_id] if object_id is not None else [
            obj.object_id for obj in self._objects.values() if obj.kind == 'entity']
        if not ids:
            ids = list(self._objects)
        bounds = self._bounds(ids, state)
        if bounds is None:
            return
        low, high = bounds
        center = (low + high) / 2
        radius = max(float(np.linalg.norm(high - low)) / 2, .2)
        direction = np.array([1., -1., .5])
        direction /= np.linalg.norm(direction)
        camera.position = center + direction * radius / np.sin(np.deg2rad(25.))
        camera.look_at = center
        camera.up = (0., 0., 1.)
        camera.fov = float(np.deg2rad(50.))

    def connect_client(self, client) -> None:
        with self._lock:
            if self._closed:
                return
            state = _ClientState(client, False)
            self._clients[client.client_id] = state
        message = None
        try:
            claim = client.local_storage.get_item('muse_view_id', timeout=5.0)
        except (TimeoutError, RuntimeError) as exc:
            message = str(exc)
        with self._lock:
            # get_item 会阻塞等待浏览器；期间的断开/close 必须先于配对结果生效。
            if self._closed or self._clients.get(client.client_id) is not state:
                return
            if message is None:
                state.claim, state.storage_read = claim, True
            self._bind_client(state, initial=True, message=message)

    def _emit_view(self, state, event):
        message = {'kind': 'view', 'event': event}
        if self.view_id is None:
            message['view_id'] = state.route
        self.emit(message)

    def _bind_client(self, state, *, initial=False, message=None):
        route = None
        if state.storage_read:
            if self.view_id is not None and state.claim == self.view_id:
                route = self.view_id
            elif self.view_id is None and state.claim in self._view_ids:
                route = state.claim
        interactive = route is not None or (
            self.view_id is None and state.storage_read and state.claim is None)
        if not initial and route == state.route and interactive == state.interactive:
            return
        if state.interactive:
            self._remove_client(state)
        state.route, state.paired, state.interactive = route, route is not None, interactive
        if interactive:
            client = state.client
            state.visible = {object_id: True for object_id in self._objects}
            state.roots = {object_id: client.scene.add_frame(name, show_axes=False)
                           for object_id, name in self._names.items()}
            state.highlight = client.scene.add_box('/muse/highlight', wireframe=True,
                color=(255, 210, 0), visible=False)
            client.scene.on_click()(self.handle_click)
        if initial:
            self._focus(state.client.camera, None, state if interactive else None)
        self._emit_view(state, {'kind': 'pairing', 'client_id': state.client.client_id,
            'connected': True, 'paired': state.paired, 'message': message})

    def set_view_ids(self, ids: list[str]) -> None:
        """Run 注册集合；只重配已读声明，不进行浏览器网络查询。"""
        with self._lock:
            if self._closed:
                return
            if self.view_id is not None:
                raise ValueError('Draft view ownership is fixed')
            updated = set(ids)
            if updated == self._view_ids:
                return
            self._view_ids = updated
            for state in self._clients.values():
                if state.storage_read:
                    self._bind_client(state)

    def _remove_client(self, state):
        if state.interactive:
            state.client.scene.remove_click_callback(self.handle_click)
            if state.gizmo is not None:
                state.gizmo.remove_update_callback()
                state.gizmo.remove()
            state.highlight.remove()
            for handle in state.roots.values():
                handle.remove()
            state.interactive = False
            state.roots, state.visible = {}, {}
            state.selected, state.highlight, state.gizmo = None, None, None
            state.gesture_id, state.delta = None, None

    def disconnect_client(self, client) -> None:
        with self._lock:
            state = self._clients.pop(client.client_id, None)
            if state is not None:
                self._remove_client(state)
                if not self._closed:
                    self._emit_view(state, {'kind': 'pairing', 'client_id': client.client_id,
                        'connected': False, 'paired': state.paired})

    def _object_pose(self, object_id):
        obj = self._objects[object_id]
        if obj.kind == 'environment':
            geom, = obj.geom_ids
            return self._data.geom_xpos[geom], self._data.geom_xmat[geom].reshape(3, 3)
        return self._data.xpos[obj.body_id], self._data.xmat[obj.body_id].reshape(3, 3)

    def _refresh_client(self, state):
        if not state.interactive or state.selected is None:
            return
        visible = state.visible[state.selected]
        bounds = self._bounds([state.selected], state)
        if bounds is not None:
            low, high = bounds
            state.highlight.position = (low + high) / 2
            state.highlight.dimensions = tuple(high - low)
        state.highlight.visible = visible and bounds is not None
        if state.gizmo is not None:
            state.gizmo.visible = visible
            if state.delta is None:
                position, rotation = self._object_pose(state.selected)
                state.gizmo.position = position
                state.gizmo.wxyz = _quaternion(rotation)

    def _select(self, state, object_id):
        if state.selected is not None:
            state.roots[state.selected].position = (0., 0., 0.)
            state.roots[state.selected].wxyz = (1., 0., 0., 0.)
        if state.gizmo is not None:
            state.gizmo.remove_update_callback()
            state.gizmo.remove()
            state.gizmo = None
        state.selected, state.delta, state.gesture_id = object_id, None, None
        state.highlight.visible = False
        if object_id is not None:
            if self.view_id is not None and state.paired and self.revision is not None and object_id in self.editable:
                position, rotation = self._object_pose(object_id)
                state.gizmo = state.client.scene.add_transform_controls('/muse/gizmo',
                    position=position, wxyz=_quaternion(rotation), scale=.4)
                state.gizmo.on_update(self.handle_transform)
            self._refresh_client(state)
        self._emit_view(state, {'kind': 'selection', 'client_id': state.client.client_id,
            'object_id': object_id, 'frame': self._frame})

    def apply_intent(self, intent: ViewIntent, *, client_id: int | None = None,
                     view_id: str | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            if view_id is not None and (self.view_id is not None or view_id not in self._view_ids):
                raise ValueError(f'Run view_id is not registered: {view_id}')
            if self.view_id is None and view_id is None and client_id is None:
                raise ValueError('Run intents require a registered view_id or a client_id')
            if intent.object_id is not None and intent.object_id not in self._objects:
                raise ValueError(f'Unknown scene object: {intent.object_id}')
            if intent.kind == 'visibility' and (intent.object_id is None or intent.visible is None):
                raise ValueError('Visibility requires object_id and visible')
            for identity, state in self._clients.items():
                if (not state.interactive or (client_id is not None and identity != client_id)
                        or (view_id is not None and state.route != view_id)):
                    continue
                if intent.kind == 'select':
                    self._select(state, intent.object_id)
                elif intent.kind == 'visibility':
                    state.visible[intent.object_id] = intent.visible
                    state.roots[intent.object_id].visible = intent.visible
                    self._refresh_client(state)
                else:
                    target = None if intent.kind == 'reset_camera' else intent.object_id or state.selected
                    self._focus(state.client.camera, target, state)

    def pick(self, ray_origin, ray_direction, *, client_id: int | None = None) -> dict:
        with self._lock:
            if self._closed or self._frame is None:
                return {'object_id': None, 'distance': -1., 'frame': None}
            state = self._clients[client_id] if client_id is not None else None
            groups = self.model.geom_group.copy()
            delta_ids = []
            if state is not None and state.delta is not None:
                delta_ids = self._objects[state.selected].geom_ids
            positions = self._data.geom_xpos[delta_ids].copy()
            rotations = self._data.geom_xmat[delta_ids].copy()
            try:
                # mj_ray 只有 group 过滤；本模型属于视口，锁内临时分组，finally 原样恢复。
                self.model.geom_group[:] = 0
                if state is not None:
                    for object_id, visible in state.visible.items():
                        if not visible:
                            self.model.geom_group[self._objects[object_id].geom_ids] = 1
                    if delta_ids:
                        rotation, position = state.delta
                        self._data.geom_xpos[delta_ids] = positions @ rotation.T + position
                        self._data.geom_xmat[delta_ids] = (
                            rotation @ rotations.reshape(-1, 3, 3)).reshape(-1, 9)
                hit = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(self.model, self._data, np.asarray(ray_origin, dtype=float),
                    np.asarray(ray_direction, dtype=float), np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
                    True, -1, hit)
                return {'object_id': None if hit[0] < 0 else self._geom_objects[int(hit[0])],
                        'distance': distance, 'frame': dict(self._frame)}
            finally:
                self.model.geom_group[:] = groups
                self._data.geom_xpos[delta_ids] = positions
                self._data.geom_xmat[delta_ids] = rotations

    async def handle_click(self, event) -> None:
        with self._lock:
            state = self._clients.get(event.client_id)
            if self._closed or state is None or not state.interactive:
                return
            hit = self.pick(event.ray_origin, event.ray_direction, client_id=event.client_id)
            self._select(state, hit['object_id'])

    async def handle_transform(self, event) -> None:
        # 不 await：固定版 Viser 在 event loop 上顺序派发 start/update/end。
        with self._lock:
            state = self._clients.get(event.client_id)
            if (self._closed or self.view_id is None or state is None or not state.paired
                    or event.target is not state.gizmo):
                return
            if event.phase == 'start':
                state.gesture_id = uuid4().hex
            pose = Pose(position=list(event.target.position), orientation_wxyz=list(event.target.wxyz))
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, np.asarray(pose.orientation_wxyz))
            base_position, base_rotation = self._object_pose(state.selected)
            rotation = rotation.reshape(3, 3) @ base_rotation.T
            position = np.asarray(pose.position) - rotation @ base_position
            state.delta = rotation, position
            state.roots[state.selected].position = position
            state.roots[state.selected].wxyz = _quaternion(rotation)
            self._refresh_client(state)
            gesture = EditGesture(base_revision=self.revision, object_id=state.selected,
                gesture_id=state.gesture_id, phase=event.phase, pose=pose)
            self.emit({'kind': 'gesture', 'gesture': gesture.model_dump(mode='json')})
            if event.phase == 'end':
                state.gesture_id = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for state in self._clients.values():
                self._remove_client(state)
            self._clients.clear()
            self._root.remove()
