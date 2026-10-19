"""官方Native显示的独立CPU状态；不持有Runtime或GPU存储。"""
import copy
import threading

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from muse_sdk.run_protocol import PreviewFrame, PreviewSelection, native_state_layout


class NativeSnapshot:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self.layout = native_state_layout(nq=model.nq, nv=model.nv, na=model.na, nu=model.nu,
            nbody=model.nbody, nmocap=model.nmocap, neq=model.neq, nsensordata=model.nsensordata)
        self.payload_floats = sum(size for _, size in self.layout)
        mujoco.mj_forward(model, self.data)

    def update(self, arrays: bytes, *, sim_time: float) -> None:
        if len(arrays) != self.payload_floats * 4:
            raise ValueError('Native snapshot payload length mismatch')
        values = np.frombuffer(arrays, dtype='<f4')
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite Native snapshot state')
        fields, offset = [], 0
        for name, size in self.layout:
            value = values[offset:offset + size].reshape(getattr(self.data, name).shape)
            if name == 'eq_active' and not ((value == 0) | (value == 1)).all():
                raise ValueError('Native eq_active must contain only 0 or 1')
            fields.append((name, value))
            offset += size
        # 完整反序列化成功后才替换canonical；每个新样本清除未采样缓存。
        mujoco.mj_resetData(self.model, self.data)
        self.data.time = sim_time
        for name, value in fields:
            getattr(self.data, name)[:] = value
        mujoco.mj_forward(self.model, self.data)
        self.data.sensordata[:] = fields[-1][1]


class OfficialNativeViewer:
    # 固定MuJoCo3.11公开Physics参数；可视偏好model.vis/stat及Handle.opt/cam不在恢复范围。
    _PHYSICS_FIELDS = (
        'ccd_iterations', 'ccd_tolerance', 'cone', 'density', 'disableactuator', 'disableflags',
        'enableflags', 'gravity', 'impratio', 'integrator', 'iterations', 'jacobian',
        'ls_iterations', 'ls_tolerance', 'magnetic', 'noslip_iterations', 'noslip_tolerance',
        'o_friction', 'o_margin', 'o_solimp', 'o_solref', 'sdf_initpoints', 'sdf_iterations',
        'sleep_tolerance', 'solver', 'timestep', 'tolerance', 'viscosity', 'wind')

    def __init__(self, snapshot: NativeSnapshot, *, run_id: str):
        self.snapshot, self.run_id = snapshot, run_id
        self.model = copy.copy(snapshot.model)
        self.data = mujoco.MjData(self.model)
        self.handle = None
        self._threads = set()
        self._selection: PreviewSelection | None = None
        self.frame: PreviewFrame | None = None
        self._pending: PreviewFrame | None = None
        self._text = None
        if not glfw.init():
            raise RuntimeError(f'GLFW initialization failed: {glfw.get_error()}')
        before = set(threading.enumerate())
        try:
            try:
                self.handle = mujoco.viewer.launch_passive(self.model, self.data,
                    show_left_ui=True, show_right_ui=True)
            finally:
                self._threads = set(threading.enumerate()) - before
            self._update_text()
        except Exception:
            self.close()
            raise

    def set_selection(self, selection: PreviewSelection) -> None:
        self._selection = selection
        self._pending = None
        self._update_text()

    def update_frame(self, header: PreviewFrame, arrays: bytes) -> None:
        self.snapshot.update(arrays, sim_time=header.sim_time)
        if self.frame is None:
            moving = (self.model.body_dofnum > 0) | (self.model.body_mocapid >= 0)
            centers = self.snapshot.data.xpos[moving]
            if len(centers):
                with self.handle.lock():
                    self.handle.cam.lookat[:] = centers.mean(axis=0)
        self._pending = header

    def sync(self) -> PreviewFrame | None:
        with self.handle.lock():
            self.handle.sync(state_only=False)
            for name in self._PHYSICS_FIELDS:
                value = getattr(self.snapshot.model.opt, name)
                if isinstance(value, np.ndarray):
                    getattr(self.model.opt, name)[:] = value
                else:
                    setattr(self.model.opt, name, value)
            self.handle.perturb.active = self.handle.perturb.active2 = 0
            mujoco.mj_copyData(self.data, self.model, self.snapshot.data)
            self.handle.sync(state_only=False)
            # sync尾部会清空/施加扰动力，公开display副本也恢复到canonical。
            mujoco.mj_copyData(self.data, self.model, self.snapshot.data)
        ack, self._pending = self._pending, None
        if ack is not None:
            self.frame = ack
        self._update_text()
        return ack

    def _update_text(self) -> None:
        if not self.is_running():
            return
        frame, selection = self.frame, self._selection
        identity = 'Frozen model initial state; no Run sample synchronized'
        if frame is not None:
            identity = (f'Last synchronized: world {frame.world_id} / episode {frame.episode} / '
                        f'decision {frame.decision} / tick {frame.tick} / time {frame.sim_time:.3f}')
        waiting = ''
        if selection is not None and (frame is None or frame.selection_seq != selection.selection_seq):
            waiting = f'\nWaiting: world {selection.display.world_id} / selection {selection.selection_seq}'
        text = (f'MUSE Run {self.run_id}\n{identity}{waiting}\n'
                'Training isolated; native writes restored from snapshot\n'
                'Worker: qpos/qvel/act/ctrl/forces/mocap/eq and native Sensor\n'
                'Contact/dynamics: CPU reconstruction; Profiler: display only\n'
                'Unsampled userdata/history/custom state unavailable; sync is not presentation')
        if text != self._text:
            # set_texts等待render回合，必须在viewer锁外调用。
            self.handle.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ''))
            self._text = text

    def is_running(self) -> bool:
        return self.handle is not None and self.handle.is_running()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
        for thread in self._threads:
            thread.join()
        self._threads.clear()
        self.handle = None
        glfw.terminate()
