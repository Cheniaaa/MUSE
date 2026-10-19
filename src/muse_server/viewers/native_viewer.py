"""官方Native显示的独立CPU状态；不持有Runtime或GPU存储。"""
import mujoco
import numpy as np

from muse_sdk.run_protocol import native_state_layout


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
