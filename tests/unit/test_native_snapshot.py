"""Native数值边界：输入状态重建、原生sensor保真与坏包原子拒绝。"""
import mujoco
import numpy as np
import pytest

from muse_sdk import run_protocol as wire
from tests.preview_helpers import native_snapshot_case


def test_native_layout_uses_fixed_field_order():
    layout = wire.native_state_layout(nq=21, nv=18, na=0, nu=12, nbody=4,
                                     nmocap=0, neq=0, nsensordata=30)
    assert layout == (('qpos', 21), ('qvel', 18), ('act', 0), ('ctrl', 12),
        ('qfrc_applied', 18), ('xfrc_applied', 24), ('mocap_pos', 0),
        ('mocap_quat', 0), ('eq_active', 0), ('sensordata', 30))
    assert sum(size for _, size in layout) * 4 == 492


def test_native_snapshot_rebuilds_site_and_preserves_sensor():
    from muse_server.viewers.native_viewer import NativeSnapshot

    model, source, raw = native_snapshot_case()
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, source, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    snapshot = NativeSnapshot(model)
    snapshot.update(raw, sim_time=7.5)
    assert snapshot.payload_floats == 32
    assert snapshot.data.time == 7.5
    np.testing.assert_allclose(snapshot.data.site_xpos, [[.75, .3, .6]], atol=1e-6)
    np.testing.assert_allclose(snapshot.data.geom_xpos, source.geom_xpos, atol=1e-6)
    np.testing.assert_allclose(snapshot.data.site_xpos, source.site_xpos, atol=1e-6)
    for name in ('qpos', 'qvel', 'act', 'ctrl', 'qfrc_applied', 'xfrc_applied',
                 'mocap_pos', 'mocap_quat', 'eq_active'):
        np.testing.assert_allclose(getattr(snapshot.data, name), getattr(source, name), atol=1e-7)
    np.testing.assert_array_equal(snapshot.data.sensordata, [123.])
    after = np.empty_like(state)
    mujoco.mj_getState(model, source, after, mujoco.mjtState.mjSTATE_INTEGRATION)
    np.testing.assert_array_equal(after, state)
    np.testing.assert_array_equal(model.opt.gravity, [0, 0, -9.81])


def test_native_snapshot_resets_unsampled_cache():
    from muse_server.viewers.native_viewer import NativeSnapshot

    model, _, raw = native_snapshot_case()
    snapshot = NativeSnapshot(model)
    snapshot.update(raw, sim_time=7.5)
    acceleration = snapshot.data.qacc.copy()
    snapshot.data.userdata[:] = 91
    snapshot.data.qacc_warmstart[:] = 1e6
    snapshot.data.site_xpos[:] = 999
    snapshot.update(raw, sim_time=8.)
    assert snapshot.data.time == 8.
    np.testing.assert_array_equal(snapshot.data.userdata, [0.])
    np.testing.assert_allclose(snapshot.data.qacc, acceleration)
    np.testing.assert_allclose(snapshot.data.site_xpos, [[.75, .3, .6]], atol=1e-6)


@pytest.mark.parametrize('invalid', ['short', 'long', 'nan', 'inf', 'equality'])
def test_native_snapshot_rejects_invalid_payload(invalid):
    from muse_server.viewers.native_viewer import NativeSnapshot

    model, _, raw = native_snapshot_case()
    snapshot = NativeSnapshot(model)
    snapshot.update(raw, sim_time=7.5)
    bad = np.frombuffer(raw, dtype='<f4').copy()
    if invalid == 'short':
        packet = raw[:-1]
    elif invalid == 'long':
        packet = raw + bytes(4)
    else:
        bad[30 if invalid == 'equality' else 0] = {
            'nan': np.nan, 'inf': np.inf, 'equality': .5}[invalid]
        packet = bad.tobytes()
    with pytest.raises(ValueError):
        snapshot.update(packet, sim_time=9.)
    assert snapshot.data.time == 7.5
    np.testing.assert_allclose(snapshot.data.qpos, [.45])
    np.testing.assert_array_equal(snapshot.data.eq_active, [1])
    np.testing.assert_array_equal(snapshot.data.sensordata, [123.])
