"""真实官方UI：状态恢复、同步语义与render线程退出，均在有界子进程中执行。"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(os.environ.get('MUSE_TEST_NATIVE') != '1' or not os.environ.get('DISPLAY'),
                               reason='Explicit official Native X11 window validation')


def run_case(tmp_path, name):
    result = subprocess.run([sys.executable, '-m', 'tests.integration.test_official_native_viewer', name, str(tmp_path)],
                            capture_output=True, text=True, timeout=45)
    (tmp_path / 'child.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'glibc' not in result.stderr and 'double free' not in result.stderr
    return json.loads((tmp_path / 'result.json').read_text())


def test_official_viewer_restores_native_edits_without_new_sample(tmp_path):
    result = run_case(tmp_path, 'restore')
    assert result['source_unchanged'] and result['restored']


def test_official_viewer_waiting_and_sync_are_not_presentation(tmp_path):
    result = run_case(tmp_path, 'waiting')
    assert result['acks'] == [1, 2] and result['old_world'] == 0
    assert result['minimized_ack'] == 6


def test_official_viewer_mesh_and_passive_controls(tmp_path):
    result = run_case(tmp_path, 'mesh')
    assert result['mesh_count'] > 0 and result['passive_time'] == 7.5


@pytest.mark.parametrize('failure', ['none', 'launch', 'sync'])
def test_official_viewer_closes_owned_threads_on_error(tmp_path, failure):
    result = run_case(tmp_path, failure)
    assert result['threads_remaining'] == 0
    assert result['error'] == (None if failure == 'none' else 'injected failure')


def child_case(name, directory):
    import threading
    from unittest.mock import patch
    import mujoco
    import mujoco.viewer
    import numpy as np
    from muse_sdk.run_protocol import DisplaySettings, PreviewFrame, PreviewSelection
    from muse_server.viewers.native_viewer import NativeSnapshot, OfficialNativeViewer
    from tests.preview_helpers import native_snapshot_case
    from tests.native_ui_helpers import NativeUI, until

    root = Path(__file__).resolve().parents[2]
    os.chdir(directory)
    model, source, raw = native_snapshot_case()
    if name == 'mesh':
        model = mujoco.MjModel.from_xml_path(str(root / 'src/muse_builtins/assets/bitcraze_crazyflie_2/cf2.xml'))
        source = mujoco.MjData(model)
        source.qpos[:3] = [.4, .2, .8]
        source.time = 7.5
        mujoco.mj_forward(model, source)
        raw = np.concatenate([source.qpos, source.qvel, source.act, source.ctrl, source.qfrc_applied,
            source.xfrc_applied.ravel(), source.mocap_pos.ravel(), source.mocap_quat.ravel(),
            source.eq_active, source.sensordata]).astype('<f4').tobytes()
    flags = mujoco.mjtState.mjSTATE_INTEGRATION
    original = np.empty(mujoco.mj_stateSize(model, flags))
    mujoco.mj_getState(model, source, original, flags)
    snapshot = NativeSnapshot(model)
    before = set(threading.enumerate())
    window = ui = None
    result = {}
    def selection(seq, world):
        return PreviewSelection(selection_seq=seq,
            display=DisplaySettings(enabled=True, world_id=world, rate_hz=1.))
    def frame(seq=1, world=0):
        return PreviewFrame(viewer='native', run_id='official-test', selection_seq=seq,
            world_id=world, episode=1, decision=5, tick=10, sim_time=7.5,
            scene_hash='a' * 64, nbody=model.nbody, payload_floats=snapshot.payload_floats)
    def display_restored():
        for field in ('qpos', 'qvel', 'act', 'ctrl', 'qfrc_applied', 'xfrc_applied',
                      'mocap_pos', 'mocap_quat', 'eq_active', 'sensordata'):
            np.testing.assert_allclose(getattr(window.data, field), getattr(source, field), atol=1e-6)
        assert window.data.time == 7.5
        np.testing.assert_array_equal(window.model.opt.gravity, model.opt.gravity)
        assert window.model.opt.disableflags == model.opt.disableflags
    try:
        if name == 'launch':
            # Handle已创建、初始化文字失败：构造器必须关闭这次launch的线程。
            with patch.object(mujoco.viewer.Handle, 'set_texts', side_effect=RuntimeError('injected failure')):
                try:
                    OfficialNativeViewer(snapshot, run_id='official-test')
                except RuntimeError as exc:
                    result['error'] = str(exc)
        else:
            window = OfficialNativeViewer(snapshot, run_id='official-test')
            ui = NativeUI(os.getpid())
            window.set_selection(selection(1, 0))
            assert window.sync() is None
            window.update_frame(frame(), raw)
            assert window.sync() == frame()
            assert window.sync() is None
            if name == 'mesh':
                np.testing.assert_allclose(window.handle.cam.lookat, [.4, .2, .8], atol=1e-6)
            ui.capture('initial-panels')
            if name == 'restore':
                ui.key(0xff08)
                ui.capture('reset-queued')
                assert window.sync() is None
                display_restored()
                ui.key(ord('c'), 0xffe9)
                ui.capture('control-open')
                ui.click(1155, 114)
                until(lambda: (window.handle.sync(state_only=False), abs(window.data.ctrl[0] - .2) > .1)[1])
                assert window.sync() is None
                display_restored()
                # 固定3.11原生面板：收起默认File/Option/Simulation，展开Physics。
                for key in 'fosp':
                    ui.key(ord(key), 0xffe9)
                ui.capture('physics-open')
                # 收起Algorithmic Parameters，使Gravity与Disable Flags可见。
                ui.click(120, 306)
                ui.click(120, 344)
                ui.capture('physics-short')
                ui.click(204, 374)
                ui.key(0xff50)  # Home
                for _ in range(20):
                    ui.key(0xffff)  # Delete
                for char in '1 2 -3':
                    ui.key(ord(char))
                ui.key(0xff0d)
                until(lambda: (window.handle.sync(state_only=False), not np.array_equal(window.model.opt.gravity, model.opt.gravity))[1])
                ui.capture('gravity-edited')
                assert window.sync() is None
                display_restored()
                ui.click(22, 584)
                until(lambda: (window.handle.sync(state_only=False), window.model.opt.disableflags != model.opt.disableflags)[1])
                assert window.sync() is None
                display_restored()
                with window.handle.lock():
                    window.handle.opt.geomgroup[5] = 0
                    window.model.vis.headlight.ambient[:] = [.3, .4, .5]
                    window.handle.perturb.active = 1
                    window.handle.perturb.active2 = 2
                camera = window.handle.cam.azimuth
                ui.drag(600, 450, 80, 30)
                until(lambda: window.handle.cam.azimuth != camera)
                camera = window.handle.cam.azimuth
                assert window.sync() is None
                display_restored()
                assert window.handle.cam.azimuth == camera and window.handle.opt.geomgroup[5] == 0
                assert window.handle.perturb.active == window.handle.perturb.active2 == 0
                np.testing.assert_allclose(window.model.vis.headlight.ambient, [.3, .4, .5])
                ui.capture('restored')
                result.update(restored=True)
            elif name == 'waiting':
                window.set_selection(selection(2, 1))
                assert window.sync() is None
                assert window.frame.world_id == 0
                ui.capture('waiting-world1')
                ui.click(65, 238)  # 原生Pause update
                ui.capture('pause-update')
                window.update_frame(frame(2, 1), raw)
                assert window.sync() == frame(2, 1)
                ui.minimize_on_host(True)
                minimized = frame(2, 1).model_copy(update={'decision': 6, 'tick': 12, 'sim_time': 8.5})
                window.update_frame(minimized, raw)
                assert window.sync() == minimized
                assert window.sync() is None
                ui.minimize_on_host(False)
                result.update(acks=[1, 2], old_world=0, minimized_ack=minimized.decision)
            elif name == 'mesh':
                for key in (0x20, 0xff53):
                    ui.key(key)
                ui.capture('passive-controls')
                window.handle.sync(state_only=False)
                assert window.data.time == 7.5
                window.sync()
                np.testing.assert_allclose(window.data.geom_xpos, source.geom_xpos, atol=1e-6)
                np.testing.assert_allclose(window.data.site_xpos, source.site_xpos, atol=1e-6)
                result.update(mesh_count=int(model.nmesh), passive_time=window.data.time)
            elif name == 'sync':
                with patch.object(mujoco.viewer.Handle, 'sync', side_effect=RuntimeError('injected failure')):
                    try:
                        window.sync()
                    except RuntimeError as exc:
                        result['error'] = str(exc)
            else:
                result['error'] = None
        after = np.empty_like(original)
        mujoco.mj_getState(model, source, after, flags)
        np.testing.assert_array_equal(after, original)
        result['source_unchanged'] = True
    finally:
        if window is not None:
            window.close()
        if ui is not None:
            ui.close()
    result['threads_remaining'] = len(set(threading.enumerate()) - before)
    assert result['threads_remaining'] == 0
    Path('result.json').write_text(json.dumps(result))


if __name__ == '__main__':
    child_case(sys.argv[1], sys.argv[2])
