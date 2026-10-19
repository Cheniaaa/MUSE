"""真实MuJoCo几何与GLFW相机；显示副本不得推进或写入物理。"""
import os
from pathlib import Path

import glfw
import mujoco
import numpy as np
import pytest
import viser


pytestmark = pytest.mark.skipif(os.environ.get('MUSE_TEST_NATIVE') != '1',
                                reason='Explicit GLFW window validation')


@pytest.fixture(params=['nested', 'mesh'])
def posed_model(request):
    if request.param == 'mesh':
        model = mujoco.MjModel.from_xml_path(str(Path(__file__).parents[2] /
            'src/muse_builtins/assets/bitcraze_crazyflie_2/cf2.xml'))
        assert model.nmesh > 0
    else:
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body pos="1 2 3"><freejoint/>
            <geom type="box" size=".2 .3 .4" pos=".1 .2 .3" euler="10 20 30"/>
            <body pos=".4 .5 .6" euler="30 20 10">
              <geom type="box" size=".1 .2 .3" pos=".2 0 .1" euler="0 40 0"/>
              <site size=".1"/>
            </body>
          </body></worldbody></mujoco>''')
    reference = mujoco.MjData(model)
    reference.qpos[:7] = [2., -1., 4., .5, .5, .5, .5]
    mujoco.mj_forward(model, reference)
    return model, reference


def test_native_scene_matches_nested_and_mesh_pose(posed_model, monkeypatch):
    from muse_sdk.run_protocol import SceneObject
    from muse_server.viewers.scene_interaction import SceneInteraction
    from muse_server.viewers.native_process import NativeWindow

    model, reference = posed_model
    server = viser.ViserServer(host='127.0.0.1', port=0, verbose=False)
    interaction = window = None
    try:
        interaction = SceneInteraction(server, model, [SceneObject(object_id='body', label='body',
            kind='entity', body_id=1, geom_ids=list(range(model.ngeom)), group_id=None,
            group_slot=None, role_id=None, model_id=None)], emit=lambda _: None)
        window = NativeWindow(model, title='MUSE test')
        initial = {name: getattr(window.data, name).copy() for name in ('qpos', 'ctrl', 'xfrc_applied')}
        def forbidden(*args, **kwargs):
            pytest.fail('Rendering must not call physics forward/step')
        monkeypatch.setattr(mujoco, 'mj_forward', forbidden)
        monkeypatch.setattr(mujoco, 'mj_step', forbidden)
        xpos, xmat = reference.xpos.astype('float32'), reference.xmat.astype('float32')
        frame = dict(world_id=1, episode=2, tick=100, sim_time=2.)
        interaction.update_frame(xpos, xmat, frame)
        window.update_frame(xpos, xmat, frame)
        window.render()
        for data in (interaction._data, window.data):
            np.testing.assert_allclose(data.geom_xpos, reference.geom_xpos, atol=1e-6)
            np.testing.assert_allclose(data.geom_xmat, reference.geom_xmat, atol=1e-6)
        geoms = {geom.objid: geom for geom in window.scene.geoms[:window.scene.ngeom]
                 if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM}
        assert set(geoms) == set(range(model.ngeom))
        for index, geom in geoms.items():
            np.testing.assert_allclose(geom.pos, reference.geom_xpos[index], atol=1e-6)
            np.testing.assert_allclose(geom.mat.reshape(9), reference.geom_xmat[index], atol=1e-6)
        assert window.data.time == 0
        for name, expected in initial.items():
            np.testing.assert_array_equal(getattr(window.data, name), expected)
        window.clear_frame()
        window.render()
        assert window.scene.ngeom == 0
    finally:
        if window is not None:
            window.close()
        if interaction is not None:
            interaction.close()
        server.stop()


def test_native_camera_callbacks_do_not_write_physics(posed_model, monkeypatch):
    from muse_server.viewers.native_process import NativeWindow

    model, reference = posed_model
    window = NativeWindow(model, title='MUSE camera test')
    try:
        window.update_frame(reference.xpos, reference.xmat,
                            dict(world_id=0, episode=0, tick=0, sim_time=0.))
        window.render()
        initial = {name: getattr(window.data, name).copy() for name in ('qpos', 'ctrl', 'xfrc_applied')}
        cursor = glfw.set_cursor_pos_callback(window.window, None)
        scroll = glfw.set_scroll_callback(window.window, None)
        assert cursor is not None and scroll is not None
        cursor(window.window, 100., 100.)
        monkeypatch.setattr(glfw, 'get_mouse_button',
                            lambda _, button: glfw.PRESS if button == glfw.MOUSE_BUTTON_LEFT else glfw.RELEASE)
        before = window.camera.azimuth, window.camera.elevation
        cursor(window.window, 160., 140.)
        assert (window.camera.azimuth, window.camera.elevation) != before
        before = window.camera.distance
        scroll(window.window, 0., 1.)
        assert window.camera.distance != before
        monkeypatch.setattr(glfw, 'get_mouse_button',
                            lambda _, button: glfw.PRESS if button == glfw.MOUSE_BUTTON_RIGHT else glfw.RELEASE)
        before = window.camera.lookat.copy()
        cursor(window.window, 190., 180.)
        assert not np.array_equal(window.camera.lookat, before)
        window.render()
        for name, expected in initial.items():
            np.testing.assert_array_equal(getattr(window.data, name), expected)
        assert window.data.time == 0
        glfw.set_window_should_close(window.window, True)
        assert not window.is_running()
    finally:
        window.close()
