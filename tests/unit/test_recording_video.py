from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('times,fps,expected', [
    ([0., .12, .25], 10., [0, 0, 1, 2]), ([0.], 25., [0]),
    ([0., 0.], 25., [1]), ([.4, .5], 10., [0, 1]),
    ([0., .28], 25., [0, 0, 0, 0, 0, 0, 0, 1]),
])
def test_video_zero_order_hold_includes_last_sample(times, fps, expected):
    from muse_server.viewers.recording_process import video_sample_indices
    assert list(video_sample_indices(np.asarray(times), fps)) == expected


def test_camera_roll_and_vertical_fov_are_preserved():
    import mujoco
    from muse_server.viewers.recording_process import apply_camera
    scene = mujoco.MjvScene()
    camera = SimpleNamespace(position=(1., 2., 3.), wxyz=(2**-.5, 0., 0., 2**-.5),
                             fov=np.pi / 2, near=.1, far=50.)
    apply_camera(scene, camera)
    for eye in scene.camera:
        np.testing.assert_allclose(eye.pos, [1., 2., 3.])
        np.testing.assert_allclose(eye.forward, [0., 0., 1.], atol=1e-7)
        np.testing.assert_allclose(eye.up, [1., 0., 0.], atol=1e-7)
        assert eye.frustum_top == pytest.approx(.1)
        assert eye.frustum_bottom == pytest.approx(-.1)
        assert eye.frustum_far == 50.
