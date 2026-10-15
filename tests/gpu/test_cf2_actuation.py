from pathlib import Path

import mujoco
import numpy as np
import pytest


ASSET = Path(__file__).parents[2] / 'src/muse_builtins/assets/bitcraze_crazyflie_2'


def test_cf2_asset_is_available():
    assert (ASSET / 'cf2.xml').is_file(), 'Crazyflie production asset has not been imported'
    assert (ASSET / 'LICENSE').is_file()


@pytest.fixture(scope='module')
def cf2_model():
    return mujoco.MjModel.from_xml_path(str(ASSET / 'cf2.xml'))


def test_cf2_hover_and_moment_sign(cf2_model):
    np.testing.assert_allclose(cf2_model.key_ctrl[0], [0.26487, 0, 0, 0])
    np.testing.assert_allclose(cf2_model.actuator_gear[:, 2:], np.diag([1, -1e-5, -1e-5, -1e-5]))
    assert abs(cf2_model.body_mass.sum() * 9.81 - 0.26487) < 1e-9
    np.testing.assert_allclose(cf2_model.body_inertia[1], [2.3951e-5, 2.3951e-5, 3.2347e-5])
    np.testing.assert_allclose(cf2_model.site_quat, [[1, 0, 0, 0], [1, 0, 0, 0]])
    np.testing.assert_allclose(cf2_model.actuator_ctrlrange, [[0, .35], [-1, 1], [-1, 1], [-1, 1]])
    assert cf2_model.ngeom == 39


@pytest.mark.parametrize('thrust, expected_vz', [(0., -.1962), (.26487, 0.)])
def test_vertical_impulse(cf2_sim, thrust, expected_vz, record_property):
    scene, sim = cf2_sim
    sim.data.ctrl[:] = 0
    sim.data.ctrl[:, 0] = thrust
    for _ in range(10):
        sim.step()
        scene.update(.002)
    sim.forward()
    np.testing.assert_allclose(sim.data.qvel[0, :3].cpu(), [0, 0, expected_vz], atol=2e-4)
    assert float(sim.data.qpos[0, 2]) > .99
    record_property('vz_mps', float(sim.data.qvel[0, 2]))


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('sign', [-1, 1])
def test_physical_torque_is_body_frame_at_nonzero_yaw(cf2_sim, axis, sign, record_property):
    import torch

    scene, sim = cf2_sim
    sim.data.qpos[:, 3:7] = torch.tensor([2**-.5, 0, 0, 2**-.5], device=sim.device)
    sim.forward()
    sim.data.ctrl[:] = 0
    sim.data.ctrl[:, axis + 1] = -sign
    for _ in range(10):
        sim.step()
        scene.update(.002)
    sim.forward()
    entity = scene['cf2']
    expected = np.zeros(3)
    expected[axis] = sign * 1e-5 / sim.mj_model.body_inertia[1, axis] * .02
    np.testing.assert_allclose(entity.data.root_link_ang_vel_b[0].cpu(), expected, rtol=.01, atol=2e-5)
    record_property('body_omega_radps', entity.data.root_link_ang_vel_b[0].tolist())
    rotation = sim.data.xmat[0, entity.indexing.root_body_id].cpu().numpy().reshape(3, 3)
    world_omega = rotation @ expected
    np.testing.assert_allclose(entity.data.root_link_ang_vel_w[0].cpu(), world_omega, rtol=.01, atol=2e-5)
    if axis == 0:
        assert sign * float(entity.data.root_link_ang_vel_w[0, 1]) > 0
        assert abs(float(entity.data.root_link_ang_vel_w[0, 0])) < 2e-5


def test_assembled_scene_preserves_physical_options(cf2_sim):
    _, sim = cf2_sim
    assert sim.mj_model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
    assert sim.mj_model.opt.density == 1.225
    assert sim.mj_model.opt.viscosity == 1.8e-5
    assert sim.mj_model.opt.timestep == .002
    assert sim.mj_model.ngeom == 40
    assert sim.mj_model.nu == 4


def test_thrust_follows_tilted_body_z(cf2_sim):
    import torch

    scene, sim = cf2_sim
    # 绕world +Y旋转90度，本体+Z指向world +X。
    sim.data.qpos[:, 3:7] = torch.tensor([2**-.5, 0, 2**-.5, 0], device=sim.device)
    sim.forward()
    sim.data.ctrl[:, 0] = .26487
    for _ in range(10):
        sim.step()
        scene.update(.002)
    np.testing.assert_allclose(sim.data.qvel[0, :3].cpu(), [.1962, 0, -.1962], atol=3e-4)
