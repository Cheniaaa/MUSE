import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.utils.lab_api.math import matrix_from_quat
from muse_builtins.controllers import VelocityController, VelocityControllerConfig


@pytest.fixture(scope='module')
def tracking_report(cf2_scene_factory):
    root = Path(__file__).parents[2]
    config = json.loads((root / 'examples/basic_flight/controller.json').read_text())
    commands = torch.tensor([
        [0, 0, 0, 0], [.1, 0, 0, 0], [-.1, 0, 0, 0],
        [0, .1, 0, 0], [0, -.1, 0, 0], [0, 0, .1, 0],
        [0, 0, -.1, 0], [0, 0, 0, .1], [0, 0, 0, -.1],
    ], device='cuda:0')
    scene, sim = cf2_scene_factory(len(commands))
    return run_tracking(scene, sim, commands[:, None], config, root / 'docs/validation/m1-tracking')


@torch.no_grad()
def run_tracking(scene, sim, commands, config, output, initial_yaw=0.):
    worlds, count = commands.shape[:2]
    sim.reset()
    scene.reset()
    for slot in range(count):
        entity = scene['cf2' if count == 1 else f'cf2_{slot}']
        state = entity.data.default_root_state.clone()
        state[:, 2] = 3.
        state[:, 3] = np.cos(initial_yaw / 2)
        state[:, 6] = np.sin(initial_yaw / 2)
        entity.write_root_state_to_sim(state)
    sim.forward()
    controller = VelocityController(VelocityControllerConfig(**config['config']),
                                    {'mass_kg': sim.mj_model.body_mass[1], 'gravity_mps2': 9.81},
                                    (worlds, count), sim.device)
    qpos = sim.data.qpos[:].view(worlds, count, 7)
    qvel = sim.data.qvel[:].view(worlds, count, 6)
    controller.reset(torch.arange(worlds, device=sim.device), qpos[..., 3:7])
    gear = torch.tensor(sim.mj_model.actuator_gear[:, 2:].reshape(count, 4, 4).diagonal(axis1=1, axis2=2).copy(),
                        dtype=torch.float32, device=sim.device)
    limits = torch.tensor(sim.mj_model.actuator_ctrlrange.reshape(count, 4, 2), dtype=torch.float32, device=sim.device)
    trajectory = torch.empty((5000, worlds, count, 13), device=sim.device)
    controls = torch.empty((5000, worlds, count, 4), device=sim.device)
    saturated = torch.zeros((worlds, count, 4), device=sim.device)
    contact_count = 0
    max_constraints = 0
    for tick in range(5000):
        wrench = controller.compute(commands[..., :3], commands[..., 3],
                                     qpos[..., 3:7], qvel[..., :3], qvel[..., 3:6], .002)
        raw_ctrl = wrench / gear
        saturated += ((raw_ctrl < limits[..., 0]) | (raw_ctrl > limits[..., 1]))
        ctrl = raw_ctrl.clamp(limits[..., 0], limits[..., 1])
        scene.write_data_to_sim()
        sim.data.ctrl[:] = ctrl.reshape(worlds, -1)
        sim.step()
        scene.update(.002)
        trajectory[tick, ..., :7] = qpos
        trajectory[tick, ..., 7:] = qvel
        controls[tick] = ctrl
        contact_count += int(sim.data.nacon[0])
        max_constraints = max(max_constraints, int(sim.data.nefc[:].max()))
        assert int(sim.data.nacon[0]) <= sim.data.naconmax
        assert max_constraints <= sim.data.njmax

    rotation = matrix_from_quat(trajectory[..., 3:7])
    body_velocity = (rotation.transpose(-1, -2) @ trajectory[..., 7:10, None]).squeeze(-1)
    velocity_rmse = ((body_velocity[2500:] - commands[None, ..., :3])**2).mean(0).sqrt()
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    yaw_delta = (yaw[1:] - yaw[:-1] + torch.pi).remainder(2 * torch.pi) - torch.pi
    yaw_rate = yaw_delta / .002
    yaw_rmse = ((yaw_rate[2499:] - commands[None, ..., 3])**2).mean(0).sqrt()
    report = {
        'finite': bool(torch.isfinite(trajectory).all() & torch.isfinite(controls).all()),
        'collision_count': contact_count,
        'max_constraints': max_constraints,
        'max_hover_height_drift_m': float((trajectory[:, 0, 0, 2] - 3.).abs().max()),
        'worst_tail_velocity_rmse_mps': float(velocity_rmse.max()),
        'worst_tail_yaw_rate_rmse_radps': float(yaw_rmse.max()),
        'velocity_rmse_by_trial': velocity_rmse.tolist(),
        'yaw_rate_rmse_by_trial': yaw_rmse.tolist(),
        'peak_tilt_rad': float(rotation[..., 2, 2].clamp(-1, 1).acos().max()),
        'saturation_fraction_by_trial_axis': (saturated / 5000).tolist(),
        'actuator_limits_unchanged': np.array_equal(sim.mj_model.actuator_ctrlrange,
                                                   np.tile([[0, .35], [-1, 1], [-1, 1], [-1, 1]], (count, 1))),
        'commands': commands.tolist(),
    }
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(output.with_suffix('.npz'), state=trajectory.cpu().numpy(),
                        ctrl=controls.cpu().numpy(), commands=commands.cpu().numpy(), dt=.002)
    return report


def test_velocity_controller_tracks_with_original_limits(tracking_report):
    assert tracking_report['finite']
    assert tracking_report['collision_count'] == 0
    assert tracking_report['max_hover_height_drift_m'] <= .1
    assert tracking_report['worst_tail_velocity_rmse_mps'] <= .05, tracking_report
    assert tracking_report['worst_tail_yaw_rate_rmse_radps'] <= .05, tracking_report
    assert tracking_report['actuator_limits_unchanged']


def test_three_entities_track_mixed_body_commands_at_nonzero_yaw(cf2_scene_factory):
    root = Path(__file__).parents[2]
    config = json.loads((root / 'examples/basic_flight/controller.json').read_text())
    commands = torch.tensor([
        [[0, 0, 0, 0], [.1, .1, .1, .1], [-.1, -.1, -.1, -.1]],
        [[.1, -.1, .1, -.1], [-.1, .1, -.1, .1], [.1, .1, -.1, -.1]],
    ], device='cuda:0')
    scene, sim = cf2_scene_factory(2, count=3)
    report = run_tracking(scene, sim, commands, config, root / 'docs/validation/m1-mixed-tracking', np.pi / 2)
    assert report['finite']
    assert report['collision_count'] == 0
    assert report['max_hover_height_drift_m'] <= .1
    assert report['worst_tail_velocity_rmse_mps'] <= .05, report
    assert report['worst_tail_yaw_rate_rmse_radps'] <= .05, report
    assert report['actuator_limits_unchanged']
