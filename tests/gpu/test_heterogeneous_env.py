"""原参数car与Crazyflie的生产环境控制、接触、重置及学习。"""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch
from flight_helpers import archive


@pytest.fixture(scope='module')
def mixed_plan(tmp_path_factory):
    from muse_builtins.metadata import export_metadata
    from muse_server.catalog import Catalog
    from muse_server.resolver import resolve_experiment
    root = tmp_path_factory.mktemp('mixed-env')
    examples = Path(__file__).resolve().parents[2] / 'examples'
    metadata = root / 'builtin.json'
    metadata.write_text(export_metadata(examples / 'basic_flight/controller.json').model_dump_json())
    catalog = Catalog(root / 'catalog', metadata)
    example = examples / 'heterogeneous'
    environment = root / 'environment'
    shutil.copytree(example / 'environment', environment)
    scene = environment / 'scene/scene.mjcf'
    scene.write_text(scene.read_text().replace('</worldbody>',
        '<geom name="wheel_obstacle" type="sphere" pos="2 .073 .03" size=".006"/></worldbody>'))
    request = json.loads((example / 'request_train.json').read_text())
    for kind, folder in [('environment', 'environment'), ('training_package', 'training_package')]:
        directory = environment if kind == 'environment' else example / folder
        resource = catalog.register_archive(kind, archive(directory, root / f'{kind}.zip'))
        request[kind + '_id'], request[kind + '_revision'] = resource.resource_id, resource.revision
    request['term_config'] = {'world': {'termination': {'episode_limit': {'config': {'episode_max_steps': 500}}}}}
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment, source=request)
    return catalog.create_run_plan(identity, mode='TRAIN', device='cuda:0', max_decision_steps=8)


@pytest.fixture(scope='module')
def mixed_env(mixed_plan, gpu_stream):
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    env = MuseEnv(mixed_plan.experiment, RuntimeContext(mixed_plan.device, mixed_plan.resource_paths, {}),
                  WorldRng(7, 2, mixed_plan.device))
    yield env
    env.close()


def commands(env, velocity=0., yaw_rate=0.):
    from muse_sdk.runtime_types import MotionAction
    return {'scout': MotionAction(torch.zeros(2, 1, 4, device=env.device)),
            'rover': MotionAction(torch.tensor([velocity, yaw_rate], device=env.device).expand(2, 1, 2).clone())}


@pytest.mark.parametrize('velocity,yaw_rate', [(.1, 0.), (-.1, 0.), (0., .8), (0., -.8), (.1, .8)])
def test_mixed_drive_tracking_and_stop(mixed_env, velocity, yaw_rate):
    env = mixed_env
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))
    uav_start = env.role_positions_world['scout'].clone()
    for _ in range(60):
        output = env.step(commands(env, velocity, yaw_rate))
        assert not output.transition.world_reset.any()
    quat = env.physics_outputs.orientation_wxyz['rover']
    yaw = torch.atan2(2*(quat[..., 0]*quat[..., 3]+quat[..., 1]*quat[..., 2]),
                      1-2*(quat[..., 2].square()+quat[..., 3].square()))
    vel = env.physics_outputs.velocity_world['rover']
    forward = vel[..., 0]*yaw.cos()+vel[..., 1]*yaw.sin()
    torch.testing.assert_close(forward, torch.full_like(forward, velocity), atol=.01, rtol=0)
    omega = env.physics_outputs.angular_velocity_body['rover'][..., 2]
    torch.testing.assert_close(omega, torch.full_like(omega, yaw_rate), atol=.15, rtol=0)
    torch.testing.assert_close(env.role_positions_world['scout'], uav_start, atol=.02, rtol=0)
    for _ in range(25):
        output = env.step(commands(env))
        assert not output.transition.world_reset.any()
    assert env.physics_outputs.velocity_world['rover'].abs().max() < .01
    assert env.physics_outputs.angular_velocity_body['rover'].abs().max() < .05


def test_mixed_subset_reset_clears_wheels_and_preserves_other_world(mixed_env):
    env = mixed_env
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))
    for _ in range(3):
        env.step(commands(env, .1, .8))
    before = {name: getattr(env.sim.data, name)[1].clone() for name in ('qpos', 'qvel', 'ctrl')}
    history = env.action_manager.action[1].clone()
    env.reset(torch.tensor([True, False], device=env.device))
    for name, expected in before.items():
        assert torch.equal(getattr(env.sim.data, name)[1], expected)
    assert torch.equal(env.action_manager.action[1], history)
    assert env.action_manager.action[0].count_nonzero() == 0
    assert env.sim.data.qvel[0].count_nonzero() == 0
    assert env.sim.data.ctrl[0].count_nonzero() == 0
    assert env.policy_input.observations.actors['rover']['previous_action'][0].count_nonzero() == 0


@pytest.mark.parametrize('role,rollover', [('scout', False), ('rover', False), ('rover', True)])
def test_mixed_failures_end_whole_world_only(mixed_env, role, rollover):
    env = mixed_env
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))
    term = env.action_manager.get_term(env.experiment.bindings[role].group_id)
    indices = term.qpos_ids[0]
    if rollover:
        env.sim.data.qpos[0, indices[3:7]] = torch.tensor([0., 1., 0., 0.], device=env.device)
    else:
        env.sim.data.qpos[0, indices[2]] = 0.
    env.sim.forward()
    output = env.step(commands(env))
    assert output.transition.world_terminated.tolist() == [True, False]
    assert output.transition.world_reset.tolist() == [True, False]
    assert env.episode_length_buf.tolist() == [0, 1]
    assert env.policy_input.observations.actors['rover']['previous_action'][0].count_nonzero() == 0


def test_mixed_heads_both_learn(mixed_env):
    from muse_builtins.algorithms.ippo import IPPO
    from muse_runtime.driver import run
    env = mixed_env
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))
    algorithm = IPPO(env.experiment.algorithm_config)
    algorithm.initialize(env.experiment, SimpleNamespace(device=env.device), env.rng)
    before = {role: torch.cat([p.detach().flatten() for p in child.actor.parameters()]).clone()
              for role, child in algorithm.children.items()}
    result = run(env, algorithm, lambda: None, lambda *args: None,
                 mode='TRAIN', collection_steps=4, max_decision_steps=8)
    assert result == dict(state='COMPLETED', steps=8, updates=2)
    for role, child in algorithm.children.items():
        assert not torch.equal(before[role], torch.cat([p.detach().flatten() for p in child.actor.parameters()]))
    assert algorithm.children['scout'].storage.actions.shape[-1] == 4
    assert algorithm.children['rover'].storage.actions.shape[-1] == 2


def test_wheel_obstacle_contact_is_not_mistaken_for_support(mixed_env):
    env = mixed_env
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))
    term = env.action_manager.get_term('rover_group')
    env.sim.data.qpos[0, term.qpos_ids[0, 0]] = 2.07
    env.sim.forward()
    contact = env.sim.data.contact
    active = (torch.arange(contact.worldid.shape[0], device=env.device) < env.sim.data.nacon[0])
    active &= (contact.worldid == 0) & (contact.dist <= 0)
    model = env.sim.mj_model
    obstacle = model.geom('environment/wheel_obstacle').id
    pairs = contact.geom[active & (contact.geom == obstacle).any(-1)]
    assert pairs.numel() > 0
    assert (pairs == model.geom('rover_group_0/left_wheel').id).any()
    assert not (pairs == model.geom('rover_group_0/chasis').id).any()
    output = env.step(commands(env))
    assert output.transition.world_terminated.tolist() == [True, False]


def test_mixed_timeout_resets_both_entities(mixed_plan, gpu_stream):
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    exp = mixed_plan.experiment.model_copy(deep=True)
    next(s for s in exp.selections if s.time_out).config['episode_max_steps'] = 4
    env = MuseEnv(exp, RuntimeContext('cuda:0', mixed_plan.resource_paths, {}), WorldRng(7, 2, 'cuda:0'))
    try:
        initial = env.sim.data.qpos.clone()
        for _ in range(4):
            output = env.step(commands(env, .1))
        assert output.transition.world_truncated.tolist() == [True, True]
        torch.testing.assert_close(env.sim.data.qpos[:], initial, rtol=0, atol=0)
        assert env.sim.data.qvel.count_nonzero() == 0
        assert env.action_manager.action.count_nonzero() == 0
    finally:
        env.close()
