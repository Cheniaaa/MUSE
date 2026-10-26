"""真实Env/native history的slot轴、时间顺序与局部/终态reset。"""
from dataclasses import replace

import pytest
import torch
from muse_sdk.run_protocol import RunRequest

from mappo_helpers import cooperative_pipeline, example_definition
from test_mappo_env import zero_actions


@pytest.fixture
def history_env(pipeline, tmp_path, gpu_stream):
    from muse_server.resolver import resolve_experiment
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    catalog, request = cooperative_pipeline(pipeline, tmp_path)
    request['term_config'] = {
        'roles': {'rover': {'observation': {'target': {'history_length': 4}}}},
        'critic_inputs': {'team': {'observation': {
            'rover_target': {'history_length': 4}, 'phase': {'history_length': 4}}}}}
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment, source=request)
    plan = catalog.create_run_plan(RunRequest(experiment_id=identity, mode='TRAIN', max_decision_steps=8), device='cuda:0')
    definition = example_definition()
    cpu_rng, gpu_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    env = MuseEnv(plan.experiment, RuntimeContext('cuda:0', plan.resource_paths, definition.terms,
        definition.tasks['team_progress']), WorldRng(7, 2, 'cuda:0'))
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    yield env
    env.close()


def raw_target(env):
    return (env.commands.target_world['rover'] - env.role_positions_world['rover'])[..., :2].clone()


def test_history_keeps_slots_scope_and_oldest_to_newest(history_env):
    env = history_env
    frames = [raw_target(env)]*4
    torch.testing.assert_close(env.observations.actors['rover']['target'], torch.stack(frames, dim=2))
    assert env.observations.actors['rover']['target'].shape == (2, 2, 4, 2)
    assert env.observations.critic_inputs['team']['rover_target'].shape == (2, 2, 4, 2)
    assert env.observations.critic_inputs['team']['phase'].shape == (2, 4, 1)
    assert env.observations.actors['rover']['velocity'].shape == (2, 2, 2)
    for step in range(1, 4):
        action = zero_actions(env)
        action['rover'].motion[..., 0] = .5*env.experiment.action_specs['rover'].maximum[0]
        out = env.step(action)
        assert not out.transition.world_reset.any()
        frames = frames[1:] + [raw_target(env)]
        expected = torch.stack(frames, dim=2)
        torch.testing.assert_close(out.policy_observations.actors['rover']['target'], expected)
        torch.testing.assert_close(out.policy_observations.critic_inputs['team']['rover_target'], expected)
        phase = torch.tensor([0.]*(4-step) + list(map(float, range(1, step+1))), device=env.device)
        torch.testing.assert_close(out.policy_observations.critic_inputs['team']['phase'],
                                   phase[None, :, None].expand(2, -1, -1))


def test_history_partial_reset_terminal_snapshot_and_raw_boundary(history_env):
    env = history_env
    for _ in range(3):
        env.step(zero_actions(env))
    other = env.observations.critic_inputs['team']['phase'][1].clone()
    env.reset(torch.tensor([True, False], device=env.device))
    torch.testing.assert_close(env.observations.critic_inputs['team']['phase'][1], other)
    assert not env.observations.critic_inputs['team']['phase'][0].any()
    env.episode_length_buf[0] = 7
    out = env.step(zero_actions(env))
    assert out.transition.world_truncated.tolist() == [True, False]
    terminal = out.transition.pre_reset_next_observations.critic_inputs['team']['phase']
    torch.testing.assert_close(terminal[:, :, 0], torch.tensor([[0., 0., 0., 8.], [1., 2., 3., 4.]], device=env.device))
    assert not out.policy_observations.critic_inputs['team']['phase'][0].any()
    torch.testing.assert_close(out.policy_observations.critic_inputs['team']['phase'][1], terminal[1])
    # 公开history不能放松底层Term形状契约；通过真实Term返回边界暴露错误。
    cfg = env.observation_manager.cfg['roles.rover'].terms['target']
    implementation = env.term_registry['builtin:planar_target'].implementation
    # checked闭包捕获原实现；直接替换配置会绕开检查，因此验证真实manager的构造边界。
    from mjlab.managers.observation_manager import ObservationGroupCfg
    from muse_runtime.managers import CheckedObservationManager
    bad = replace(cfg, func=lambda current, **params: implementation(current, **params)[..., :1])
    with pytest.raises(ValueError, match='shape/dtype'):
        CheckedObservationManager({'raw': ObservationGroupCfg(terms={'target': bad}, concatenate_terms=False)}, env)
