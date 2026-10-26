"""合作 MAPPO 的真实混合物理、Task、recurrent 与恢复边界。"""
import copy
from types import SimpleNamespace

import pytest
import torch
from muse_sdk.run_protocol import RunRequest

from mappo_helpers import cooperative_pipeline, example_definition
from muse_runtime.checkpoint import snapshot_learning_state


@pytest.fixture
def cooperative_plan(pipeline, tmp_path):
    from muse_server.resolver import resolve_experiment
    catalog, request = cooperative_pipeline(pipeline, tmp_path)
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment, source=request)
    return catalog.create_run_plan(RunRequest(experiment_id=identity, mode='TRAIN', max_decision_steps=8), device='cuda:0')


@pytest.fixture
def cooperative_env(cooperative_plan, gpu_stream):
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    definition = example_definition()
    env = MuseEnv(cooperative_plan.experiment,
        RuntimeContext('cuda:0', cooperative_plan.resource_paths, definition.terms,
                       definition.tasks['team_progress']), WorldRng(7, 2, 'cuda:0'))
    yield env
    env.close()


def zero_actions(env):
    from muse_sdk.runtime_types import MotionAction
    return {r: MotionAction(torch.zeros(spec.shape, device=env.device))
            for r, spec in env.experiment.action_specs.items()}


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_state(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_state(a, b)
    else:
        assert left == right


def test_task_learning_boundary_and_subset_reset(cooperative_env):
    env = cooperative_env
    assert env.task.phase.shape == (2, 1) and not env.task.phase.any()
    for _ in range(3):
        out = env.step(zero_actions(env))
    assert out.transition.learning_done['scout'].all()
    assert not out.transition.learning_done['rover'].any()
    assert not out.transition.world_reset.any()
    assert all(mask.controllable.all() for mask in out.policy_role_masks.values())
    before = env.task.phase[1].clone()
    qpos = env.sim.data.qpos[1].clone()
    env.reset(torch.tensor([True, False], device=env.device))
    assert not env.task.phase[0].any()
    assert torch.equal(env.task.phase[1], before)
    assert torch.equal(env.sim.data.qpos[1], qpos)
    assert not env.learning_done['scout'][0].any()
    out = env.step(zero_actions(env))
    assert not out.transition.learning_done['scout'].any()


def test_terminal_task_diagnostics_survive_reset(cooperative_env):
    env = cooperative_env
    update = env.task.update
    def with_diagnostics(current):
        return update(current)._replace(diagnostics={'phase': env.task.phase})
    env.task.update = with_diagnostics
    env.episode_length_buf.fill_(7)
    out = env.step(zero_actions(env))
    assert out.transition.world_truncated.all()
    assert not env.task.phase.any()
    torch.testing.assert_close(out.diagnostics['task']['phase'], torch.full((2, 1), 8., device=env.device))


def test_privileged_input_changes_only_central_critic(cooperative_env):
    from muse_builtins.algorithms.ippo import IPPO, IPPOConfig
    from muse_builtins.algorithms.mappo import MAPPO
    env = cooperative_env
    obs = env.policy_input.observations
    changed = copy.deepcopy(obs)
    for fields in changed.critic_inputs.values():
        for value in fields.values():
            value.add_(10)
    central = MAPPO(env.experiment.algorithm_config)
    central.initialize(env.experiment, env.context, env.rng)
    local = IPPO(IPPOConfig(**{k: v for k, v in env.experiment.algorithm_config.items()
                              if k in IPPOConfig.model_fields}).model_dump())
    local.initialize(env.experiment, env.context, env.rng)
    with torch.no_grad():
        for owner, child in central.children.items():
            first, second = central._observations(obs, owner), central._observations(changed, owner)
            torch.testing.assert_close(child.actor(first), child.actor(second), rtol=0, atol=0)
            assert not torch.equal(child.critic(first), child.critic(second))
        for role, child in local.children.items():
            first, second = local._observations(obs, role), local._observations(changed, role)
            torch.testing.assert_close(child.critic(first), child.critic(second), rtol=0, atol=0)


@pytest.mark.parametrize('fault', ['extra_role', 'missing_role', 'dtype', 'shape', 'status', 'published_nonfinite'])
def test_task_output_rejected_before_status_commit(cooperative_env, fault):
    from muse_sdk.runtime_types import TaskUpdateResult
    env = cooperative_env
    original = env.task.update
    def malformed(current):
        result = original(current)
        done = {'scout': torch.ones(2, 1, dtype=torch.bool, device=env.device)}
        status = {}
        if fault == 'extra_role':
            done['rover'] = torch.ones(2, 2, dtype=torch.bool, device=env.device)
        elif fault == 'missing_role':
            done = {}
        elif fault == 'dtype':
            done['scout'] = done['scout'].float()
        elif fault == 'shape':
            done['scout'] = done['scout'].flatten()
        elif fault == 'status':
            status['scout'] = SimpleNamespace(alive=done['scout'], controllable=done['scout'])
        else:
            env.task.phase.fill_(float('nan'))
        return TaskUpdateResult(status, done, result.diagnostics)
    env.task.update = malformed
    with pytest.raises((ValueError, RuntimeError), match='Task|task'):
        env.step(zero_actions(env))
    assert not env.learning_done['scout'].any()


@pytest.mark.parametrize('recurrent,sharing', [(None, True), (None, False), ('gru', True), ('lstm', False)])
def test_real_mappo_updates_and_frozen_evaluation(cooperative_env, recurrent, sharing):
    from muse_builtins.algorithms.mappo import MAPPO
    from muse_runtime.driver import run
    env = cooperative_env
    exp = env.experiment.model_copy(deep=True)
    exp.algorithm_config.update(recurrent=None if recurrent is None else {'type': recurrent, 'hidden_size': 16},
                                share_actor_parameters=sharing)
    learner = MAPPO(exp.algorithm_config)
    learner.initialize(exp, env.context, env.rng)
    before = snapshot_learning_state(learner.learning_state_dict())
    result = run(env, learner, lambda: None, lambda *args: None, mode='TRAIN',
                 collection_steps=4, max_decision_steps=8)
    assert result == dict(state='COMPLETED', steps=8, updates=2)
    assert len(learner.children) == (2 if sharing else 3)
    saved = snapshot_learning_state(learner.learning_state_dict())
    for owner in learner.children:
        for kind in ('actor', 'critic'):
            assert any(not torch.equal(v, saved['roles'][owner][kind][k])
                       for k, v in before['roles'][owner][kind].items())
    fresh = MAPPO(exp.algorithm_config)
    fresh.initialize(exp, env.context, env.rng)
    fresh.load_learning_state_dict(saved)
    equal_state(saved, fresh.learning_state_dict())
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))
    frozen = snapshot_learning_state(fresh.learning_state_dict())
    result = run(env, fresh, lambda: None, lambda *args: None, mode='EVALUATE',
                 collection_steps=4, max_decision_steps=9)
    assert result['updates'] == 0 and result['steps'] == 9
    equal_state(frozen, fresh.learning_state_dict())


@pytest.mark.parametrize('recurrent', ['gru', 'lstm'])
def test_horizon_timeout_uses_terminal_central_input(cooperative_env, recurrent):
    from muse_builtins.algorithms.mappo import MAPPO
    env = cooperative_env
    exp = env.experiment.model_copy(deep=True)
    exp.algorithm_config.update(recurrent={'type': recurrent, 'hidden_size': 8})
    learner = MAPPO(exp.algorithm_config)
    learner.initialize(exp, env.context, env.rng)
    # 把episode尾部放在H末；Task边界已由其他测试独立验证。
    env.episode_length_buf.fill_(4)
    for step in range(4):
        result = learner.act(env.policy_input.observations, env.role_masks, None, False)
        out = env.step(result.role_actions)
        expected = {}
        for owner, child in learner.children.items():
            hidden = copy.deepcopy(child.critic.get_hidden_state())
            with torch.no_grad():
                expected[owner] = child.critic(learner._observations(out.transition.pre_reset_next_observations, owner)).clone()
            child.critic.reset(hidden_state=hidden)
        learner.process_env_step(out.transition, result.act_aux)
        for owner, child in learner.children.items():
            torch.testing.assert_close(child.storage.final_values[step], expected[owner])
    assert out.transition.world_truncated.all()
    assert env.task.phase.count_nonzero() == 0
    for child in learner.children.values():
        assert child.storage.truncated[-1].all()
        hidden = child.critic.get_hidden_state()
        assert all(not h.any() for h in (hidden if isinstance(hidden, tuple) else (hidden,)))
    learner.update()
