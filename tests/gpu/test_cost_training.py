"""真实 cost、非共享 GRU MAPPO、Runtime checkpoint 与冻结评估。"""
import copy
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from mappo_helpers import EXAMPLE, cooperative_pipeline
from muse_runtime.checkpoint import snapshot_learning_state
from test_checkpoint_learning import assert_learning_state_equal


@contextmanager
def initialized(plan):
    from muse_builtins.algorithms.mappo import MAPPO
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.package_loader import load_package
    from muse_runtime.rng import WorldRng
    with load_package(Path(plan.resource_paths['training_package'])) as package:
        exp = plan.experiment
        rng = WorldRng(exp.root_seed, exp.topology.world_count, plan.device)
        env = MuseEnv(exp, RuntimeContext(plan.device, plan.resource_paths, package.definition.terms,
                                         package.definition.tasks['team_progress']), rng)
        try:
            learner = MAPPO(exp.algorithm_config)
            learner.initialize(exp, env.context, rng)
            yield env, learner
        finally:
            env.close()


def sampling_state(learner):
    return {key: copy.deepcopy((cost.episode_sum, cost.episode_discount, cost.episode_valid,
                               cost.completed_sum, cost.completed_count, cost.critic.get_hidden_state(),
                               cost.storage.step, cost.storage.pending))
            for key, cost in learner.cost_state.items()}


def assert_updated(before, after):
    for owner, old in before['roles'].items():
        for kind in ('actor', 'critic'):
            assert any(not torch.equal(value, after['roles'][owner][kind][name])
                       for name, value in old[kind].items())
    assert len(after['costs']) == 2
    for key, old in before['costs'].items():
        new = after['costs'][key]
        assert any(not torch.equal(value, new['critic'][name]) for name, value in old['critic'].items())
        assert not torch.equal(old['lambda'], new['lambda'])
        assert new['lambda'] >= 0 and torch.isfinite(new['lambda'])
        assert new['optimizer']['state']


def test_real_cost_training_checkpoint_new_episode_and_eval_freeze(pipeline, tmp_path, gpu_stream):
    from muse_runtime.checkpoint import load_checkpoint, publish_checkpoint
    from muse_runtime.driver import run
    from muse_sdk.run_protocol import CheckpointReference
    from muse_server.resolver import resolve_experiment

    catalog, baseline = cooperative_pipeline(pipeline, tmp_path)
    assert (EXAMPLE / 'request_cost.json').is_file(), 'Missing real cost example'
    request = json.loads((EXAMPLE / 'request_cost.json').read_text())
    for key in ('environment_id', 'environment_revision', 'training_package_id', 'training_package_revision'):
        request[key] = baseline[key]
    request['algorithm_config'].update(share_actor_parameters=False, recurrent={'type': 'gru', 'hidden_size': 16})
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment)
    source = catalog.create_run_plan(identity, mode='TRAIN', device='cuda:0', max_decision_steps=12)
    with initialized(source) as (env, learner):
        assert set(learner.children) == {'scout/slot/0', 'rover/slot/0', 'rover/slot/1'}
        assert set(learner.cost_state) == {('scout', 'scout_squared_error'), (None, 'team_squared_error')}
        before = snapshot_learning_state(learner.learning_state_dict())
        result = run(env, learner, lambda: None, lambda *args: None, mode='TRAIN',
                     collection_steps=4, max_decision_steps=12)
        assert result == dict(state='COMPLETED', steps=12, updates=3)
        saved = snapshot_learning_state(learner.learning_state_dict())
        assert_updated(before, saved)
        # 12步在8步timeout后留下未结束episode，证明恢复必须丢弃采样累计。
        assert all(cost.episode_sum.any() for cost in learner.cost_state.values())
        artifact = publish_checkpoint(source, learner, steps=12, updates=3)

    target = catalog.create_run_plan(identity, mode='TRAIN', device='cuda:0', max_decision_steps=8)
    target.checkpoint = CheckpointReference(artifact_id=artifact.id,
        path=str(Path(source.artifact_directory) / artifact.path), size=artifact.size, sha256=artifact.sha256)
    with initialized(target) as (env, learner):
        load_checkpoint(target, learner)
        assert_learning_state_equal(saved, learner.learning_state_dict())
        assert env.common_step_counter == 0 and not env.episode_length_buf.any()
        for cost in learner.cost_state.values():
            assert not cost.episode_sum.any() and not cost.episode_valid.any()
            assert (cost.episode_discount == 1).all()
            assert cost.completed_count == 0 and cost.completed_sum == 0
            assert cost.storage.step == 0 and cost.storage.pending is None
            assert not cost.critic.get_hidden_state().any()
        result = run(env, learner, lambda: None, lambda *args: None, mode='TRAIN',
                     collection_steps=4, max_decision_steps=8)
        assert result == dict(state='COMPLETED', steps=8, updates=2)
        continued = snapshot_learning_state(learner.learning_state_dict())
        assert continued['updates'] == 5
        assert_updated(saved, continued)

        frozen_sampling = sampling_state(learner)
        result = run(env, learner, lambda: None, lambda *args: None, mode='EVALUATE',
                     collection_steps=4, max_decision_steps=9)
        assert result == dict(state='COMPLETED', steps=9, updates=0)
        assert_learning_state_equal(continued, learner.learning_state_dict())
        assert_learning_state_equal(frozen_sampling, sampling_state(learner))
        assert all(child.storage.step == 0 and child.storage.pending is None for child in learner.children.values())
