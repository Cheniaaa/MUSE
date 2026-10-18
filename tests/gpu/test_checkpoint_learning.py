"""真实 IPPO 的跨 B 恢复、新 episode 与冻结评估。"""
import copy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_flight_learning import training_case
from muse_runtime.checkpoint import snapshot_learning_state


def assert_learning_state_equal(expected, actual):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert expected.dtype == actual.dtype and expected.shape == actual.shape
        assert torch.equal(expected.cpu(), actual.cpu())
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_learning_state_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert type(expected) is type(actual) and len(expected) == len(actual)
        for left, right in zip(expected, actual):
            assert_learning_state_equal(left, right)
    else:
        assert type(expected) is type(actual) and expected == actual


@contextmanager
def initialized(plan):
    from muse_builtins.algorithms.ippo import IPPO
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    exp = plan.experiment
    rng = WorldRng(exp.root_seed, exp.topology.world_count, plan.device)
    env = MuseEnv(exp, RuntimeContext(plan.device, plan.resource_paths, {}), rng)
    try:
        algorithm = IPPO(exp.algorithm_config)
        algorithm.initialize(exp, SimpleNamespace(device=plan.device), rng)
        yield env, algorithm
    finally:
        env.close()


def test_checkpoint_batch_restore_and_frozen_evaluate(training_case, gpu_stream):
    from muse_runtime.checkpoint import load_checkpoint, publish_checkpoint
    from muse_runtime.driver import run
    from muse_sdk.run_protocol import CheckpointReference
    from muse_server.resolver import resolve_experiment
    catalog, identity, request = training_case
    source = catalog.create_run_plan(identity, mode='TRAIN', device='cuda:0', max_decision_steps=8)
    with initialized(source) as (env, algorithm):
        assert env.num_envs == 2
        result = run(env, algorithm, lambda: None, lambda *args: None,
                     mode='TRAIN', collection_steps=4, max_decision_steps=8)
        assert result == dict(state='COMPLETED', steps=8, updates=2)
        saved = snapshot_learning_state(algorithm.learning_state_dict())
        assert saved['updates'] == 2
        artifact = publish_checkpoint(source, algorithm, steps=8, updates=2)
        assert env.common_step_counter == 8
    reference = CheckpointReference(artifact_id=artifact.id,
        path=str(Path(source.artifact_directory) / artifact.path), size=artifact.size, sha256=artifact.sha256)
    target_request = copy.deepcopy(request)
    target_request['topology']['world_count'] = 1
    target_request['root_seed'] = source.experiment.root_seed + 1
    report = resolve_experiment(target_request, catalog.snapshot())
    assert report.valid, report.errors
    target_id = catalog.save_experiment(report.resolved_experiment, source=target_request)
    target = catalog.create_run_plan(target_id, mode='TRAIN', device='cuda:0', max_decision_steps=8)
    target.checkpoint = reference
    with initialized(target) as (env, algorithm):
        assert env.num_envs == 1 and env.common_step_counter == 0
        assert torch.count_nonzero(env.episode_length_buf) == 0
        positions = env.sim.data.qpos.clone()
        load_checkpoint(target, algorithm)
        assert_learning_state_equal(saved, algorithm.learning_state_dict())
        assert torch.equal(env.sim.data.qpos, positions)
        assert env.common_step_counter == 0 and torch.count_nonzero(env.episode_length_buf) == 0
        result = run(env, algorithm, lambda: None, lambda *args: None,
                     mode='TRAIN', collection_steps=4, max_decision_steps=8)
        assert result == dict(state='COMPLETED', steps=8, updates=2)
        continued = snapshot_learning_state(algorithm.learning_state_dict())
        assert continued['updates'] == 4
        for role, state in saved['roles'].items():
            assert any(not torch.equal(value, continued['roles'][role]['actor'][name])
                       for name, value in state['actor'].items())

    # 新环境原始 timeout=4；5 次真实 step 必须跨过一次 reset。
    with initialized(target) as (env, algorithm):
        load_checkpoint(target, algorithm)
        before = snapshot_learning_state(algorithm.learning_state_dict())
        events = []
        result = run(env, algorithm, lambda: None, lambda name, data: events.append((name, data)),
                     mode='EVALUATE', collection_steps=4, max_decision_steps=5)
        assert result == dict(state='COMPLETED', steps=5, updates=0)
        steps = [data for name, data in events if name == 'step']
        assert len(steps) == 5 and env.common_step_counter == 5
        assert steps[3]['diagnostics']['termination'] == {'terminated': 0, 'truncated': 1}
        assert env.episode_length_buf.tolist() == [1]
        assert_learning_state_equal(before, algorithm.learning_state_dict())
        assert all(child.storage.step == 0 and child.storage.pending is None
                   for child in algorithm.children.values())

    for change in ('hidden_dims', 'actor_order'):
        incompatible = target.model_copy(deep=True)
        if change == 'hidden_dims':
            incompatible.experiment.algorithm_config['hidden_dims'] = [32, 32]
        else:
            next(iter(incompatible.experiment.algorithm_inputs.actors.values())).reverse()
        with initialized(incompatible) as (_, algorithm):
            with pytest.raises(ValueError, match='learning contract mismatch'):
                load_checkpoint(incompatible, algorithm)
