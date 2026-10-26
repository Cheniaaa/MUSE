"""真实GPU交互：subset reset、recurrent隔离及冻结cost学习状态。"""
from collections import deque
import json
from pathlib import Path

import pytest
import torch

from mappo_helpers import EXAMPLE, cooperative_pipeline
from test_checkpoint_learning import assert_learning_state_equal
from test_cost_training import initialized, sampling_state
from preview_helpers import preview_control, preview_model


@pytest.mark.parametrize('recurrent', ['gru', 'lstm'])
def test_interactive_subset_reset_freezes_learning_and_refreshes_preview(pipeline, tmp_path, gpu_stream, recurrent):
    from muse_runtime.checkpoint import load_checkpoint, publish_checkpoint, snapshot_learning_state
    from muse_runtime.driver import run
    from muse_runtime.preview import WorkerPreview
    from muse_sdk.run_protocol import (CheckpointReference, InteractiveCommand, decode_preview, preview_socketpair)
    from muse_server.resolver import resolve_experiment

    catalog, baseline = cooperative_pipeline(pipeline, tmp_path)
    request = json.loads((EXAMPLE / 'request_cost.json').read_text())
    for key in ('environment_id', 'environment_revision', 'training_package_id', 'training_package_revision'):
        request[key] = baseline[key]
    request['algorithm_config'].update(share_actor_parameters=False,
                                      recurrent={'type': recurrent, 'hidden_size': 16})
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment, source=request)
    source = catalog.create_run_plan(identity, mode='TRAIN', device='cuda:0', max_decision_steps=4)
    with initialized(source) as (env, learner):
        run(env, learner, lambda: None, lambda *_: None, mode='TRAIN',
            collection_steps=4, max_decision_steps=4)
        artifact = publish_checkpoint(source, learner, steps=4, updates=1)
        saved = snapshot_learning_state(learner.learning_state_dict())
    # Runtime测试直接构造不可变引用；正式artifact登记与Catalog准入在HTTP验收。
    target = source.model_copy(update={'mode': 'INTERACTIVE', 'checkpoint': CheckpointReference(
        artifact_id=artifact.id, path=str(Path(source.artifact_directory) / artifact.path),
        size=artifact.size, sha256=artifact.sha256)})
    sender, receiver = preview_socketpair()
    with initialized(target) as (env, learner), sender, receiver:
        load_checkpoint(target, learner)
        assert_learning_state_equal(saved, learner.learning_state_dict())
        frozen_cost_sampling = sampling_state(learner)
        preview = WorkerPreview(target.run_id, env, sender, lambda *_: None,
                                model=preview_model(env, target.artifact_directory))
        assert preview.error is None
        preview.apply(preview_control(target.run_id, 1, rate=30))
        commands = deque([None, None, InteractiveCommand(command_seq=1, command='pause'),
            InteractiveCommand(command_seq=2, command='step'),
            InteractiveCommand(command_seq=3, command='reset_world', world_id=0),
            InteractiveCommand(command_seq=3, command='reset_world', world_id=0),
            InteractiveCommand(command_seq=4, command='step')])
        results, frames, before_reset = [], [], {}

        def hidden_tensors(child):
            hidden = child.actor.get_hidden_state()
            return hidden if isinstance(hidden, tuple) else (hidden,)

        def event_sink(kind, payload):
            if kind == 'step':
                preview.after_step()
            elif kind == 'reset_world':
                preview.after_reset(payload['world_id'])
                assert env.episode_length_buf[0] == 0
                torch.testing.assert_close(env.sim.data.qpos[1], before_reset['qpos'], rtol=0, atol=0)
                assert env.episode_length_buf[1] == before_reset['clock']
                for owner, child in learner.children.items():
                    for current, old in zip(hidden_tensors(child), before_reset[owner]):
                        assert not current[:, 0].any()
                        torch.testing.assert_close(current[:, 1], old[:, 1], rtol=0, atol=0)
                assert_learning_state_equal(saved, learner.learning_state_dict())
            elif kind == 'command_result':
                results.append(payload)
                if payload['command_seq'] == 2:
                    before_reset['qpos'] = env.sim.data.qpos[1].clone()
                    before_reset['clock'] = env.episode_length_buf[1].clone()
                    for owner, child in learner.children.items():
                        before_reset[owner] = tuple(value.clone() for value in hidden_tensors(child))
                    assert any(value[:, 0].any() for owner in learner.children for value in before_reset[owner])
                return
            preview.sample(force=True)
            frames.append(decode_preview(receiver.recv(128 * 1024))[0])

        result = run(env, learner, lambda: None if commands else 'cancel', event_sink,
            mode='INTERACTIVE', collection_steps=4, max_decision_steps=20,
            command_reader=lambda: commands.popleft())
        assert result == {'state': 'CANCELLED', 'steps': 4, 'updates': 0}
        assert all(item['status'] == 'applied' for item in results)
        assert results[2] == results[3]
        assert [item['steps'] for item in results] == [2, 3, 3, 3, 4]
        assert [(frame.episode, frame.decision) for frame in frames] == [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
        assert_learning_state_equal(saved, learner.learning_state_dict())
        assert_learning_state_equal(frozen_cost_sampling, sampling_state(learner))
        assert all(child.storage.step == 0 and child.storage.pending is None for child in learner.children.values())
