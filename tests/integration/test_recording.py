"""Runtime录制边界使用真实CPU writer；不以preview缓存伪造历史。"""
from pathlib import Path
from types import SimpleNamespace as NS
import zipfile

import numpy as np
import pytest
import torch

from muse_sdk.runtime_types import MotionAction, RoleMask, TransitionView


@pytest.fixture
def capture(recording_source):
    from muse_runtime.recording import RuntimeRecorder
    plan, model = recording_source
    masks = {role: RoleMask(*(torch.ones(2, b.role_slot_count, dtype=torch.bool) for _ in range(3)))
             for role, b in plan.experiment.bindings.items()}
    velocity = {role: torch.zeros(2, b.role_slot_count, 3) for role, b in plan.experiment.bindings.items()}
    env = NS(device='cpu', num_envs=2, common_step_counter=0, experiment=plan.experiment,
        episode_length_buf=torch.zeros(2, dtype=torch.long), cfg=NS(decimation=20),
        sim=NS(data=NS(xpos=torch.zeros(2, model.nbody, 3),
                      xmat=torch.eye(3).reshape(1, 1, 9).repeat(2, model.nbody, 1))),
        physics_outputs=NS(velocity_world=velocity, angular_velocity_body=velocity,
            crashed={r: torch.zeros_like(m.alive) for r, m in masks.items()}), role_masks=masks)
    events = []
    recorder = RuntimeRecorder(plan, env, model, lambda k, p: events.append((k, p)))
    env.recorder = recorder
    recorder.start()
    yield plan, env, recorder, events
    recorder.close()


def advance(env, recorder, *, end=False):
    recorder.before_step()
    env.common_step_counter += 1
    env.episode_length_buf += 1
    env.sim.data.xpos += 1
    actions = {r: MotionAction(torch.full(spec.shape, .25)) for r, spec in env.experiment.action_specs.items()}
    masks = torch.tensor([end, False])
    transition = TransitionView(env.role_masks, None,
        {r: torch.full_like(mask.alive, env.common_step_counter + .5, dtype=torch.float32)
         for r, mask in env.role_masks.items()}, None, torch.zeros(2, dtype=torch.bool), masks,
        masks.long(), {r: torch.zeros_like(mask.alive) for r, mask in env.role_masks.items()},
        masks, {'roles': {}, 'world': {}}, None)
    recorder.after_step(actions, transition)
    if end:
        reset(env, recorder, masks, automatic=True)


def reset(env, recorder, mask, *, automatic=False):
    recorder.before_reset(mask, automatic=automatic)
    env.episode_length_buf[mask] = 0
    env.sim.data.xpos[mask] = 0
    recorder.after_reset(mask, automatic=automatic)


def read_capture(plan, events):
    from muse_sdk.trajectory import TrajectoryReader
    artifact, = [value for kind, value in events if kind == 'trajectory']
    target = Path(plan.artifact_directory) / 'read'
    with zipfile.ZipFile(Path(plan.artifact_directory) / artifact['path']) as archive:
        archive.extractall(target)
    return TrajectoryReader(target)


def test_stride_miss_still_preserves_terminal_and_reset(capture):
    plan, env, recorder, events = capture
    advance(env, recorder)
    advance(env, recorder, end=True)
    recorder.finish('completed')
    reader = read_capture(plan, events)
    try:
        assert [reader.frame(0, n)[2]['phase'] for n in range(3)] == ['initial', 'step', 'reset_after']
        assert reader.frame(0, 1)[2]['roles']['scout']['reward'] == [2.5]
        assert reader.frame(0, 1)[2]['roles']['scout']['action']['motion'] == [[.25] * 4]
        np.testing.assert_array_equal(reader.frame(0, 1)[0], np.full((4, 3), 2.))
        assert reader.frame(0, 2)[2]['episode'] == 1
        assert reader.frame(0, 2)[2]['decision'] == 0
        assert reader.manifest.summary['worlds']['1']['sample_count'] == 1
    finally:
        reader.close()


def test_command_reset_pairs_share_run_decision(capture):
    plan, env, recorder, events = capture
    reset(env, recorder, torch.tensor([True, False]))
    reset(env, recorder, torch.tensor([True, False]))
    recorder.finish('cancelled')
    recorder.finish('completed')
    reader = read_capture(plan, events)
    try:
        assert [reader.frame(0, n)[2]['phase'] for n in range(5)] == [
            'initial', 'reset_before', 'reset_after', 'reset_before', 'reset_after']
        assert reader.frame(0, 4)[2]['run_decision'] == 0
        assert reader.frame(0, 4)[2]['episode'] == 2
        assert reader.frame(0, 4)[2]['roles']['scout']['reward'] is None
        assert reader.manifest.summary['reason'] == 'cancelled'
    finally:
        reader.close()


def test_capacity_reserves_reset_pair_for_all_selected_worlds(capture):
    plan, env, recorder, events = capture
    # 真实writer容量边界，仅缩小可用行数，不替换can_append。
    recorder.writer.rows_per_world = 3
    advance(env, recorder, end=True)
    advance(env, recorder)
    advance(env, recorder)
    assert env.common_step_counter == 3
    assert env.recorder is None
    reader = read_capture(plan, events)
    try:
        assert reader.manifest.summary['reason'] == 'capacity'
        assert reader.manifest.summary['sealed_run_decision'] == 1
        assert reader.manifest.summary['worlds']['0']['sample_count'] == 3
        assert reader.manifest.summary['worlds']['1']['sample_count'] == 1
    finally:
        reader.close()


def test_writer_io_error_propagates(capture, monkeypatch):
    _, _, recorder, _ = capture
    def fail(*args):
        raise OSError('recording disk full')
    monkeypatch.setattr(recorder.writer, 'seal', fail)
    with pytest.raises(OSError, match='disk full'):
        recorder.finish('completed')
