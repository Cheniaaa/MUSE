"""直接Env定向采集验证：每次模块远少于128decision，无learner更新。"""
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from test_flight_env import flight_plan, flight_env, actions
from test_cost_env import cost_plans, cost_env_factory, zero_actions


def start(plan, env, directory, *, worlds=(0, 1), stride=3, budget=1024 * 1024):
    from muse_runtime.recording import RuntimeRecorder
    from muse_runtime.scene_snapshot import export_model, scene_objects
    from muse_sdk.recording_protocol import RecordingConfig
    directory.mkdir(exist_ok=True)
    env.recorder = None
    env.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    env.common_step_counter = 0
    plan = plan.model_copy(update={'artifact_directory': str(directory), 'experiment': env.experiment,
        'recording': RecordingConfig(world_ids=list(worlds), every_n_decisions=stride, max_data_bytes=budget)})
    model = export_model(env.sim.mj_model, directory, objects=scene_objects(env.experiment, env.scene, env.sim.mj_model))
    events = []
    recorder = RuntimeRecorder(plan, env, model, lambda k, p: events.append((k, p)))
    env.recorder = recorder
    recorder.start()
    return recorder, events, model


def finish(recorder, events):
    from muse_sdk.trajectory import TrajectoryReader
    recorder.finish('completed')
    plan = recorder.plan
    recorder.close()
    artifact, = [value for kind, value in events if kind == 'trajectory']
    target = Path(plan.artifact_directory) / 'reader'
    with zipfile.ZipFile(Path(plan.artifact_directory) / artifact['path']) as archive:
        archive.extractall(target)
    return TrajectoryReader(target)


def test_real_pre_reset_capture_and_selected_world_copy(flight_plan, flight_env, tmp_path, monkeypatch, record_property):
    env = flight_env
    # 真实manager仍计算终止原因，仅用独立world阈值让两world不同步结束。
    timeout = next(cfg for cfg in env.termination_manager.cfg.values() if cfg.time_out)
    monkeypatch.setattr(timeout, 'func', lambda env, **_: env.episode_length_buf >= torch.tensor([2, 5], device=env.device))
    recorder, events, _ = start(flight_plan, env, tmp_path, worlds=(0,))
    expected = []
    original = recorder.after_step
    def before_reset(actions, transition):
        expected.append(env.sim.data.xpos[0].clone())
        original(actions, transition)
    monkeypatch.setattr(recorder, 'after_step', before_reset)
    state = env.rng.generator(0, 'recording-test', 'unchanged').get_state().clone()
    action = actions(env, .1)
    env.step(action)
    out = env.step(action)
    assert out.transition.world_reset.tolist() == [True, False]
    env.reset(torch.tensor([True, False], device=env.device))
    reader = finish(recorder, events)
    try:
        assert list(reader.arrays) == [0]
        xpos, _, diagnostics = reader.frame(0, 1)
        np.testing.assert_array_equal(xpos, expected[-1].cpu().numpy())
        np.testing.assert_allclose(diagnostics['roles']['scout']['reward'], out.transition.rewards['scout'][0].cpu().numpy())
        np.testing.assert_allclose(diagnostics['roles']['guard']['action']['motion'], action['guard'].motion[0].cpu().numpy())
        assert reader.frame(0, 2)[2]['episode'] == 1
        assert reader.frame(0, 2)[2]['decision'] == 0
        assert reader.frame(0, 3)[2]['phase'] == 'reset_before'
        assert reader.frame(0, 4)[2]['phase'] == 'reset_after'
        assert reader.frame(0, 4)[2]['episode'] == 2
        assert reader.frame(0, 4)[2]['run_decision'] == 2
        torch.testing.assert_close(env.rng.generator(0, 'recording-test', 'unchanged').get_state(), state)
        record_property('recording_statistics', recorder.statistics)
    finally:
        reader.close()


def test_viewer_modes_do_not_change_recorded_timeline(flight_plan, flight_env, tmp_path):
    from muse_runtime.preview import WorkerPreview
    from muse_sdk.run_protocol import PreviewSelections, PreviewSelection, DisplaySettings, WorkerControl, preview_socketpair
    env, timelines = flight_env, []
    for mode in ((), ('viser',), ('native',), ('viser', 'native')):
        directory = tmp_path / ('_'.join(mode) or 'off')
        recorder, events, model = start(flight_plan, env, directory)
        sender, receiver = preview_socketpair()
        preview = WorkerPreview(flight_plan.run_id, env, sender, lambda *_: None, model=model)
        try:
            for decision in range(6):
                selections = PreviewSelections(**{name: PreviewSelection(selection_seq=decision + 1,
                    display=DisplaySettings(enabled=name in mode, world_id=decision % 2, rate_hz=30.))
                    for name in ('viser', 'native')})
                preview.apply(WorkerControl(run_id=flight_plan.run_id, kind='preview', preview=selections))
                env.step(actions(env, 0.))
                preview.after_step()
                if decision == 2:
                    receiver.close()
                preview.sample(force=True)
            reader = finish(recorder, events)
            try:
                timelines.append({w: a['index'].tolist() for w, a in reader.arrays.items()})
            finally:
                reader.close()
        finally:
            sender.close()
            receiver.close()
            recorder.close()
    assert timelines == [timelines[0]] * 4
    assert [row[3] for row in timelines[0][0]] == [0, 3, 6]


def test_real_named_costs_are_per_decision_and_per_slot(cost_plans, cost_env_factory, tmp_path):
    env = cost_env_factory()
    recorder, events, _ = start(cost_plans[True], env, tmp_path, stride=1)
    out = env.step(zero_actions(env))
    reader = finish(recorder, events)
    try:
        diagnostics = reader.frame(0, 1)[2]
        assert diagnostics['roles']['rover']['costs']['effort'] == [1., 2.]
        assert diagnostics['roles']['scout']['costs']['energy'] == [1.]
        assert diagnostics['world']['costs']['risk'] == 10.
        np.testing.assert_allclose(diagnostics['world']['world_reward'], out.transition.world_reward[0].item())
    finally:
        reader.close()


def test_real_hybrid_action_keeps_int32_channels(tmp_path, gpu_stream):
    from indoor_radio_helpers import indoor_radio_catalog
    from test_indoor_radio_env import indoor_env
    from muse_sdk.experiment import RunPlan
    catalog, request = indoor_radio_catalog(tmp_path / 'radio')
    with indoor_env(catalog, request) as (env, rule):
        plan = RunPlan(run_id='radio', experiment_id='radio-exp', mode='EVALUATE', device='cuda:0',
            platform_revision=env.experiment.platform_revision, artifact_directory=str(tmp_path),
            resource_paths={}, experiment=env.experiment, max_decision_steps=1)
        recorder, events, _ = start(plan, env, tmp_path / 'capture', stride=1)
        action = rule.act(env.policy_input.observations, env.role_masks, env.rng, True)
        env.step(action.role_actions)
        reader = finish(recorder, events)
        try:
            diagnostics = reader.frame(1, 1)[2]
            for role, value in action.role_actions.items():
                assert diagnostics['roles'][role]['action']['radio']['channel'] == value.radio.channel[1].cpu().tolist()
                np.testing.assert_allclose(diagnostics['roles'][role]['action']['radio']['power_w'], value.radio.power_w[1].cpu().numpy())
        finally:
            reader.close()
