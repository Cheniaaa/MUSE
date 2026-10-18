"""原生IMU物理值、非decision整除采样、group投影和局部重置。"""
import copy
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import pytest
import torch
import yaml

from flight_helpers import archive


ROOT = Path(__file__).resolve().parents[2]
SIGNALS = ('gyro', 'specific_force')


def make_imu_plan(root, period=3, enabled=True, task=False):
    from muse_builtins.metadata import export_metadata
    from muse_server.catalog import Catalog
    from muse_server.resolver import resolve_experiment
    example = ROOT / 'examples/basic_flight'
    root.mkdir(exist_ok=True)
    metadata = root / 'builtin.json'
    metadata.write_text(export_metadata(example / 'controller.json').model_dump_json())
    catalog = Catalog(root / 'catalog', metadata)
    package = root / 'package'
    shutil.copytree(example / 'training_package', package)
    composition_path = package / 'training_package.yaml'
    package_config = yaml.safe_load(composition_path.read_text())
    composition = package_config['compositions']['default']
    if task:
        shutil.copyfile(ROOT / 'tests/fixtures/imu_task.py', package / 'imu_task.py')
        entry = package / '__init__.py'
        entry.write_text(entry.read_text() + '\nfrom .imu_task import ImuTask\n'
            '_base_definition = build_training_package\n'
            'def build_training_package():\n'
            '    definition = _base_definition()\n'
            '    definition.tasks["imu_task"] = ImuTask\n'
            '    return definition\n')
        composition['task'] = 'imu_task'
    for role in composition['agent_roles'].values():
        role['observation_terms'].extend(dict(selection_id='imu_' + name, term='builtin:imu_' + name,
            default_enabled=enabled) for name in ('gyro', 'specific_force', 'valid', 'age'))
    composition_path.write_text(yaml.safe_dump(package_config, sort_keys=False))
    payload = json.loads((example / 'request.json').read_text())
    for kind, source in [('environment', example / 'environment'), ('training_package', package)]:
        resource = catalog.register_archive(kind, archive(source, root / (kind + '.zip')))
        payload[kind + '_id'], payload[kind + '_revision'] = resource.resource_id, resource.revision
    first, second = payload['scenario']['entity_groups']
    first.update(count=3, sensor_profile_id='cf2_imu', sensor_config={'update_rate_hz': 1 / (.002 * period)})
    payload['scenario']['placements'][first['placement_id']].extend(
        payload['scenario']['placements'][second['placement_id']])
    payload['scenario']['entity_groups'] = [first]
    payload['role_config']['scout']['slots'] = [1]
    payload['role_config']['guard'] = dict(group_id=first['group_id'], slots=[2, 0])
    # 平移几何避免与地面接触；角色顺序刻意不同于物理slot。
    payload['scenario']['scene']['density'] = 0.
    payload['scenario']['scene']['viscosity'] = 0.
    report = resolve_experiment(payload, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment, source=payload)
    return catalog.create_run_plan(identity, mode='EVALUATE', device='cuda:0')


@pytest.fixture(scope='module')
def imu_plan(tmp_path_factory):
    return make_imu_plan(tmp_path_factory.mktemp('imu-plan'))


@pytest.fixture(scope='module')
def imu_env(imu_plan, gpu_stream):
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    env = MuseEnv(imu_plan.experiment, RuntimeContext('cuda:0', imu_plan.resource_paths, {}),
                  WorldRng(42, 2, 'cuda:0'))
    yield env
    env.close()


def commands(env):
    from muse_sdk.runtime_types import MotionAction
    return {role: MotionAction(torch.zeros(spec.shape, device=env.device))
            for role, spec in env.experiment.action_specs.items()}


def reset_all(env):
    env.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))


def test_plain_flight_has_no_measurement_cache(tmp_path, gpu_stream):
    from test_flight_env import build_plan
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    plan = build_plan(tmp_path)
    env = MuseEnv(plan.experiment, RuntimeContext('cuda:0', plan.resource_paths, {}), WorldRng(42, 2, 'cuda:0'))
    try:
        assert env.sensor_outputs == {}
        env.step(commands(env))
        assert env.sensor_outputs == {}
    finally:
        env.close()


def test_initial_measurement_is_physics_not_assumed_hover(imu_env):
    env = imu_env
    reset_all(env)
    assert set(env.sensor_outputs) == {'scout_group'}
    for signal in SIGNALS:
        sample = env.sensor_outputs['scout_group'][signal]
        assert sample.data.shape == (2, 3, 3)
        assert sample.valid.all()
        assert sample.sample_tick.count_nonzero() == sample.age.count_nonzero() == 0
        torch.testing.assert_close(sample.data, torch.zeros_like(sample.data), atol=1e-5, rtol=0)


def test_native_imu_force_and_rotated_axes_match_cpu_oracle(imu_env):
    from mjlab.sensor.builtin_sensor import BuiltinSensor
    env = imu_env
    reset_all(env)
    motion = env.action_manager.get_term('scout_group')
    model = env.sim.mj_model
    # 90度yaw，世界+x的支持力应出现在site的-y轴；竖直支持力抵消重力。
    quat = torch.tensor([2**-.5, 0., 0., 2**-.5], device=env.device)
    env.sim.data.qpos[:, motion.qpos_ids[:, 3:7]] = quat
    for slot, body in enumerate(motion.body_ids):
        mass = model.body_mass[body]
        env.sim.data.xfrc_applied[:, body, :3] = torch.tensor([mass*2, 0., mass*9.81], device=env.device)
        env.sim.data.qvel[:, motion.qvel_ids[slot, 5]] = .4 + slot * .2
    env.sim.forward()
    # 在真实状态上发布测量；不通过Term重算加速度。
    env.sensor_sampler.capture(torch.zeros(2, dtype=torch.long, device=env.device),
        {'scout_group': {signal: torch.ones(2, dtype=torch.bool, device=env.device) for signal in SIGNALS}})
    cpu = mujoco.MjData(model)
    for field in ('qpos', 'qvel', 'ctrl', 'xfrc_applied'):
        getattr(cpu, field)[:] = getattr(env.sim.data, field)[0].cpu().numpy()
    mujoco.mj_forward(model, cpu)
    for slot in range(3):
        for signal, native in [('gyro', 'body_gyro'), ('specific_force', 'body_linacc')]:
            name = f'scout_group_{slot}/{native}'
            assert isinstance(env.scene[name], BuiltinSensor)
            actual = env.sensor_outputs['scout_group'][signal].data[:, slot]
            expected = [0., 0., .4 + slot * .2] if signal == 'gyro' else [0., -2., 9.81]
            np.testing.assert_allclose(cpu.sensor(name).data, expected, atol=2e-5)
            np.testing.assert_allclose(actual.cpu(), np.tile(expected, (2, 1)), atol=2e-5)
    obs = env._observe().actors
    np.testing.assert_allclose(obs['scout']['imu_gyro'][..., 2].cpu(), [[.6], [.6]], atol=1e-6)
    np.testing.assert_allclose(obs['guard']['imu_gyro'][..., 2].cpu(), [[.8, .4], [.8, .4]], atol=1e-6)
    reset_all(env)


def test_substep_sample_is_held_and_reading_does_not_resample(imu_env, monkeypatch):
    env = imu_env
    reset_all(env)
    forwarded, captured, originals = [], [], {}
    original_forward = env.sim.forward
    original_capture = env.sensor_sampler.capture
    def forward():
        original_forward()
        forwarded.append(round(float(env.sim.data.time[0]) / .002))
    def capture(ticks, masks):
        original_capture(ticks, masks)
        captured.append(ticks.clone())
        originals[int(ticks[0])] = env.sensor_outputs['scout_group']['gyro'].data.clone()
    monkeypatch.setattr(env.sim, 'forward', forward)
    monkeypatch.setattr(env.sensor_sampler, 'capture', capture)
    action = commands(env)
    for value in action.values():
        value.motion[..., 3] = .08
    for tick, sampled, age in [(20, 18, .004), (40, 39, .002)]:
        output = env.step(action)
        assert not output.transition.world_reset.any()
        sample = env.sensor_outputs['scout_group']['gyro']
        assert (sample.sample_tick == sampled).all()
        torch.testing.assert_close(sample.age, torch.full_like(sample.age, age))
        torch.testing.assert_close(sample.data, originals[sampled], rtol=0, atol=0)
        native = torch.stack([env.scene[f'scout_group_{s}/body_gyro'].data for s in range(3)], dim=1)
        assert not torch.equal(sample.data, native)
        before = copy.deepcopy(captured)
        for _ in range(3):
            env._observe()
        assert len(captured) == len(before)
    assert forwarded == [3, 6, 9, 12, 15, 18, 20, 21, 24, 27, 30, 33, 36, 39, 40]
    assert [int(value[0]) for value in captured] == list(range(3, 40, 3))


def test_subset_reset_preserves_other_world_measurement_and_phase(imu_env):
    env = imu_env
    reset_all(env)
    action = commands(env)
    env.step(action)
    env.step(action)
    control = {name: tuple(value[1].clone() for value in sample)
               for name, sample in env.sensor_outputs['scout_group'].items()}
    reset_all(env)
    env.step(action)
    before = {name: tuple(value[1].clone() for value in sample)
              for name, sample in env.sensor_outputs['scout_group'].items()}
    env.reset(torch.tensor([True, False], device=env.device))
    for name, sample in env.sensor_outputs['scout_group'].items():
        for actual, expected in zip(sample, before[name]):
            torch.testing.assert_close(actual[1], expected, rtol=0, atol=0)
    env.step(action)
    for sample in env.sensor_outputs['scout_group'].values():
        assert sample.sample_tick.tolist() == [[18]*3, [39]*3]
        torch.testing.assert_close(sample.age, torch.tensor([[.004]*3, [.002]*3], device=env.device))
    for name, sample in env.sensor_outputs['scout_group'].items():
        for actual, expected in zip(sample, control[name]):
            torch.testing.assert_close(actual[1], expected)


def test_timeout_keeps_terminal_imu_snapshot(imu_env):
    env = imu_env
    reset_all(env)
    timeout = next(cfg for cfg in env.termination_manager.cfg.values() if cfg.time_out)
    old = timeout.params['episode_max_steps']
    timeout.params['episode_max_steps'] = 1
    try:
        output = env.step(commands(env))
        assert output.transition.world_truncated.all()
        terminal = output.transition.pre_reset_next_observations.actors['guard']['imu_age']
        torch.testing.assert_close(terminal, torch.full_like(terminal, .004))
        assert output.policy_observations.actors['guard']['imu_age'].count_nonzero() == 0
        saved = terminal.clone()
        env.step(commands(env))
        torch.testing.assert_close(terminal, saved, rtol=0, atol=0)
    finally:
        timeout.params['episode_max_steps'] = old


@pytest.mark.parametrize('enabled', [False, True])
def test_slow_sensor_or_unconsumed_profile(tmp_path, gpu_stream, enabled):
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    plan = make_imu_plan(tmp_path, period=30, enabled=enabled)
    env = MuseEnv(plan.experiment, RuntimeContext('cuda:0', plan.resource_paths, {}), WorldRng(42, 2, 'cuda:0'))
    try:
        env.step(commands(env))
        if enabled:
            sample = env.sensor_outputs['scout_group']['specific_force']
            assert sample.sample_tick.count_nonzero() == 0
            torch.testing.assert_close(sample.data, torch.zeros_like(sample.data), rtol=0, atol=1e-5)
            torch.testing.assert_close(sample.age, torch.full_like(sample.age, .04))
        else:
            assert env.sensor_outputs == {}
    finally:
        env.close()


def test_task_only_consumer_reads_published_measurement(tmp_path, gpu_stream):
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.package_loader import load_package
    from muse_runtime.rng import WorldRng
    plan = make_imu_plan(tmp_path, enabled=False, task=True)
    with load_package(Path(plan.resource_paths['training_package'])) as loaded:
        env = MuseEnv(plan.experiment, RuntimeContext('cuda:0', plan.resource_paths, {},
            loaded.definition.tasks['imu_task']), WorldRng(42, 2, 'cuda:0'))
        try:
            assert not env.task.initial_valid.any()
            action = commands(env)
            action['scout'].motion[..., 3] = .08
            env.step(action)
            assert env.task.reading.abs().sum() > 0
            torch.testing.assert_close(env.task.reading, env.sensor_outputs['scout_group']['gyro'].data)
            assert 'imu_gyro' not in env.observations.actors['scout']
        finally:
            env.close()


def test_record_imu_sampling_cost(tmp_path, gpu_stream, record_property):
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    # 同负载、充分初始化后记录开销；不以时间阈值作为正确性判断。
    for enabled in (False, True):
        plan = make_imu_plan(tmp_path / str(enabled), period=5, enabled=enabled)
        env = MuseEnv(plan.experiment, RuntimeContext('cuda:0', plan.resource_paths, {}), WorldRng(42, 2, 'cuda:0'))
        try:
            action = commands(env)
            for _ in range(3):
                env.step(action)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(10):
                output = env.step(action)
                assert not output.transition.world_reset.any()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) / 10
            record_property('imu_active' if enabled else 'imu_disabled', elapsed)
            print(f'IMU_BENCH enabled={enabled} worlds=2 slots=3 rate_hz=100 decision_s={elapsed:.6f}')
        finally:
            env.close()
