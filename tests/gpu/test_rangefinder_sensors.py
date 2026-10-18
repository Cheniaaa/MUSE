"""真实原生测距的几何、独立采样和生命周期验收。"""
import json
import math
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import pytest
import torch
import yaml

from flight_helpers import archive
from test_imu_sensors import commands, reset_all

ROOT = Path(__file__).resolve().parents[2]
SIGNALS = ('gyro', 'specific_force', 'range')


def make_range_plan(root, *, enabled=True, task=False, period=73, imu_only=False):
    from muse_builtins.metadata import export_metadata
    from muse_server.catalog import Catalog
    from muse_server.resolver import resolve_experiment
    root.mkdir(exist_ok=True)
    example = ROOT / 'examples/basic_flight'
    metadata = root / 'builtin.json'
    metadata.write_text(export_metadata(example / 'controller.json').model_dump_json())
    catalog = Catalog(root / 'catalog', metadata)
    package = root / 'package'
    shutil.copytree(example / 'training_package', package)
    composition_path = package / 'training_package.yaml'
    package_config = yaml.safe_load(composition_path.read_text())
    composition = package_config['compositions']['default']
    for role in composition['agent_roles'].values():
        role['observation_terms'].extend(dict(selection_id=name, term='builtin:' + name,
            default_enabled=enabled) for name in ('imu_gyro', 'imu_specific_force', 'imu_valid', 'imu_age',
                'rangefinder_distance', 'rangefinder_valid', 'rangefinder_age'))
        if imu_only:
            role['observation_terms'] = [term for term in role['observation_terms']
                                         if not term['selection_id'].startswith('rangefinder_')]
    if task:
        shutil.copyfile(ROOT / 'tests/fixtures/rangefinder_task.py', package / 'rangefinder_task.py')
        entry = package / '__init__.py'
        entry.write_text(entry.read_text() + '\nfrom .rangefinder_task import RangefinderTask\n'
            '_base_definition = build_training_package\n'
            'def build_training_package():\n'
            '    definition = _base_definition()\n'
            '    definition.tasks["rangefinder_task"] = RangefinderTask\n'
            '    return definition\n')
        composition['task'] = 'rangefinder_task'
    composition_path.write_text(yaml.safe_dump(package_config, sort_keys=False))
    environment = root / 'environment'
    shutil.copytree(example / 'environment', environment)
    scene_file = environment / yaml.safe_load((environment / 'environment.yaml').read_text())['scene']
    scene_file.write_text('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="20 20 .1"/>
      <geom name="box" type="box" pos="10 0 .5" size="1 1 .5"/>
      <geom name="invisible" type="box" pos="20 0 .5" size="1 1 .5" rgba="1 1 1 0"/>
      <geom name="noncolliding" type="box" pos="30 0 .5" size="1 1 .5" contype="0" conaffinity="0"/>
    </worldbody></mujoco>''')
    payload = json.loads((example / 'request.json').read_text())
    for kind, source in [('environment', environment), ('training_package', package)]:
        resource = catalog.register_archive(kind, archive(source, root / (kind + '.zip')))
        payload[kind + '_id'], payload[kind + '_revision'] = resource.resource_id, resource.revision
    first, second = payload['scenario']['entity_groups']
    first.update(count=3, sensor_profile_id='cf2_imu_rangefinder_down',
        sensor_config={'imu_update_rate_hz': 100., 'rangefinder_update_rate_hz': 1 / (.002 * period)})
    if imu_only:
        first.update(sensor_profile_id='cf2_imu', sensor_config={'update_rate_hz': 100.})
    payload['scenario']['placements'][first['placement_id']].extend(
        payload['scenario']['placements'][second['placement_id']])
    payload['scenario']['entity_groups'] = [first]
    payload['role_config']['scout']['slots'] = [1]
    payload['role_config']['guard'] = dict(group_id=first['group_id'], slots=[2, 0])
    report = resolve_experiment(payload, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment)
    return catalog.create_run_plan(identity, mode='EVALUATE', device='cuda:0')


def create_env(plan, task_factory=None):
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    return MuseEnv(plan.experiment, RuntimeContext('cuda:0', plan.resource_paths, {}, task_factory),
                   WorldRng(42, 2, 'cuda:0'))


@pytest.fixture(scope='module')
def range_env(tmp_path_factory, gpu_stream):
    env = create_env(make_range_plan(tmp_path_factory.mktemp('range-plan')))
    yield env
    env.close()


def publish(env):
    env.sim.forward()
    env.sensor_sampler.capture(torch.zeros(2, dtype=torch.int64, device=env.device),
        {'scout_group': {signal: torch.ones(2, dtype=torch.bool, device=env.device) for signal in SIGNALS}})
    return env.sensor_outputs['scout_group']['range']


def place(env, positions, quaternion=(1., 0., 0., 0.)):
    reset_all(env)
    term = env.action_manager.get_term('scout_group')
    env.sim.data.qpos[:, term.qpos_ids[:, :3]] = torch.tensor(positions, device=env.device)
    env.sim.data.qpos[:, term.qpos_ids[:, 3:7]] = torch.tensor(quaternion, device=env.device)
    return publish(env)


def cpu_distances(env):
    data = mujoco.MjData(env.sim.mj_model)
    distances = []
    for world in range(2):
        for field in ('qpos', 'qvel', 'ctrl', 'xfrc_applied'):
            getattr(data, field)[:] = getattr(env.sim.data, field)[world].cpu().numpy()
        mujoco.mj_forward(env.sim.mj_model, data)
        distances.append([data.sensor(f'scout_group_{s}/rangefinder_down').data[0] for s in range(3)])
    return np.array(distances)


def test_native_mount_plane_rotation_and_role_projection(range_env):
    env = range_env
    sample = place(env, [[[0, 0, 1.01], [2, 0, 2.01], [4, 0, 3.01]],
                         [[0, 0, 1.51], [2, 0, 2.51], [4, 0, 3.51]]])
    expected = [[1., 2., 3.], [1.5, 2.5, 3.5]]
    np.testing.assert_allclose(sample.data[..., 0].cpu(), expected, atol=2e-5)
    np.testing.assert_allclose(cpu_distances(env), expected, atol=2e-5)
    assert sample.valid.all()
    obs = env._observe().actors
    np.testing.assert_allclose(obs['scout']['rangefinder_distance'][..., 0].cpu(), [[2.], [2.5]], atol=2e-5)
    np.testing.assert_allclose(obs['guard']['rangefinder_distance'][..., 0].cpu(), [[3., 1.], [3.5, 1.5]], atol=2e-5)
    # body倾斜60°，site垂直高度=1.005-.01*cos60=1；沿束距离=2。
    sample = place(env, [[0, 0, 1.005], [4, 0, 1.005], [8, 0, 1.005]],
                   (math.cos(math.pi/6), 0., math.sin(math.pi/6), 0.))
    np.testing.assert_allclose(sample.data.cpu(), 2., atol=2e-5)
    np.testing.assert_allclose(cpu_distances(env), 2., atol=2e-5)


@pytest.mark.parametrize(('height', 'distance', 'valid'), [(.049, 4., False), (.05, .04, True),
    (4.01, 4., True), (4.011, 4., False)])
def test_physical_range_limits(range_env, height, distance, valid):
    sample = place(range_env, [[0, 0, height], [4, 0, height], [8, 0, height]])
    np.testing.assert_allclose(sample.data.cpu(), distance, atol=2e-5)
    assert sample.valid.tolist() == [[valid]*3]*2


def test_upward_no_hit_and_geometric_occlusion_filters(range_env):
    env = range_env
    sample = place(env, [[0, 0, 2.01], [4, 0, 2.01], [8, 0, 2.01]], (0., 1., 0., 0.))
    assert not sample.valid.any()
    np.testing.assert_allclose(cpu_distances(env), -1.)
    np.testing.assert_allclose(sample.data.cpu(), 4.)
    sample = place(env, [[10, 0, 2.01], [20, 0, 2.01], [30, 0, 2.01]])
    expected = [[1., 2., 1.]]*2
    np.testing.assert_allclose(sample.data[..., 0].cpu(), expected, atol=2e-5)
    np.testing.assert_allclose(cpu_distances(env), expected, atol=2e-5)


def test_other_cf2_occludes_and_moving_it_restores_floor(range_env):
    env = range_env
    sample = place(env, [[0, 0, 2.01], [0, 0, 1.01], [4, 0, 2.01]])
    hit = float(sample.data[0, 0, 0])
    assert .9 < hit < 1.1
    np.testing.assert_allclose(sample.data[..., 0].cpu(), cpu_distances(env), atol=2e-5)
    motion = env.action_manager.get_term('scout_group')
    env.sim.data.qpos[:, motion.qpos_ids[1, 0]] = 2.
    sample = publish(env)
    np.testing.assert_allclose(sample.data[:, 0, 0].cpu(), 2., atol=2e-5)


def test_independent_sampling_union_and_terminal_snapshot(range_env, monkeypatch):
    env = range_env
    reset_all(env)
    forward_ticks, samples = [], {}
    forward, capture = env.sim.forward, env.sensor_sampler.capture
    def record_forward():
        forward()
        forward_ticks.append(round(float(env.sim.data.time[0]) / .002))
    def record_capture(ticks, masks):
        capture(ticks, masks)
        if masks['scout_group']['range'][0]:
            samples[int(ticks[0])] = env.sensor_outputs['scout_group']['range'].data.clone()
    monkeypatch.setattr(env.sim, 'forward', record_forward)
    monkeypatch.setattr(env.sensor_sampler, 'capture', record_capture)
    action = commands(env)
    for value in action.values():
        value.motion[..., 2] = .08
    for decision in range(1, 9):
        result = env.step(action)
        assert not result.transition.world_reset.any()
        if decision in (4, 8):
            sample = env.sensor_outputs['scout_group']['range']
            tick = 73 if decision == 4 else 146
            assert (sample.sample_tick == tick).all()
            torch.testing.assert_close(sample.data, samples[tick], rtol=0, atol=0)
            torch.testing.assert_close(sample.age, torch.full_like(sample.age, .014 if decision == 4 else .028))
            assert (env.sensor_outputs['scout_group']['gyro'].sample_tick == decision*20).all()
            latest = torch.stack([env.scene[f'scout_group_{s}/rangefinder_down'].data for s in range(3)], dim=1)
            assert not torch.equal(sample.data, latest)
    assert forward_ticks == sorted(set(range(5, 161, 5)) | {73, 146})
    before = len(forward_ticks)
    env._observe()
    env._observe()
    assert len(forward_ticks) == before
    timeout = next(cfg for cfg in env.termination_manager.cfg.values() if cfg.time_out)
    old = timeout.params['episode_max_steps']
    timeout.params['episode_max_steps'] = 9
    try:
        output = env.step(action)
        assert output.transition.world_truncated.all()
        terminal = output.transition.pre_reset_next_observations.actors['guard']['rangefinder_age']
        torch.testing.assert_close(terminal, torch.full_like(terminal, .068))
        assert output.policy_observations.actors['guard']['rangefinder_age'].count_nonzero() == 0
    finally:
        timeout.params['episode_max_steps'] = old


def test_subset_reset_keeps_other_world_future_samples(range_env):
    env = range_env
    action = commands(env)
    reset_all(env)
    for _ in range(8):
        env.step(action)
    control = {name: tuple(f[1].clone() for f in value) for name, value in env.sensor_outputs['scout_group'].items()}
    reset_all(env)
    for _ in range(4):
        env.step(action)
    saved = {name: tuple(f[1].clone() for f in value) for name, value in env.sensor_outputs['scout_group'].items()}
    env.reset(torch.tensor([True, False], device=env.device))
    for name, value in env.sensor_outputs['scout_group'].items():
        for actual, expected in zip(value, saved[name]):
            torch.testing.assert_close(actual[1], expected, rtol=0, atol=0)
    for _ in range(4):
        env.step(action)
    for name, value in env.sensor_outputs['scout_group'].items():
        for actual, expected in zip(value, control[name]):
            torch.testing.assert_close(actual[1], expected)
    assert env.sensor_outputs['scout_group']['range'].sample_tick.tolist() == [[73]*3, [146]*3]


@pytest.mark.parametrize('enabled', [False, True])
def test_profile_without_consumer_does_not_add_native_rays(tmp_path, gpu_stream, enabled):
    env = create_env(make_range_plan(tmp_path, enabled=enabled, period=100))
    try:
        names = [mujoco.mj_id2name(env.sim.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                 for i in range(env.sim.mj_model.nsensor)]
        assert sum(name.endswith('/rangefinder_down') for name in names) == (3 if enabled else 0)
        env.step(commands(env))
        if enabled:
            assert env.sensor_outputs['scout_group']['range'].sample_tick.count_nonzero() == 0
            for _ in range(4):
                env.step(commands(env))
            assert (env.sensor_outputs['scout_group']['range'].sample_tick == 100).all()
        else:
            assert env.sensor_outputs == {}
    finally:
        env.close()


def test_task_only_range_consumer(tmp_path, gpu_stream):
    from muse_runtime.package_loader import load_package
    plan = make_range_plan(tmp_path, enabled=False, task=True)
    with load_package(Path(plan.resource_paths['training_package'])) as loaded:
        env = create_env(plan, loaded.definition.tasks['rangefinder_task'])
        try:
            assert not env.task.initial_valid.any()
            env.step(commands(env))
            torch.testing.assert_close(env.task.reading, env.sensor_outputs['scout_group']['range'].data)
            assert env.task.reading.min() > .04
            assert 'rangefinder_distance' not in env.observations.actors['scout']
        finally:
            env.close()


def test_combined_profile_keeps_imu_physics_oracle(range_env):
    from test_imu_sensors import test_native_imu_force_and_rotated_axes_match_cpu_oracle as check_imu
    check_imu(range_env)


@pytest.mark.parametrize(('offset', 'distance', 'near_valid'), [(0., 1., True), (-.02, .98, False)])
def test_nominal_mount_offset_sensitivity(tmp_path, gpu_stream, offset, distance, near_valid):
    plan = make_range_plan(tmp_path)
    # 研究fixture直接改变冻结安装定义；不是训练API的用户安装覆盖入口。
    plan.experiment.sensors['scout_group']['outputs']['range']['position'][2] = offset
    env = create_env(plan)
    try:
        sample = place(env, [[0, 0, 1.], [4, 0, 1.], [8, 0, 1.]])
        np.testing.assert_allclose(sample.data.cpu(), distance, atol=2e-5)
        sample = place(env, [[0, 0, .055], [4, 0, .055], [8, 0, .055]])
        assert sample.valid.tolist() == [[near_valid]*3]*2
        np.testing.assert_allclose(sample.data.cpu(), .055 if near_valid else 4., atol=2e-5)
    finally:
        env.close()


def test_center_ray_does_not_represent_finite_field_of_view(range_env):
    env = range_env
    # 箱体x边界11：中心束在外侧.1m，27°视场在1m距离的半径约.24m仍覆盖箱体。
    sample = place(env, [[11.1, 0, 2.01], [20, 0, 2.01], [30, 0, 2.01]])
    assert .1 < math.tan(math.radians(13.5))
    np.testing.assert_allclose(sample.data[:, 0, 0].cpu(), 2., atol=2e-5)
    term = env.action_manager.get_term('scout_group')
    env.sim.data.qpos[:, term.qpos_ids[0, 0]] = 10.9
    sample = publish(env)
    np.testing.assert_allclose(sample.data[:, 0, 0].cpu(), 1., atol=2e-5)


def test_record_rangefinder_cost(tmp_path, gpu_stream, record_property):
    for imu_only in (True, False):
        env = create_env(make_range_plan(tmp_path / str(imu_only), imu_only=imu_only, period=100))
        try:
            action = commands(env)
            for _ in range(3):
                env.step(action)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                result = env.step(action)
                assert not result.transition.world_reset.any()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / 10
            record_property('imu_only_s' if imu_only else 'imu_range_s', elapsed)
            print(f'RANGE_BENCH imu_only={imu_only} decision_s={elapsed:.6f}')
        finally:
            env.close()
