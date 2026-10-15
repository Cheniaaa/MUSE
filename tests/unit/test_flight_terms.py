from types import SimpleNamespace

import torch
from pathlib import Path


def test_distance_scores_action_target_observation_uses_next_target():
    from muse_builtins.terms import distance_reward, relative_target, collision, timeout
    env = SimpleNamespace(
        role_positions_world={'scout': torch.tensor([[[0., 0., 0.]], [[1., 0., 0.]]])},
        commands=SimpleNamespace(target_world={'scout': torch.tensor([[[1., 2., 3.]], [[4., 0., 0.]]])}),
        act_commands=SimpleNamespace(target_world={'scout': torch.tensor([[[3., 4., 0.]], [[1., 0., 0.]]])}),
        physics_outputs=SimpleNamespace(crashed={'scout': torch.tensor([[True], [False]])},
                                        episode_decision=torch.tensor([3, 4])),
    )
    torch.testing.assert_close(distance_reward(env, role_id='scout', scale=2.), torch.tensor([[-10.], [0.]]))
    torch.testing.assert_close(relative_target(env, role_id='scout'), torch.tensor([[[1., 2., 3.]], [[3., 0., 0.]]]))
    assert collision(env, controlled_role='scout', reduction='any').tolist() == [True, False]
    assert timeout(env, episode_max_steps=4).tolist() == [False, True]


def test_rule_returns_typed_motion_without_learner():
    from muse_builtins.rules import ConstantVelocity
    from muse_sdk.runtime_types import ActResult, MotionAction
    rule = ConstantVelocity({'motion': [0.05, 0., 0., -0.05]})
    experiment = SimpleNamespace(topology=SimpleNamespace(world_count=2),
                                 bindings={'scout': SimpleNamespace(role_slot_count=1),
                                           'guard': SimpleNamespace(role_slot_count=2)})
    rule.initialize(experiment, SimpleNamespace(device='cpu'), None)
    rule.set_mode('EVALUATE')
    result = rule.act(None, None, None, deterministic=True)
    assert isinstance(result, ActResult)
    assert isinstance(result.role_actions['guard'], MotionAction)
    assert result.role_actions['guard'].motion.shape == (2, 2, 4)
    torch.testing.assert_close(result.role_actions['scout'].motion[1, 0], torch.tensor([.05, 0., 0., -.05]))
    assert result.act_aux == () and not hasattr(rule, 'optimizer') and not hasattr(rule, 'update')


def test_platform_revision_changes_when_sdk_runtime_or_assets_change(tmp_path, monkeypatch):
    from muse_builtins import metadata
    root = tmp_path / 'src'
    builtin = root / 'muse_builtins'
    asset = builtin / 'assets/bitcraze_crazyflie_2'
    asset.mkdir(parents=True)
    (asset / 'cf2.xml').write_text('<mujoco><worldbody><body name="cf2"><geom size=".1"/></body></worldbody></mujoco>')
    car = builtin / 'assets/ugv_diff_drive'
    car.mkdir()
    (car / 'car.xml').write_text('<mujoco/>')
    station = builtin / 'assets/radio_base_station'
    station.mkdir()
    (station / 'station.xml').write_text('<mujoco><worldbody><body name="station"><geom size=".1"/></body></worldbody></mujoco>')
    for name in ('metadata.py', 'terms.py', 'rules.py', 'controllers.py'):
        (builtin / name).write_text('# fixed builtin\n')
    for package in ('muse_sdk', 'muse_runtime', 'muse_server'):
        (root / package).mkdir()
        (root / package / '__init__.py').write_text('# version one\n')
    monkeypatch.setattr(metadata, '__file__', str(builtin / 'metadata.py'))
    config = Path(__file__).resolve().parents[2] / 'examples/basic_flight/controller.json'
    first = metadata.export_metadata(config).platform_revision
    (root / 'muse_sdk/__init__.py').write_text('# version two\n')
    second = metadata.export_metadata(config).platform_revision
    assert first != second
    (root / 'muse_runtime/__init__.py').write_text('# version two\n')
    third = metadata.export_metadata(config)
    assert second != third.platform_revision
    # 平台包递归覆盖资产；删除额外digest汇入后仍能识别实际资源变化。
    (asset / 'cf2.xml').write_text((asset / 'cf2.xml').read_text().replace('size=".1"', 'size=".2"'))
    changed = metadata.export_metadata(config)
    assert third.platform_revision != changed.platform_revision
    assert third.entity_models['bitcraze_crazyflie_2'].sha256 != changed.entity_models['bitcraze_crazyflie_2'].sha256


def test_imu_terms_project_shared_group_slots_without_mutating_samples():
    from muse_builtins import terms
    from muse_sdk.runtime_types import SensorSample
    data = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    gyro = SensorSample(data, torch.tensor([[True, False, True], [False, True, False]]),
                        torch.tensor([[.1, .2, .3], [.4, .5, .6]]), torch.ones(2, 3, dtype=torch.int64))
    force = SensorSample(data + 100, ~gyro.valid, gyro.age + 1, gyro.sample_tick + 2)
    bindings = {role: SimpleNamespace(group_id='fleet', role_slot_to_group_slot=slots)
                for role, slots in [('scout', [1]), ('guard', [2, 0])]}
    env = SimpleNamespace(experiment=SimpleNamespace(bindings=bindings),
                          sensor_outputs={'fleet': {'gyro': gyro, 'specific_force': force}})
    before = {name: tuple(x.clone() for x in sample) for name, sample in env.sensor_outputs['fleet'].items()}
    for role, slots in [('scout', [1]), ('guard', [2, 0])]:
        for _ in range(2):
            torch.testing.assert_close(terms.imu_gyro(env, role_id=role), data[:, slots])
            torch.testing.assert_close(terms.imu_specific_force(env, role_id=role), (data + 100)[:, slots])
            valid = terms.imu_valid(env, role_id=role)
            assert valid.dtype == torch.float32
            torch.testing.assert_close(valid, torch.stack((gyro.valid, force.valid), dim=-1)[:, slots].float())
            torch.testing.assert_close(terms.imu_age(env, role_id=role),
                                       torch.stack((gyro.age, force.age), dim=-1)[:, slots])
    for name, sample in env.sensor_outputs['fleet'].items():
        for actual, expected in zip(sample, before[name]):
            torch.testing.assert_close(actual, expected)


def test_rangefinder_terms_read_only_scalar_projection_of_reordered_shared_group():
    from muse_builtins import terms
    from muse_sdk.runtime_types import SensorSample
    sample = SensorSample(
        torch.tensor([[[.04], [4.], [1.2]], [[4.], [.8], [2.3]]]),
        torch.tensor([[True, False, True], [False, True, True]]),
        torch.tensor([[.1, .2, .3], [.4, .5, .6]]),
        torch.tensor([[0, 13, 26], [39, 52, 65]], dtype=torch.int64))
    bindings = {role: SimpleNamespace(group_id='fleet', role_slot_to_group_slot=slots)
                for role, slots in [('scout', [1]), ('guard', [2, 0])]}
    env = SimpleNamespace(experiment=SimpleNamespace(bindings=bindings),
                          sensor_outputs={'fleet': {'range': sample}})
    before = tuple(value.clone() for value in sample)
    for role, slots, distance, valid, age in (
        ('scout', [1], [[[4.]], [[.8]]], [[[0.]], [[1.]]], [[[.2]], [[.5]]]),
        ('guard', [2, 0], [[[1.2], [.04]], [[2.3], [4.]]],
         [[[1.], [1.]], [[1.], [0.]]], [[[.3], [.1]], [[.6], [.4]]]),
    ):
        for _ in range(2):
            for name, expected in [('rangefinder_distance', distance), ('rangefinder_valid', valid),
                                   ('rangefinder_age', age)]:
                output = getattr(terms, name)(env, role_id=role)
                assert output.dtype == torch.float32 and output.shape == (2, len(slots), 1)
                torch.testing.assert_close(output, torch.tensor(expected))
    for actual, expected in zip(sample, before):
        torch.testing.assert_close(actual, expected)
