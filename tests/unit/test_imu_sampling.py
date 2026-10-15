from types import SimpleNamespace

import pytest
import torch

from muse_sdk.runtime_types import SensorSample


def sampling_case(periods=None, active=None):
    periods = {'fleet': 3} if periods is None else periods
    active = list(periods) if active is None else active
    counts = {'fleet': 3, 'reserve': 1}
    profiles = {group: {
        'sensor_profile_id': 'cf2_imu',
        'outputs': {
            'gyro': {'sensor_name': 'body_gyro', 'units': 'rad/s', 'frame': 'imu_flu',
                     'width': 3, 'sensor_type': 'gyro', 'period_ticks': period},
            'specific_force': {'sensor_name': 'body_linacc', 'units': 'm/s²', 'frame': 'imu_flu',
                               'width': 3, 'sensor_type': 'accelerometer', 'period_ticks': period},
        },
        'config': {'update_rate_hz': 1 / (period * .002)},
    } for group, period in periods.items()}
    experiment = SimpleNamespace(
        topology=SimpleNamespace(world_count=2),
        scenario=SimpleNamespace(
            entity_groups=[SimpleNamespace(group_id=group, count=counts[group]) for group in periods],
            scene=SimpleNamespace(physics_dt=.002, control_substeps=20)),
        sensors=profiles,
        activation={'sensor:' + group: ['task:consume_imu'] for group in active},
    )
    sources = {}
    for group in active:
        for slot in range(counts[group]):
            for native in ('body_gyro', 'body_linacc'):
                # 原生BuiltinSensor返回sensordata的非连续[B,3] view。
                buffer = torch.zeros(2, 7)
                sources[f'{group}_{slot}/{native}'] = SimpleNamespace(data=buffer[:, 2:5])
    return experiment, SimpleNamespace(sensors=sources)


def make_sampler(experiment, scene):
    from muse_runtime.sensors import SensorSampler
    return SensorSampler(experiment, scene, 'cpu')


def fill_sources(scene, values):
    for index, source in enumerate(scene.sensors.values()):
        source.data.copy_(torch.tensor(values, dtype=torch.float32)[:, None] + index * 100)


def test_constructs_fixed_group_buffers_without_reading_measurements():
    experiment, scene = sampling_case()
    for source in scene.sensors.values():
        source.data.fill_(float('nan'))
    sampler = make_sampler(experiment, scene)
    assert set(sampler.outputs) == {'fleet'}
    assert set(sampler.outputs['fleet']) == {'gyro', 'specific_force'}
    for sample in sampler.outputs['fleet'].values():
        assert isinstance(sample, SensorSample)
        assert sample.data.shape == (2, 3, 3)
        assert sample.data.dtype == torch.float32
        for field, dtype in ((sample.valid, torch.bool), (sample.age, torch.float32),
                             (sample.sample_tick, torch.int64)):
            assert field.shape == (2, 3)
            assert field.dtype == dtype
            assert field.device.type == 'cpu'
            assert not field.any()
        assert not sample.data.any()


def test_only_active_groups_allocate_outputs_and_due_contains_all_active_groups():
    experiment, scene = sampling_case({'fleet': 3, 'reserve': 5}, active=['fleet'])
    sampler = make_sampler(experiment, scene)
    assert set(sampler.outputs) == {'fleet'}
    assert set(sampler.due(torch.tensor([1, 2]))) == {'fleet'}
    experiment, scene = sampling_case({'fleet': 3, 'reserve': 5})
    sampler = make_sampler(experiment, scene)
    masks = sampler.due(torch.tensor([3, 5]))
    assert set(masks) == {'fleet', 'reserve'}
    for signal in ('gyro', 'specific_force'):
        assert masks['fleet'][signal].tolist() == [True, False]
        assert masks['reserve'][signal].tolist() == [False, True]
    for signals in masks.values():
        for mask in signals.values():
            assert mask.shape == (2,)
            assert mask.dtype == torch.bool


def test_no_consumer_never_binds_native_sources_or_allocates_samples():
    experiment, scene = sampling_case(active=[])
    sampler = make_sampler(experiment, scene)
    assert sampler.outputs == {}
    assert sampler.due(torch.tensor([3, 6])) == {}
    sampler.reset(torch.tensor([0, 1]))
    sampler.capture(torch.tensor([3, 6]), {})
    sampler.refresh_age(torch.tensor([20, 40]))
    assert sampler.outputs == {}
    experiment.sensors['fleet'] = {'sensor_profile_id': 'none', 'outputs': {}, 'config': {}}
    assert make_sampler(experiment, scene).outputs == {}


@pytest.mark.parametrize(('period', 'expected_ticks', 'expected_ages'), [
    (3, [18, 39], [.004, .002]),
    (30, [0, 30], [.040, .020]),
])
def test_integer_sampling_holds_across_decisions(period, expected_ticks, expected_ages):
    experiment, scene = sampling_case({'fleet': period})
    fill_sources(scene, [0, 10])
    sampler = make_sampler(experiment, scene)
    sampler.reset(torch.tensor([0, 1]))
    for tick in range(1, 41):
        fill_sources(scene, [tick, tick + 10])
        ticks = torch.tensor([tick, tick])
        masks = sampler.due(ticks)
        if any(mask.any() for signals in masks.values() for mask in signals.values()):
            sampler.capture(ticks, masks)
        if tick in (20, 40):
            index = tick // 20 - 1
            sampler.refresh_age(ticks)
            for signal, offset in (('gyro', 0), ('specific_force', 100)):
                sample = sampler.outputs['fleet'][signal]
                assert sample.valid.all()
                assert sample.sample_tick.tolist() == [[expected_ticks[index]] * 3] * 2
                torch.testing.assert_close(sample.age, torch.full((2, 3), expected_ages[index]))
                torch.testing.assert_close(sample.data[:, 0, 0],
                    torch.tensor([expected_ticks[index] + offset, expected_ticks[index] + 10 + offset],
                                 dtype=torch.float32))


def test_reset_captures_real_values_all_slots_and_buffers_do_not_alias_source_views():
    experiment, scene = sampling_case()
    fill_sources(scene, [7, 11])
    sampler = make_sampler(experiment, scene)
    sampler.reset(torch.tensor([0, 1]))
    for signal, offsets in (('gyro', [0, 200, 400]), ('specific_force', [100, 300, 500])):
        sample = sampler.outputs['fleet'][signal]
        expected = torch.tensor([[7 + x for x in offsets], [11 + x for x in offsets]], dtype=torch.float32)
        torch.testing.assert_close(sample.data, expected[:, :, None].expand(2, 3, 3))
        assert sample.valid.all()
        assert not sample.sample_tick.any()
        assert not sample.age.any()
    before = {name: sample.data.clone() for name, sample in sampler.outputs['fleet'].items()}
    fill_sources(scene, [-99, -88])
    for name, sample in sampler.outputs['fleet'].items():
        torch.testing.assert_close(sample.data, before[name])
    sampler.outputs['fleet']['gyro'].data.fill_(42)
    torch.testing.assert_close(scene.sensors['fleet_0/body_gyro'].data[:, 0], torch.tensor([-99., -88.]))


def test_subset_reset_preserves_other_world_and_future_world_phases():
    experiment, scene = sampling_case()
    fill_sources(scene, [0, 10])
    sampler = make_sampler(experiment, scene)
    sampler.reset(torch.tensor([0, 1]))
    fill_sources(scene, [18, 28])
    sampler.capture(torch.tensor([18, 18]), sampler.due(torch.tensor([18, 18])))
    sampler.refresh_age(torch.tensor([20, 20]))
    before = {name: tuple(field[1].clone() for field in sample)
              for name, sample in sampler.outputs['fleet'].items()}
    fill_sources(scene, [100, 200])
    sampler.reset(torch.tensor([0]))
    for name, sample in sampler.outputs['fleet'].items():
        for actual, expected in zip(sample, before[name]):
            torch.testing.assert_close(actual[1], expected)
        assert sample.valid[0].all()
        assert not sample.sample_tick[0].any()
        assert not sample.age[0].any()
    masks = sampler.due(torch.tensor([1, 21]))
    assert masks['fleet']['gyro'].tolist() == [False, True]
    fill_sources(scene, [101, 221])
    sampler.capture(torch.tensor([1, 21]), masks)
    sampler.refresh_age(torch.tensor([1, 21]))
    sample = sampler.outputs['fleet']['gyro']
    assert sample.sample_tick.tolist() == [[0, 0, 0], [21, 21, 21]]
    torch.testing.assert_close(sample.data[:, 0, 0], torch.tensor([100., 221.]))
    torch.testing.assert_close(sample.age, torch.tensor([[.002] * 3, [0.] * 3]))
    masks = sampler.due(torch.tensor([3, 23]))
    assert masks['fleet']['gyro'].tolist() == [True, False]
    fill_sources(scene, [103, 223])
    sampler.capture(torch.tensor([3, 23]), masks)
    sampler.refresh_age(torch.tensor([3, 23]))
    assert sample.sample_tick.tolist() == [[3, 3, 3], [21, 21, 21]]
    torch.testing.assert_close(sample.data[:, 0, 0], torch.tensor([103., 221.]))
    torch.testing.assert_close(sample.age, torch.tensor([[0.] * 3, [.004] * 3]))


def test_capture_uses_supplied_masks_and_does_not_deduplicate_or_own_a_clock():
    experiment, scene = sampling_case({'fleet': 3, 'reserve': 5})
    sampler = make_sampler(experiment, scene)
    fill_sources(scene, [4, 8])
    masks = {group: {signal: torch.tensor(value) for signal in ('gyro', 'specific_force')}
             for group, value in [('fleet', [True, False]), ('reserve', [False, True])]}
    ticks = torch.tensor([7, 11])
    sampler.capture(ticks, masks)
    assert sampler.outputs['fleet']['gyro'].sample_tick.tolist() == [[7] * 3, [0] * 3]
    assert sampler.outputs['reserve']['gyro'].sample_tick.tolist() == [[0], [11]]
    assert sampler.outputs['fleet']['gyro'].valid.tolist() == [[True] * 3, [False] * 3]
    fill_sources(scene, [44, 88])
    sampler.capture(ticks, masks)
    torch.testing.assert_close(sampler.outputs['fleet']['gyro'].data[:, 0, 0], torch.tensor([44., 0.]))


def test_read_and_refresh_age_do_not_resample_or_change_measurement_ticks():
    experiment, scene = sampling_case()
    sampler = make_sampler(experiment, scene)
    fill_sources(scene, [6, 16])
    sampler.capture(torch.tensor([6, 6]), sampler.due(torch.tensor([6, 6])))
    before = {name: (sample.data.clone(), sample.valid.clone(), sample.sample_tick.clone())
              for name, sample in sampler.outputs['fleet'].items()}
    fill_sources(scene, [999, 999])
    for ticks in ([20, 40], [20, 40], [21, 42]):
        sampler.refresh_age(torch.tensor(ticks))
        for name, sample in sampler.outputs['fleet'].items():
            for actual, expected in zip((sample.data, sample.valid, sample.sample_tick), before[name]):
                torch.testing.assert_close(actual, expected)
    for sample in sampler.outputs['fleet'].values():
        torch.testing.assert_close(sample.age, torch.tensor([[.030] * 3, [.072] * 3]))


@pytest.mark.parametrize('native', ['body_gyro', 'body_linacc'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
@pytest.mark.parametrize('operation', ['capture', 'reset'])
def test_nonfinite_selected_native_measurement_fails_at_sampling_boundary(native, value, operation):
    experiment, scene = sampling_case()
    sampler = make_sampler(experiment, scene)
    fill_sources(scene, [1, 2])
    sampler.reset(torch.tensor([0, 1]))
    scene.sensors[f'fleet_2/{native}'].data[0, 1] = value
    with pytest.raises(RuntimeError, match='nonfinite.*fleet'):
        if operation == 'reset':
            sampler.reset(torch.tensor([0]))
        else:
            sampler.capture(torch.tensor([3, 4]), {'fleet': {
                signal: torch.tensor([True, False]) for signal in ('gyro', 'specific_force')}})


def test_nonfinite_unselected_worlds_and_non_due_groups_are_not_sampled():
    experiment, scene = sampling_case({'fleet': 3, 'reserve': 5})
    fill_sources(scene, [3, 4])
    sampler = make_sampler(experiment, scene)
    sampler.reset(torch.tensor([0, 1]))
    for name, source in scene.sensors.items():
        if name.startswith('reserve_'):
            source.data.fill_(float('nan'))
        else:
            source.data[1] = float('inf')
    sampler.capture(torch.tensor([3, 4]), sampler.due(torch.tensor([3, 4])))
    for sample in sampler.outputs['fleet'].values():
        assert sample.sample_tick.tolist() == [[3] * 3, [0] * 3]
        assert torch.isfinite(sample.data).all()
    for sample in sampler.outputs['reserve'].values():
        assert not sample.sample_tick.any()
        assert torch.isfinite(sample.data).all()
    experiment, scene = sampling_case()
    sampler = make_sampler(experiment, scene)
    fill_sources(scene, [9, 10])
    for source in scene.sensors.values():
        source.data[1] = float('nan')
    sampler.reset(torch.tensor([0]))
    for sample in sampler.outputs['fleet'].values():
        assert sample.valid.tolist() == [[True] * 3, [False] * 3]
        assert torch.isfinite(sample.data).all()
