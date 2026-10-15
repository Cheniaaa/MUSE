"""合成测量只验证异步采样/无可用距离语义，不代表硬件工作规格。"""
from types import SimpleNamespace

import pytest
import torch


def sampling_case():
    outputs = {
        'gyro': dict(sensor_name='body_gyro', width=3, sensor_type='gyro', period_ticks=5),
        'specific_force': dict(sensor_name='body_linacc', width=3, sensor_type='accelerometer', period_ticks=5),
        'range': dict(sensor_name='rangefinder_down', width=1, sensor_type='rangefinder',
                      period_ticks=13, min_range_m=.1, max_range_m=2.),
    }
    exp = SimpleNamespace(topology=SimpleNamespace(world_count=2),
        scenario=SimpleNamespace(scene=SimpleNamespace(physics_dt=.002),
            entity_groups=[SimpleNamespace(group_id='fleet', count=2)]),
        activation={'sensor:fleet': ['task']}, sensors={'fleet': {'outputs': outputs}})
    scene = SimpleNamespace(sensors={f'fleet_{slot}/{meta["sensor_name"]}':
        SimpleNamespace(data=torch.ones(2, meta['width']))
        for slot in range(2) for meta in outputs.values()})
    from muse_runtime.sensors import SensorSampler
    return SensorSampler(exp, scene, 'cpu'), scene


@pytest.mark.parametrize(('raw', 'distance', 'valid'), [(-1, 2, False), (0, 2, False),
    (.1, .1, True), (2, 2, True), (2.1, 2, False)])
def test_range_bounds_publish_unavailable_instead_of_clipping_to_valid(raw, distance, valid):
    sampler, scene = sampling_case()
    for slot in range(2):
        scene.sensors[f'fleet_{slot}/rangefinder_down'].data.fill_(raw)
    sampler.reset(torch.tensor([0, 1]))
    sample = sampler.outputs['fleet']['range']
    assert sample.data.shape == (2, 2, 1)
    torch.testing.assert_close(sample.data, torch.full((2, 2, 1), float(distance)))
    assert sample.valid.tolist() == [[valid]*2]*2
    assert sample.sample_tick.count_nonzero() == sample.age.count_nonzero() == 0


def test_independent_periods_hold_measurement_and_age_per_signal():
    sampler, scene = sampling_case()
    assert not sampler.outputs['fleet']['range'].valid.any()
    sampler.reset(torch.tensor([0, 1]))
    for tick in range(1, 41):
        for source in scene.sensors.values():
            source.data.fill_(tick / 100)
        ticks = torch.tensor([tick]*2)
        sampler.capture(ticks, sampler.due(ticks))
        if tick in (20, 40):
            expected = 13 if tick == 20 else 39
            sampler.refresh_age(ticks)
            samples = sampler.outputs['fleet']
            assert (samples['gyro'].sample_tick == tick).all()
            assert (samples['specific_force'].sample_tick == tick).all()
            assert (samples['range'].sample_tick == expected).all()
            torch.testing.assert_close(samples['range'].data, torch.full((2, 2, 1), expected / 100))
            torch.testing.assert_close(samples['range'].age,
                torch.full((2, 2), .014 if tick == 20 else .002))
            assert samples['gyro'].age.count_nonzero() == 0


def test_miss_updates_sample_tick_and_subset_reset_preserves_other_world():
    sampler, scene = sampling_case()
    sampler.reset(torch.tensor([0, 1]))
    for slot in range(2):
        scene.sensors[f'fleet_{slot}/rangefinder_down'].data[0] = -1
    sampler.capture(torch.tensor([13, 13]), sampler.due(torch.tensor([13, 13])))
    sampler.refresh_age(torch.tensor([20, 20]))
    sample = sampler.outputs['fleet']['range']
    assert sample.valid.tolist() == [[False]*2, [True]*2]
    assert (sample.sample_tick == 13).all()
    torch.testing.assert_close(sample.age, torch.full((2, 2), .014))
    saved = {name: tuple(field[1].clone() for field in value)
             for name, value in sampler.outputs['fleet'].items()}
    sampler.reset(torch.tensor([0]))
    for name, value in sampler.outputs['fleet'].items():
        for field, expected in zip(value, saved[name]):
            torch.testing.assert_close(field[1], expected, rtol=0, atol=0)
    assert sample.sample_tick[0].count_nonzero() == 0
    assert not sample.valid[0].any()


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_range_fails_only_when_selected_for_capture(value):
    sampler, scene = sampling_case()
    sampler.reset(torch.tensor([0, 1]))
    scene.sensors['fleet_1/rangefinder_down'].data[0] = value
    sampler.capture(torch.tensor([5, 5]), sampler.due(torch.tensor([5, 5])))
    with pytest.raises(RuntimeError, match='nonfinite.*fleet/range'):
        sampler.capture(torch.tensor([13, 5]), sampler.due(torch.tensor([13, 5])))
