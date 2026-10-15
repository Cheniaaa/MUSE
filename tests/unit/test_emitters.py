"""声明式emitter的名义时间、联合状态和world随机流隔离。"""
import pytest
import torch

from muse_runtime.rng import WorldRng
from muse_sdk.network import (EmitterBinding, FixedEmitterConfig, MarkovEmitterConfig,
    PeriodicEmitterConfig, SweepEmitterConfig)


CONFIG_TYPES = {'fixed': FixedEmitterConfig, 'periodic': PeriodicEmitterConfig,
                'sweep': SweepEmitterConfig, 'markov': MarkovEmitterConfig}
STATES = [dict(active=[True, False], power_w=[2., 5.], channel_id=[7, -3]),
          dict(active=[False, True], power_w=[0., 4.], channel_id=[19, 7])]


class TraceRng(WorldRng):
    """记录真实WorldRng抽样边界，不替换其随机数实现。"""
    def __init__(self, seed=13, batch=2):
        super().__init__(seed, batch, 'cpu')
        self.calls = []

    def uniform(self, world_ids, instance, purpose, shape):
        ids = tuple(int(world) for world in world_ids)
        self.calls.append((ids, instance, purpose, shape))
        return super().uniform(ids, instance, purpose, shape)


def binding(behavior, config, *, emitter_id='background', indices=(3, 0)):
    return EmitterBinding(emitter_id=emitter_id, endpoint_indices=indices, behavior=behavior,
                          config=CONFIG_TYPES[behavior].model_validate(config))


def process(declaration, *, batch=2, rng=None):
    from muse_runtime.network.emitters import EmitterProcess
    return EmitterProcess(declaration, batch_size=batch, device='cpu',
                          rng=WorldRng(13, batch, 'cpu') if rng is None else rng)


def markov_config(*, randomized=False):
    return dict(states=STATES, transition_matrix=[[.3, .7], [.8, .2]] if randomized else [[0., 1.], [1., 0.]],
                dwell_ticks=5, **({'initial_distribution': [.4, .6]} if randomized else {'initial_state': 0}))


@pytest.mark.parametrize('behavior,config', [
    ('fixed', dict(active=[True, False], power_w=[0., 2.], channel_id=[7, -3])),
    ('periodic', dict(on_ticks=5, off_ticks=7, phase_ticks=3, power_w=[2., 4.], channel_id=[7, -3])),
    ('sweep', dict(channel_ids=[7, -3, 19], dwell_ticks=5, initial_index=1,
                   active=[True, False], power_w=[2., 4.])),
    ('markov', markov_config(randomized=True)),
])
def test_constructor_allocates_fixed_cache_without_sampling(behavior, config):
    rng = TraceRng()
    emitter = process(binding(behavior, config), rng=rng)
    assert emitter.active.shape == emitter.power_w.shape == emitter.channel.shape == (2, 2)
    assert emitter.active.dtype == torch.bool
    assert emitter.power_w.dtype == torch.float32
    assert emitter.channel.dtype == torch.int32
    assert not emitter.active.any() and not emitter.power_w.any() and not emitter.channel.any()
    assert rng.calls == []
    if behavior == 'fixed':
        assert not hasattr(emitter, 'state_index') and not hasattr(emitter, 'next_switch_tick')
    else:
        assert emitter.state_index.shape == emitter.next_switch_tick.shape == (2,)
        assert emitter.state_index.dtype == emitter.next_switch_tick.dtype == torch.int64


@pytest.mark.parametrize('batch', [1, 3])
def test_fixed_emitter_keeps_declared_endpoint_order_and_zero_power(batch):
    declaration = binding('fixed', dict(active=[True, False, True], power_w=[0., 2., 3.],
                                        channel_id=[7, -3, 19]), indices=(4, 0, 2))
    emitter = process(declaration, batch=batch)
    pointers = [tensor.data_ptr() for tensor in (emitter.active, emitter.power_w, emitter.channel)]
    mask = torch.ones(batch, dtype=torch.bool)
    emitter.reset(mask)
    for tick in (0, 4, 8, 10, 100):
        emitter.advance(torch.full((batch,), tick, dtype=torch.int64))
        assert emitter.active.tolist() == [[True, False, True]] * batch
        assert emitter.power_w.tolist() == [[0., 2., 3.]] * batch
        assert emitter.channel.tolist() == [[7, -3, 19]] * batch
    emitter.reset(mask)
    assert [tensor.data_ptr() for tensor in (emitter.active, emitter.power_w, emitter.channel)] == pointers


def test_emitter_quantization_uses_nominal_deadlines():
    rng = TraceRng()
    emitter = process(binding('markov', markov_config()), rng=rng)
    emitter.reset(torch.ones(2, dtype=torch.bool))
    assert rng.calls == []  # 明确初态不需要随机抽样。
    states, deadlines = [], []
    for tick in (0, 4, 8, 10, 14, 18, 20):
        emitter.advance(torch.full((2,), tick, dtype=torch.int64))
        states.append(emitter.state_index.tolist())
        deadlines.append(emitter.next_switch_tick.tolist())
    assert states == [[0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [1, 1], [0, 0]]
    assert deadlines == [[5, 5], [5, 5], [10, 10], [15, 15], [15, 15], [20, 20], [25, 25]]
    assert len(rng.calls) == 4
    assert all(call[0] == (0, 1) and call[2:] == ('transition', ()) for call in rng.calls)


@pytest.mark.parametrize('phase,expected_on,expected_deadlines', [
    (3, [True, False, False, True, False, False, False], [2, 9, 9, 14, 21, 21, 21]),
    (7, [False, False, True, False, False, True, True], [5, 5, 10, 17, 17, 22, 22]),
])
def test_periodic_phase_and_nominal_deadlines(phase, expected_on, expected_deadlines):
    emitter = process(binding('periodic', dict(on_ticks=5, off_ticks=7, phase_ticks=phase,
                                              power_w=[2., 4.], channel_id=[7, -3])))
    emitter.reset(torch.ones(2, dtype=torch.bool))
    on, deadlines = [], []
    for tick in (0, 4, 8, 10, 14, 18, 20):
        emitter.advance(torch.full((2,), tick, dtype=torch.int64))
        on.append(emitter.active[0, 0].item())
        deadlines.append(emitter.next_switch_tick[0].item())
        assert emitter.active[0].tolist() == emitter.active[1].tolist() == [on[-1], on[-1]]
        assert emitter.power_w.tolist() == ([[2., 4.]] if on[-1] else [[0., 0.]]) * 2
        assert emitter.channel.tolist() == [[7, -3]] * 2
    assert on == expected_on
    assert deadlines == expected_deadlines


def test_sweep_keeps_declared_channel_order_and_exact_boundary():
    emitter = process(binding('sweep', dict(channel_ids=[7, -3, 19], dwell_ticks=5, initial_index=1,
                                          active=[True, False], power_w=[3., 9.])))
    emitter.reset(torch.ones(2, dtype=torch.bool))
    channels, deadlines = [], []
    for tick in (0, 4, 8, 10, 14, 18, 20):
        emitter.advance(torch.full((2,), tick, dtype=torch.int64))
        channels.append(emitter.channel[0, 0].item())
        deadlines.append(emitter.next_switch_tick[0].item())
        assert emitter.channel.tolist() == [[channels[-1], channels[-1]]] * 2
        assert emitter.active.tolist() == [[True, False]] * 2
        assert emitter.power_w.tolist() == [[3., 9.]] * 2
    assert channels == [-3, -3, 19, 7, 7, -3, 19]
    assert deadlines == [5, 5, 10, 15, 15, 20, 25]


def test_joint_markov_state_updates_all_endpoints_from_one_draw():
    rng = TraceRng()
    emitter = process(binding('markov', markov_config()), rng=rng)
    emitter.reset(torch.ones(2, dtype=torch.bool))
    assert emitter.active.tolist() == [[True, False]] * 2
    assert emitter.power_w.tolist() == [[2., 5.]] * 2
    assert emitter.channel.tolist() == [[7, -3]] * 2
    emitter.advance(torch.tensor([5, 4], dtype=torch.int64))
    assert emitter.state_index.tolist() == [1, 0]
    assert emitter.next_switch_tick.tolist() == [10, 5]
    assert emitter.active.tolist() == [[False, True], [True, False]]
    assert emitter.power_w.tolist() == [[0., 4.], [2., 5.]]
    assert emitter.channel.tolist() == [[19, 7], [7, -3]]
    assert rng.calls == [((0,), 'emitter:background', 'transition', ())]


def test_only_due_or_reset_worlds_draw_and_empty_masks_do_nothing():
    rng = TraceRng(batch=3)
    emitter = process(binding('markov', markov_config(randomized=True)), batch=3, rng=rng)
    emitter.reset(torch.tensor([False, True, True]))
    assert rng.calls == [((1, 2), 'emitter:background', 'init', ())]
    saved = [tensor.clone() for tensor in (emitter.active, emitter.power_w, emitter.channel,
                                          emitter.state_index, emitter.next_switch_tick)]
    emitter.reset(torch.zeros(3, dtype=torch.bool))
    for current, old in zip((emitter.active, emitter.power_w, emitter.channel,
                             emitter.state_index, emitter.next_switch_tick), saved):
        torch.testing.assert_close(current, old, rtol=0, atol=0)
    # 全B advance前由宿主补齐world0首reset，不增加initialized探测。
    emitter.reset(torch.tensor([True, False, False]))
    assert rng.calls[1] == ((0,), 'emitter:background', 'init', ())
    emitter.advance(torch.tensor([4, 4, 4], dtype=torch.int64))
    assert len(rng.calls) == 2
    emitter.advance(torch.tensor([4, 5, 4], dtype=torch.int64))
    assert rng.calls[2] == ((1,), 'emitter:background', 'transition', ())
    assert len(rng.calls) == 3
    for current, old in zip((emitter.active, emitter.power_w, emitter.channel,
                             emitter.state_index, emitter.next_switch_tick), saved):
        torch.testing.assert_close(current[2], old[2], rtol=0, atol=0)


def test_subset_reset_keeps_other_world_future_draws():
    declaration = binding('markov', markov_config(randomized=True))
    control_rng, reset_rng = TraceRng(), TraceRng()
    control, reset = process(declaration, rng=control_rng), process(declaration, rng=reset_rng)
    for emitter in (control, reset):
        emitter.reset(torch.ones(2, dtype=torch.bool))
        emitter.advance(torch.tensor([5, 5], dtype=torch.int64))
    pointers = [tensor.data_ptr() for tensor in (reset.active, reset.power_w, reset.channel,
                                                reset.state_index, reset.next_switch_tick)]
    reset.reset(torch.tensor([True, False]))
    assert reset_rng.calls[-1] == ((0,), 'emitter:background', 'init', ())
    for step in range(2, 10):
        control.advance(torch.tensor([step * 5, step * 5], dtype=torch.int64))
        reset.advance(torch.tensor([(step - 1) * 5, step * 5], dtype=torch.int64))
        for name in ('active', 'power_w', 'channel', 'state_index', 'next_switch_tick'):
            torch.testing.assert_close(getattr(control, name)[1], getattr(reset, name)[1], rtol=0, atol=0)
    torch.testing.assert_close(control_rng.uniform([1], 'emitter:background', 'transition', (8,)),
                               reset_rng.uniform([1], 'emitter:background', 'transition', (8,)), rtol=0, atol=0)
    assert [tensor.data_ptr() for tensor in (reset.active, reset.power_w, reset.channel,
                                           reset.state_index, reset.next_switch_tick)] == pointers


def test_init_and_transition_purposes_are_independent():
    declaration = binding('markov', markov_config(randomized=True))
    control_rng, reset_rng = TraceRng(), TraceRng()
    control, reset = process(declaration, rng=control_rng), process(declaration, rng=reset_rng)
    mask = torch.ones(2, dtype=torch.bool)
    control.reset(mask)
    reset.reset(mask)
    reset.reset(mask)
    # 不同初态抽样次数不能改变尚未消费的transition随机流。
    torch.testing.assert_close(control_rng.uniform(range(2), 'emitter:background', 'transition', (8,)),
                               reset_rng.uniform(range(2), 'emitter:background', 'transition', (8,)), rtol=0, atol=0)


def test_markov_initial_distribution_and_zero_probability_states():
    config = dict(states=STATES, transition_matrix=[[1., 0.], [1., 0.]], dwell_ticks=5,
                  initial_distribution=[0., 1.])
    emitter = process(binding('markov', config))
    emitter.reset(torch.ones(2, dtype=torch.bool))
    assert emitter.state_index.tolist() == [1, 1]
    emitter.advance(torch.tensor([5, 5], dtype=torch.int64))
    assert emitter.state_index.tolist() == [0, 0]


def test_markov_supports_three_states_and_one_endpoint():
    config = dict(states=[dict(active=[True], power_w=[1.], channel_id=[7]),
                          dict(active=[False], power_w=[2.], channel_id=[-3]),
                          dict(active=[True], power_w=[3.], channel_id=[19])],
                  transition_matrix=[[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]],
                  dwell_ticks=5, initial_state=2)
    emitter = process(binding('markov', config, indices=(6,)), batch=3)
    emitter.reset(torch.ones(3, dtype=torch.bool))
    assert emitter.channel.tolist() == [[19]] * 3
    emitter.advance(torch.tensor([5, 0, 5], dtype=torch.int64))
    assert emitter.state_index.tolist() == [0, 2, 0]
    assert emitter.active.tolist() == [[True]] * 3
    assert emitter.power_w.tolist() == [[1.], [3.], [1.]]
    assert emitter.channel.tolist() == [[7], [19], [7]]


def test_markov_draws_match_independent_scalar_probability_oracle():
    emitter = process(binding('markov', markov_config(randomized=True)))
    oracle = WorldRng(13, 2, 'cpu')
    initial = oracle.uniform(range(2), 'emitter:background', 'init', ()).tolist()
    expected = [0 if draw < .4 else 1 for draw in initial]
    emitter.reset(torch.ones(2, dtype=torch.bool))
    assert emitter.state_index.tolist() == expected
    for tick in (5, 10, 15, 20, 25, 30):
        draws = oracle.uniform(range(2), 'emitter:background', 'transition', ()).tolist()
        expected = [0 if draw < (.3 if old == 0 else .8) else 1 for old, draw in zip(expected, draws)]
        emitter.advance(torch.full((2,), tick, dtype=torch.int64))
        assert emitter.state_index.tolist() == expected


def test_distinct_emitter_ids_isolate_shared_rng():
    config = markov_config(randomized=True)
    shared = WorldRng(13, 2, 'cpu')
    target = process(binding('markov', config), rng=shared)
    unrelated = process(binding('markov', config, emitter_id='unrelated'), rng=shared)
    control = process(binding('markov', config))
    mask = torch.ones(2, dtype=torch.bool)
    unrelated.reset(mask)
    target.reset(mask)
    control.reset(mask)
    for tick in (5, 10, 15, 20, 25, 30):
        unrelated.advance(torch.full((2,), tick, dtype=torch.int64))
        target.advance(torch.full((2,), tick, dtype=torch.int64))
        control.advance(torch.full((2,), tick, dtype=torch.int64))
        for name in ('state_index', 'active', 'power_w', 'channel', 'next_switch_tick'):
            torch.testing.assert_close(getattr(target, name), getattr(control, name), rtol=0, atol=0)
