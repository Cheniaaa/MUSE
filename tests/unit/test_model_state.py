"""模型world边界、缓存所有权及失败提交的真实调用验证。"""
import pytest
import torch

from model_helpers import create_instance, make_stateful_case
from muse_sdk.models import ModelResetMask


def leaves(instance):
    return [instance.state['memory'], instance.diagnostics['calls'], instance.sim_time, instance.interval_s,
            *instance.output.link, *instance.output.rate]


def reset_inputs(case):
    return case.step_inputs._replace(sim_time=torch.tensor([0., 9.], dtype=torch.float64,
        device=case.kwargs['context'].device), interval_s=torch.tensor([0., 1.], dtype=torch.float64,
        device=case.kwargs['context'].device))


def test_model_constructor_does_not_reset_or_draw_rng():
    from muse_runtime.rng import WorldRng

    case = make_stateful_case(randomized=True)
    calls = []
    class Counting(case.implementation):
        def initial_state(self, batch_shape):
            calls.append('initial_state')
            return super().initial_state(batch_shape)

        def reset_state(self, *args):
            calls.append('reset')
            return super().reset_state(*args)

        def evaluate(self, *args):
            calls.append('evaluate')
            return super().evaluate(*args)

    case.kwargs['model_type'] = Counting
    global_rng = torch.get_rng_state().clone()
    instance = create_instance(case)
    assert calls == ['initial_state']
    assert not instance.state['memory'].any()
    assert not instance.diagnostics['calls'].any()
    assert instance.diagnostics['calls'].shape == (2,)
    assert instance.diagnostics['calls'].dtype == torch.int64
    assert not instance.output.link.valid.any()
    assert not instance.output.rate.valid.any()
    assert not instance.sim_time.any() and not instance.interval_s.any()
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    oracle = WorldRng(7, 2, 'cpu')
    torch.testing.assert_close(case.kwargs['rng'].uniform(range(2), 'radio_a', 'memory', (3, 5)),
        oracle.uniform(range(2), 'radio_a', 'memory', (3, 5)), rtol=0, atol=0)


def test_explicit_first_reset_runs_once():
    case = make_stateful_case()
    calls = []
    class Counting(case.implementation):
        def reset_state(self, state, mask, inputs, rng):
            calls.append(inputs.interval_s.clone())
            return super().reset_state(state, mask, inputs, rng)

    case.kwargs['model_type'] = Counting
    instance = create_instance(case)
    pointers = [value.data_ptr() for value in leaves(instance)]
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    assert len(calls) == 1
    assert calls[0].tolist() == [0., 0.]
    assert (instance.state['memory'] == 1).all()
    assert not instance.output.rate.rate_bps.any() and not instance.output.rate.valid.any()
    assert [value.data_ptr() for value in leaves(instance)] == pointers


def test_unrelated_construction_keeps_target_first_reset_rng():
    control_case = make_stateful_case(randomized=True)
    target_case = make_stateful_case(randomized=True)
    control, target = create_instance(control_case), create_instance(target_case)
    unrelated = make_stateful_case(instance_id='unrelated', randomized=True)
    unrelated.kwargs['rng'] = target_case.kwargs['rng']
    create_instance(unrelated)
    mask = ModelResetMask(torch.ones(2, dtype=torch.bool))
    control.reset(mask, control_case.reset_inputs)
    target.reset(mask, target_case.reset_inputs)
    torch.testing.assert_close(control.state['memory'], target.state['memory'], rtol=0, atol=0)


def test_world_reset_preserves_state_and_time_of_other_world():
    case = make_stateful_case()
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    pointers = [x.data_ptr() for x in leaves(instance)]
    assert (instance.state['memory'] == 1).all()
    instance.evaluate(case.step_inputs)
    instance.evaluate(case.step_inputs)
    assert (instance.state['memory'] == 5).all()
    assert instance.diagnostics['calls'].tolist() == [2, 2]
    instance.reset(ModelResetMask(torch.tensor([True, False])), reset_inputs(case))
    assert instance.state['memory'][:, 0, 0].tolist() == [1, 5]
    assert instance.diagnostics['calls'].tolist() == [0, 2]
    assert instance.sim_time.tolist() == [0, .4]
    assert instance.interval_s.tolist() == [0, .1]
    assert not instance.output.rate.valid[0].any()
    instance.evaluate(case.step_inputs)
    assert instance.state['memory'][:, 0, 0].tolist() == [3, 7]
    assert instance.diagnostics['calls'].tolist() == [1, 3]
    assert [x.data_ptr() for x in leaves(instance)] == pointers


@pytest.mark.parametrize('failure', ['diagnostics', 'rate', 'raise'])
def test_late_invalid_result_does_not_partially_commit(failure):
    case = make_stateful_case()
    class Broken(case.implementation):
        def evaluate(self, inputs, state, rng):
            result = super().evaluate(inputs, state, rng)
            if failure == 'raise':
                raise LookupError('model failure')
            if failure == 'diagnostics':
                result.diagnostics['calls'] = result.diagnostics['calls'].float()
            else:
                result.output.rate.rate_bps[0, 0, 0] = -1
            return result
    case.kwargs['model_type'] = Broken
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    saved = [x.clone() for x in leaves(instance)]
    with pytest.raises(LookupError if failure == 'raise' else RuntimeError):
        instance.evaluate(case.step_inputs)
    for actual, expected in zip(leaves(instance), saved):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_committed_cache_does_not_alias_model_buffers():
    case = make_stateful_case()
    returned = []
    class Reusing(case.implementation):
        def evaluate(self, inputs, state, rng):
            result = super().evaluate(inputs, state, rng)
            returned.append(result)
            return result
    case.kwargs['model_type'] = Reusing
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    initial = case.reset_inputs.geometry['distance'].clone()
    instance.evaluate(case.step_inputs)
    returned[0].next_state['memory'].fill_(100)
    returned[0].output.rate.rate_bps.fill_(100)
    returned[0].diagnostics['calls'].fill_(100)
    assert (instance.state['memory'] == 3).all()
    assert instance.output.rate.rate_bps[0, 0, 0] == 10
    assert instance.diagnostics['calls'].tolist() == [1, 1]
    torch.testing.assert_close(case.reset_inputs.geometry['distance'], initial)


def test_subset_reset_ignores_unselected_numeric_values():
    case = make_stateful_case()
    class Partial(case.implementation):
        def reset_state(self, state, reset_mask, inputs, rng):
            result = super().reset_state(state, reset_mask, inputs, rng)
            ignored = ~reset_mask.world_reset
            result.next_state['memory'][ignored] = float('nan')
            result.output.link.received_power_w[ignored] = float('nan')
            return result
    case.kwargs['model_type'] = Partial
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    instance.evaluate(case.step_inputs)
    saved = [x[1].clone() for x in leaves(instance)]
    instance.reset(ModelResetMask(torch.tensor([True, False])), reset_inputs(case))
    for actual, expected in zip(leaves(instance), saved):
        torch.testing.assert_close(actual[1], expected, rtol=0, atol=0)


def test_empty_reset_does_not_call_model_or_require_inputs():
    case = make_stateful_case()
    calls = []
    class Counting(case.implementation):
        def reset_state(self, *args):
            calls.append('reset')
            return super().reset_state(*args)
        def evaluate(self, *args):
            calls.append('evaluate')
            return super().evaluate(*args)
    case.kwargs['model_type'] = Counting
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    instance.reset(ModelResetMask(torch.zeros(2, dtype=torch.bool)), None)
    assert calls == ['reset']


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_invalid_evaluate_interval_rejected_before_model_call(value):
    case = make_stateful_case()
    class Unreachable(case.implementation):
        def evaluate(self, *args):
            pytest.fail('invalid time reached model')
    case.kwargs['model_type'] = Unreachable
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    inputs = case.step_inputs._replace(interval_s=torch.tensor([value, .1], dtype=torch.float64))
    with pytest.raises(RuntimeError, match='interval_s'):
        instance.evaluate(inputs)


def test_subset_reset_preserves_other_world_future_rng_sequence():
    control_case = make_stateful_case(randomized=True)
    reset_case = make_stateful_case(randomized=True)
    control, reset = create_instance(control_case), create_instance(reset_case)
    control.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), control_case.reset_inputs)
    reset.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), reset_case.reset_inputs)
    control.evaluate(control_case.step_inputs)
    reset.evaluate(reset_case.step_inputs)
    reset.reset(ModelResetMask(torch.tensor([True, False])), reset_inputs(reset_case))
    for _ in range(3):
        control.evaluate(control_case.step_inputs)
        reset.evaluate(reset_case.step_inputs)
        torch.testing.assert_close(control.state['memory'][1], reset.state['memory'][1], rtol=0, atol=0)


def test_model_instance_ids_isolate_shared_rng():
    from muse_runtime.rng import WorldRng
    shared = WorldRng(7, 2, 'cpu')
    cases = [make_stateful_case(instance_id=name, randomized=True) for name in ('a', 'b')]
    for case in cases:
        case.kwargs['rng'] = shared
    instances = [create_instance(case) for case in cases]
    for instance, case in zip(instances, cases):
        instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    independent_case = make_stateful_case(instance_id='b', randomized=True)
    independent = create_instance(independent_case)
    independent.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), independent_case.reset_inputs)
    for _ in range(3):
        instances[0].evaluate(cases[0].step_inputs)
        instances[1].evaluate(cases[1].step_inputs)
        independent.evaluate(independent_case.step_inputs)
    torch.testing.assert_close(instances[1].state['memory'], independent.state['memory'], rtol=0, atol=0)


def test_state_cross_aliases_are_committed_from_same_previous_snapshot():
    case = make_stateful_case()
    descriptor = case.kwargs['descriptor']
    case.kwargs['descriptor'] = descriptor.model_copy(update={'state_schema': {
        'memory': descriptor.state_schema['memory'], 'other': descriptor.state_schema['memory']}})
    class Swapping(case.implementation):
        def initial_state(self, batch_shape):
            return {**super().initial_state(batch_shape), 'other': torch.full((2, 3, 5), 9.)}

        def reset_state(self, state, reset_mask, inputs, rng):
            result = super().reset_state(state, reset_mask, inputs, rng)
            return result._replace(next_state={**result.next_state, 'other': state['other']})

        def evaluate(self, inputs, state, rng):
            result = super().evaluate(inputs, state, rng)
            return result._replace(next_state={'other': state['memory'], 'memory': state['other']})
    case.kwargs['model_type'] = Swapping
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    instance.evaluate(case.step_inputs)
    assert (instance.state['memory'] == 9).all()
    assert (instance.state['other'] == 1).all()


@pytest.mark.parametrize('failure', ['initial_dtype', 'state_nonfinite', 'state_key', 'diagnostic_nonfinite'])
def test_model_state_structure_and_finite_boundary(failure):
    case = make_stateful_case()
    if failure == 'diagnostic_nonfinite':
        descriptor = case.kwargs['descriptor']
        case.kwargs['descriptor'] = descriptor.model_copy(update={'diagnostics_schema': {
            'calls': descriptor.diagnostics_schema['calls'].model_copy(update={'dtype': 'float32'})}})
    class Broken(case.implementation):
        def initial_state(self, batch_shape):
            state = super().initial_state(batch_shape)
            if failure == 'initial_dtype':
                state['memory'] = state['memory'].double()
            return state

        def reset_state(self, state, reset_mask, inputs, rng):
            result = super().reset_state(state, reset_mask, inputs, rng)
            if failure == 'diagnostic_nonfinite':
                result.diagnostics['calls'] = torch.zeros(2)
            return result

        def evaluate(self, inputs, state, rng):
            result = super().evaluate(inputs, state, rng)
            if failure == 'state_key':
                result.next_state['extra'] = torch.zeros(2)
            elif failure == 'state_nonfinite':
                result.next_state['memory'][0, 0, 0] = float('inf')
            elif failure == 'diagnostic_nonfinite':
                result.diagnostics['calls'] = torch.full((2,), float('nan'))
            return result
    case.kwargs['model_type'] = Broken
    with pytest.raises(RuntimeError, match='state|diagnostics'):
        instance = create_instance(case)
        instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
        instance.evaluate(case.step_inputs)


@pytest.mark.parametrize('value', [-1., float('nan'), float('inf')])
def test_invalid_host_time_rejected(value):
    case = make_stateful_case()
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    with pytest.raises(RuntimeError, match='sim_time'):
        instance.evaluate(case.step_inputs._replace(sim_time=torch.tensor([value, .1], dtype=torch.float64)))


def test_nonzero_reset_interval_rejected_before_call():
    case = make_stateful_case()
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    with pytest.raises(RuntimeError, match='interval_s'):
        instance.reset(ModelResetMask(torch.tensor([True, False])), case.step_inputs)


@pytest.mark.parametrize('all_worlds', [False, True])
def test_reset_mask_borrowed_from_cache_keeps_one_selection_through_commit(all_worlds):
    case = make_stateful_case()
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    instance.evaluate(case.step_inputs)
    saved = [x.clone() for x in leaves(instance)]
    pointers = [x.data_ptr() for x in leaves(instance)]
    mask = instance.output.rate.valid[:, 0, 0] if all_worlds else instance.output.rate.valid[:, 2, 4]
    assert mask.tolist() == [True, all_worlds]
    instance.reset(ModelResetMask(mask), case.reset_inputs)
    assert instance.state['memory'][:, 0, 0].tolist() == [1, 1 if all_worlds else 3]
    assert instance.diagnostics['calls'].tolist() == [0, 0 if all_worlds else 1]
    assert instance.sim_time.tolist() == [0, 0 if all_worlds else .4]
    assert instance.interval_s.tolist() == [0, 0 if all_worlds else .1]
    assert not instance.output.rate.valid[0].any()
    assert instance.output.rate.rate_bps[0].count_nonzero() == 0
    if all_worlds:
        assert not instance.output.rate.valid.any()
        assert instance.output.rate.rate_bps.count_nonzero() == 0
    else:
        for actual, expected in zip(leaves(instance), saved):
            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert [x.data_ptr() for x in leaves(instance)] == pointers
