"""实际模型调用边界，覆盖能力、有效性和阶段输入。"""
import itertools

import pytest
import torch

from model_helpers import create_instance, make_radio_case, make_sensing_case
from muse_sdk.descriptor import InputField
from muse_sdk.models import ModelResetMask, RadioEvaluationResult, LinkMetrics


COMBINATIONS = [c for n in (1, 2, 3) for c in itertools.combinations(('link', 'interference', 'rate'), n)]


@pytest.mark.parametrize('blocks', COMBINATIONS)
def test_radio_constructor_allocates_only_declared_invalid_placeholders(blocks):
    instance = create_instance(make_radio_case(blocks))
    for name in ('link', 'interference', 'rate'):
        block = getattr(instance.output, name)
        assert (block is not None) == (name in blocks)
        if block is not None:
            for value in block:
                assert value.shape == (2, 3, 5)
                assert value.dtype == (torch.bool if value is block.valid else torch.float32)
                assert not value.any()


@pytest.mark.parametrize('fields', [('measurement',), ('fim',), ('measurement', 'fim')])
def test_sensing_constructor_allocates_only_declared_invalid_placeholders(fields):
    instance = create_instance(make_sensing_case(fields))
    for name in ('measurement', 'fim'):
        value = getattr(instance.output, name)
        assert (value is not None) == (name in fields)
        if value is not None:
            assert value.shape == ((2, 4, 2) if name == 'measurement' else (2, 4, 2, 2))
            assert value.dtype == (torch.float32 if name == 'measurement' else torch.float64)
            assert not value.any()
            assert not instance.output.field_valid[name].any()
    assert set(instance.output.field_valid) == set(fields)
    assert instance.output.time_semantics == {name: 'snapshot' if name == 'measurement' else 'increment'
        for name in fields}


@pytest.mark.parametrize('blocks', COMBINATIONS)
def test_radio_seven_block_combinations(blocks):
    case = make_radio_case(blocks)
    instance = create_instance(case)
    cache = [value for block in instance.output if block is not None for value in block]
    pointers = [value.data_ptr() for value in cache]
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    if 'rate' in blocks:
        assert instance.output.rate.rate_bps.count_nonzero() == 0
        assert not instance.output.rate.valid.any()
    case.step_inputs.geometry['distance'][0, 0, 0] = 0
    result = instance.evaluate(case.step_inputs)
    assert result.next_state == ()
    for name in ('link', 'interference', 'rate'):
        assert (getattr(result.output, name) is not None) == (name in blocks)
    if 'rate' in blocks:
        assert result.output.rate.rate_bps[0, 0, 0] == 0
        assert result.output.rate.valid[0, 0, 0]
        assert result.output.rate.rate_bps[0, 0, 1] == 20
        assert not result.output.rate.valid[1, 2, 4]
    assert instance.interval_s.tolist() == [.1, .1]
    assert [value.data_ptr() for block in instance.output if block is not None for value in block] == pointers


def corrupt_radio(case, change):
    class Broken(case.implementation):
        def evaluate(self, inputs, state, rng):
            result = super().evaluate(inputs, state, rng)
            return result._replace(output=change(result.output))
    case.kwargs['model_type'] = Broken


@pytest.mark.parametrize('failure', ['none', 'extra', 'dict', 'shape', 'dtype', 'valid_dtype',
    'negative', 'nan', 'inf', 'unbound'])
def test_model_result_boundary_rejects_invalid_result(failure):
    case = make_radio_case()
    def change(output):
        rate = output.rate
        if failure == 'none':
            return output._replace(rate=None)
        if failure == 'extra':
            return output._replace(link=rate)
        if failure == 'dict':
            return output._asdict()
        if failure == 'shape':
            return output._replace(rate=rate._replace(rate_bps=rate.rate_bps.transpose(1, 2)))
        if failure == 'dtype':
            return output._replace(rate=rate._replace(rate_bps=rate.rate_bps.double()))
        if failure == 'valid_dtype':
            return output._replace(rate=rate._replace(valid=rate.valid.float()))
        if failure == 'unbound':
            rate.valid[1, 2, 4] = True
        else:
            rate.rate_bps[0, 0, 0] = {'negative': -1., 'nan': float('nan'), 'inf': float('inf')}[failure]
        return output
    corrupt_radio(case, change)
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    with pytest.raises(RuntimeError, match='model_a.*output'):
        instance.evaluate(case.step_inputs)


def test_invalid_numeric_values_are_not_mistaken_for_valid_measurements():
    case = make_radio_case()
    def change(output):
        output.rate.rate_bps[1, 2, 4] = float('nan')
        return output
    corrupt_radio(case, change)
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    result = instance.evaluate(case.step_inputs)
    assert torch.isnan(result.output.rate.rate_bps[1, 2, 4])
    assert not result.output.rate.valid[1, 2, 4]


@pytest.mark.parametrize('fields', [('measurement',), ('fim',), ('measurement', 'fim')])
def test_sensing_field_combinations_and_reset_semantics(fields):
    case = make_sensing_case(fields)
    instance = create_instance(case)
    pointers = [(getattr(instance.output, name).data_ptr(), instance.output.field_valid[name].data_ptr())
                for name in fields]
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    if 'fim' in fields:
        assert not instance.output.field_valid['fim'].any()
    output = instance.evaluate(case.step_inputs).output
    assert set(output.field_valid) == set(fields)
    if 'measurement' in fields:
        assert (output.measurement == -1).all()
        assert output.field_valid['measurement'].shape == (2, 4, 2)
    if 'fim' in fields:
        assert output.field_valid['fim'].shape == (2, 4)
        assert output.field_valid['fim'].all()
    assert [(getattr(output, name).data_ptr(), output.field_valid[name].data_ptr()) for name in fields] == pointers


@pytest.mark.parametrize('failure', ['mask', 'time', 'missing', 'nonfinite', 'increment_reset'])
def test_sensing_boundary_rejects_wrong_mask_structure_or_semantics(failure):
    case = make_sensing_case(('measurement', 'fim'))
    class Broken(case.implementation):
        def result(self, inputs, reset):
            result = super().result(inputs, reset)
            output = result.output
            if failure == 'increment_reset':
                output.field_valid['fim'].fill_(True)
            elif not reset:
                if failure == 'mask':
                    output.field_valid['fim'] = torch.ones(2, 4, 2, 2, dtype=torch.bool)
                elif failure == 'time':
                    output.time_semantics['fim'] = 'snapshot'
                elif failure == 'missing':
                    output = output._replace(measurement=None)
                else:
                    output.fim[0, 0, 0, 1] = float('nan')
            return result._replace(output=output)
    case.kwargs['model_type'] = Broken
    with pytest.raises(RuntimeError, match='model_a.*output'):
        instance = create_instance(case)
        instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
        instance.evaluate(case.step_inputs)


def test_inputs_are_projected_and_read_only_in_each_phase():
    case = make_radio_case()
    observed = []
    class Observer(case.implementation):
        def result(self, inputs, *, reset):
            observed.append((reset, set(inputs.geometry), inputs.sim_time.clone()))
            with pytest.raises(TypeError):
                inputs.geometry['new'] = inputs.geometry['distance']
            assert not torch.is_grad_enabled()
            return super().result(inputs, reset=reset)
    case.kwargs['model_type'] = Observer
    case.reset_inputs.geometry['undeclared'] = torch.zeros(1)
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    instance.evaluate(case.step_inputs)
    assert [(reset, keys) for reset, keys, time in observed] == [(True, {'distance'}), (False, {'distance'})]
    assert observed[0][2].tolist() == [0, 0]
    assert observed[1][2].tolist() == [.2, .4]


@pytest.mark.parametrize('failure', ['missing', 'units', 'axis', 'bound'])
def test_invalid_host_binding_is_rejected_before_model_execution(failure):
    case = make_radio_case()
    if failure == 'missing':
        case.kwargs['available_inputs'] = case.kwargs['available_inputs'][1:]
    elif failure == 'units':
        first, *rest = case.kwargs['available_inputs']
        case.kwargs['available_inputs'] = (first.model_copy(update={'tensor': first.tensor.model_copy(update={'units': 'cm'})}), *rest)
    elif failure == 'axis':
        case.kwargs['layout'] = {'instance_id': 'model_a', 'axes': {'B': 2, 'L': 3}}
    else:
        case.kwargs['bound_slots'] = torch.ones(2, 3, 5)
    with pytest.raises((ValueError, RuntimeError), match='model_a|layout'):
        create_instance(case)


def test_reset_tensor_shape_is_checked_at_explicit_reset_boundary():
    case = make_radio_case()
    instance = create_instance(case)
    case.reset_inputs.geometry['distance'] = torch.ones(2, 5, 3)
    with pytest.raises(RuntimeError, match='inputs.*distance.*shape'):
        instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)


def test_reset_only_fields_do_not_leak_into_evaluate():
    case = make_radio_case()
    item = InputField.model_validate(dict(path=['network', 'geometry', 'initial_only'], phase='reset',
        tensor=dict(shape=['B'], dtype='float32', units='m', frame='none', rate='radio_epoch')))
    descriptor = case.kwargs['descriptor']
    case.kwargs['descriptor'] = descriptor.model_copy(update={'input_fields': [*descriptor.input_fields, item]})
    concrete = item.model_copy(update={'tensor': item.tensor.model_copy(update={'shape': [2]})})
    case.kwargs['available_inputs'] += (concrete,)
    case.reset_inputs.geometry['initial_only'] = torch.ones(2)
    phases = []
    class Observer(case.implementation):
        def result(self, inputs, *, reset):
            phases.append(set(inputs.geometry))
            return super().result(inputs, reset=reset)
    case.kwargs['model_type'] = Observer
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    instance.evaluate(case.step_inputs)
    assert phases == [{'distance', 'initial_only'}, {'distance'}]


@pytest.mark.parametrize('supplied', [True, False])
def test_sensing_radio_dependency_receives_original_complete_block(supplied):
    case = make_sensing_case()
    item = InputField.model_validate(dict(path=['network', 'radio', 'link', 'received_power_w'],
        phase='post_physics', tensor=dict(shape=['B', 'L', 'F'], dtype='float32', units='W',
            frame='none', rate='radio_epoch')))
    descriptor = case.kwargs['descriptor']
    case.kwargs['descriptor'] = descriptor.model_copy(update={'input_fields': [*descriptor.input_fields, item]})
    case.kwargs['available_inputs'] += (item.model_copy(update={'tensor': item.tensor.model_copy(update={'shape': [2, 3, 5]})}),)
    radio = RadioEvaluationResult(LinkMetrics(torch.ones(2, 3, 5), torch.full((2, 3, 5), 7.),
        torch.ones(2, 3, 5, dtype=torch.bool)), None, None)
    observed = []
    class Observer(case.implementation):
        def result(self, inputs, reset):
            observed.append(inputs.radio_outputs)
            return super().result(inputs, reset)
    case.kwargs['model_type'] = Observer
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    if supplied:
        instance.evaluate(case.step_inputs._replace(radio_outputs=radio))
        assert observed[0] is None
        assert observed[1] is radio
        assert observed[1].link.path_gain_linear.shape == (2, 3, 5)
    else:
        with pytest.raises(RuntimeError, match='radio_outputs'):
            instance.evaluate(case.step_inputs)


def test_sensing_without_radio_dependency_does_not_receive_supplied_output():
    case = make_sensing_case()
    class Observer(case.implementation):
        def result(self, inputs, reset):
            assert inputs.radio_outputs is None
            return super().result(inputs, reset)
    case.kwargs['model_type'] = Observer
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    instance.evaluate(case.step_inputs._replace(radio_outputs=RadioEvaluationResult(None, None, None)))


@pytest.mark.parametrize('semantics', ['snapshot', 'increment'])
def test_sensing_reset_uses_each_fields_declared_time_semantics(semantics):
    case = make_sensing_case(('measurement', 'fim'), semantics={'measurement': semantics, 'fim': semantics})
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    assert all(bool(mask.all()) == (semantics == 'snapshot') for mask in instance.output.field_valid.values())
    assert all(mask.all() for mask in instance.evaluate(case.step_inputs).output.field_valid.values())


def test_interval_sample_model_reset_is_invalid_and_subset_only():
    case = make_sensing_case(('measurement', 'fim'),
        semantics={'measurement': 'interval_sample', 'fim': 'increment'})
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
    assert not instance.output.field_valid['measurement'].any()
    instance.evaluate(case.step_inputs)
    instance.reset(ModelResetMask(torch.tensor([True, False])), case.reset_inputs)
    assert not instance.output.field_valid['measurement'][0].any()
    assert instance.output.field_valid['measurement'][1].all()


def test_reset_cannot_publish_valid_interval_sample():
    case = make_sensing_case(semantics={'measurement': 'interval_sample'})

    class Broken(case.implementation):
        def reset_state(self, state, reset_mask, inputs, rng):
            result = super().reset_state(state, reset_mask, inputs, rng)
            result.output.field_valid['measurement'].fill_(True)
            return result

    case.kwargs['model_type'] = Broken
    instance = create_instance(case)
    with pytest.raises(RuntimeError, match='Reset.*interval_sample'):
        instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), case.reset_inputs)
