"""在真实CUDA执行模型边界，不启动物理/网络场景。"""
import itertools

import pytest
import torch

from model_helpers import create_instance, make_radio_case, make_sensing_case, make_stateful_case
from muse_sdk.models import ModelResetMask


@pytest.mark.parametrize('blocks', [c for n in (1, 2, 3)
    for c in itertools.combinations(('link', 'interference', 'rate'), n)])
def test_radio_contract_on_cuda(blocks):
    assert torch.cuda.is_available(), 'M3.3 CUDA acceptance requires a CUDA device'
    case = make_radio_case(blocks, 'cuda:0')
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool, device='cuda:0')), case.reset_inputs)
    output = instance.evaluate(case.step_inputs).output
    for block in output:
        if block is not None:
            assert all(value.device == torch.device('cuda:0') for value in block)
            assert not block.valid[1, 2, 4]
    if output.rate is not None:
        assert output.rate.rate_bps[0, 0, 1] == 20


@pytest.mark.parametrize('fields', [('measurement',), ('fim',), ('measurement', 'fim')])
def test_sensing_contract_on_cuda(fields):
    case = make_sensing_case(fields, 'cuda:0')
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool, device='cuda:0')), case.reset_inputs)
    output = instance.evaluate(case.step_inputs).output
    if 'measurement' in fields:
        assert (output.measurement == -1).all()
    if 'fim' in fields:
        torch.testing.assert_close(output.fim[0, 0], torch.eye(2, dtype=torch.float64, device='cuda:0'))
        assert output.field_valid['fim'].all()


def test_cuda_subset_reset_keeps_unselected_rng_and_cache_address():
    case = make_stateful_case('cuda:0', randomized=True)
    control_case = make_stateful_case('cuda:0', randomized=True)
    instance, control = create_instance(case), create_instance(control_case)
    mask = ModelResetMask(torch.ones(2, dtype=torch.bool, device='cuda:0'))
    instance.reset(mask, case.reset_inputs)
    control.reset(mask, control_case.reset_inputs)
    pointer = instance.state['memory'].data_ptr()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        instance.evaluate(case.step_inputs)
        control.evaluate(control_case.step_inputs)
        instance.reset(ModelResetMask(torch.tensor([True, False], device='cuda:0')), case.reset_inputs)
        for _ in range(3):
            instance.evaluate(case.step_inputs)
            control.evaluate(control_case.step_inputs)
    stream.synchronize()
    torch.testing.assert_close(instance.state['memory'][1], control.state['memory'][1], rtol=0, atol=0)
    assert instance.state['memory'].data_ptr() == pointer


@pytest.mark.parametrize('failure', ['cpu_tensor', 'late_nonfinite'])
def test_cuda_rejects_model_error_without_committing(failure):
    case = make_stateful_case('cuda:0')
    class Broken(case.implementation):
        def evaluate(self, inputs, state, rng):
            result = super().evaluate(inputs, state, rng)
            if failure == 'cpu_tensor':
                result.diagnostics['calls'] = result.diagnostics['calls'].cpu()
            else:
                result.output.rate.rate_bps[0, 0, 0] = float('nan')
            return result
    case.kwargs['model_type'] = Broken
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool, device='cuda:0')), case.reset_inputs)
    with pytest.raises(RuntimeError, match='radio_a'):
        instance.evaluate(case.step_inputs)
    assert (instance.state['memory'] == 1).all()
    assert instance.sim_time.count_nonzero() == 0
    assert not instance.output.rate.valid.any()


@pytest.mark.parametrize('all_worlds', [False, True])
def test_cuda_reset_with_mask_borrowed_from_output_cache(all_worlds):
    case = make_stateful_case('cuda:0')
    instance = create_instance(case)
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool, device='cuda:0')), case.reset_inputs)
    instance.evaluate(case.step_inputs)
    pointer = instance.state['memory'].data_ptr()
    mask = instance.output.rate.valid[:, 0, 0] if all_worlds else instance.output.rate.valid[:, 2, 4]
    instance.reset(ModelResetMask(mask), case.reset_inputs)
    assert instance.state['memory'][:, 0, 0].tolist() == [1, 1 if all_worlds else 3]
    assert instance.diagnostics['calls'].tolist() == [0, 0 if all_worlds else 1]
    assert instance.sim_time.tolist() == [0, 0 if all_worlds else .4]
    assert instance.interval_s.tolist() == [0, 0 if all_worlds else .1]
    assert instance.state['memory'].data_ptr() == pointer
