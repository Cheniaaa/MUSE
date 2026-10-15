"""同一非对称矩阵的真实CUDA/CPU/scalar对照，包含P/E/F重排和未知源。"""
import json

import pytest
import torch

from muse_sdk.models import ModelResetMask
from radio_reference_helpers import radio_case, scalar_radio_reference
from scientific.test_radio_reference import config_for, instance_for, plain_inputs


@pytest.mark.parametrize('law', ['free_space_radio', 'indoor_office_radio'])
@pytest.mark.parametrize('permutation', [False, True])
def test_cuda_scalar_reference(tmp_path, gpu_stream, law, permutation):
    layout, inputs, bound = radio_case(device='cuda:0', permutation=permutation)
    model = instance_for(law, layout, bound)
    reference = scalar_radio_reference(law=law, config=config_for(law),
        geometry_layout=layout['geometry'], network_spec=layout['network'], inputs=plain_inputs(inputs))
    actual = model.evaluate(inputs).output
    cpu_layout, cpu_inputs, cpu_bound = radio_case(permutation=permutation)
    cpu = instance_for(law, cpu_layout, cpu_bound).evaluate(cpu_inputs).output
    errors = {}
    for name, block in actual._asdict().items():
        for field, value in block._asdict().items():
            target = torch.tensor(reference[name][field], dtype=torch.bool if field == 'valid' else torch.float64,
                                  device='cuda:0')
            observed = value if field == 'valid' else value.double()
            torch.testing.assert_close(observed, target, rtol=1e-5, atol=0)
            torch.testing.assert_close(value.cpu(), getattr(getattr(cpu, name), field), rtol=1e-5, atol=0)
            if field != 'valid':
                assert (value[target == 0] == 0).all()
                mask = target != 0
                errors[f'{name}.{field}'] = ((observed[mask]-target[mask]).abs()/target[mask].abs()).max().item()
    (tmp_path / f'cuda-{law}-permutation{permutation}.json').write_text(json.dumps(errors, indent=2))


@pytest.mark.parametrize('law', ['free_space_radio', 'indoor_office_radio'])
def test_cuda_invalid_sources_and_subset_reset(gpu_stream, law):
    layout, inputs, bound = radio_case(device='cuda:0')
    model = instance_for(law, layout, bound)
    model.evaluate(inputs)
    pointers = [value.data_ptr() for block in model.output for value in block]
    other = [value[1].clone() for block in model.output for value in block]
    inputs.geometry['valid'][0] = False
    inputs.geometry['distance_m'][0] = float('nan')
    model.reset(ModelResetMask(torch.tensor([True, False], device='cuda:0')),
                inputs._replace(interval_s=torch.zeros_like(inputs.interval_s)))
    assert not model.output.rate.valid[0].any()
    assert not model.output.link.valid[0].any()
    for value, expected in zip((v for block in model.output for v in block), other):
        torch.testing.assert_close(value[1], expected, rtol=0, atol=0)
    assert pointers == [value.data_ptr() for block in model.output for value in block]
    reference = scalar_radio_reference(law=law, config=config_for(law),
        geometry_layout=layout['geometry'], network_spec=layout['network'], inputs=plain_inputs(inputs))
    actual = model.evaluate(inputs).output
    for name, fields in reference.items():
        for field, values in fields.items():
            value = getattr(getattr(actual, name), field)
            target = torch.tensor(values, dtype=value.dtype, device='cuda:0')
            torch.testing.assert_close(value, target, rtol=1e-5, atol=0)
