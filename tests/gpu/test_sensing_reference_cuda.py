"""CUDA确定性EFIM对照CPU/math；Gaussian样本按设备验证统计与重置隔离。"""
import json
import math

import pytest
import torch

from muse_runtime.rng import WorldRng
from muse_sdk.models import ModelResetMask
from sensing_reference_helpers import sensing_case, sensing_instance, scalar_sensing_reference


CONFIG = dict(receiver_noise_temperature_k=290., receiver_noise_figure_db=7.,
              target_rcs_m2=1., minimum_distance_m=1.)
MODELS = ('free_space_path_length_measurement', 'free_space_position_efim_2d',
          'free_space_path_length_efim_2d')


def scalar_inputs(inputs):
    plain = {name: {key: value.cpu().tolist() for key, value in getattr(inputs, name).items()}
             for name in ('geometry', 'endpoint_masks', 'resources', 'allocations', 'simultaneous_transmissions')}
    return dict(plain, sim_time=inputs.sim_time.tolist(), interval_s=inputs.interval_s.tolist())


@pytest.mark.parametrize('permutation', [False, True])
def test_cuda_capabilities_and_scalar_information(tmp_path, gpu_stream, permutation):
    layout, inputs = sensing_case(device='cuda:0', permutation=permutation)
    reference = scalar_sensing_reference(config=CONFIG, geometry_layout=layout['geometry'],
        network_spec=layout['network'], inputs=scalar_inputs(inputs))
    cpu_layout, cpu_inputs = sensing_case(permutation=permutation)
    outputs, errors = {}, {}
    for name in MODELS:
        model = sensing_instance(name, layout, config=CONFIG, device='cuda:0',
                                 rng=WorldRng(7, layout['axes']['B'], 'cuda:0'))
        outputs[name] = model.evaluate(inputs).output
        cpu = sensing_instance(name, cpu_layout, config=CONFIG, device='cpu',
            rng=WorldRng(7, cpu_layout['axes']['B'], 'cpu')).evaluate(cpu_inputs).output
        for field, valid in outputs[name].field_valid.items():
            torch.testing.assert_close(valid.cpu(), cpu.field_valid[field], rtol=0, atol=0)
            assert torch.isfinite(getattr(outputs[name], field)).all()
        if outputs[name].fim is not None:
            expected = torch.tensor(reference['fim'], dtype=torch.float64, device='cuda:0')
            error = torch.linalg.matrix_norm(outputs[name].fim-expected)
            scale = torch.linalg.matrix_norm(expected).clamp_min(1.)
            assert (error <= 1e-11*scale).all()
            assert (outputs[name].fim[expected == 0] == 0).all()
            cross_device_error = torch.linalg.matrix_norm(outputs[name].fim.cpu()-cpu.fim)
            assert (cross_device_error <= 1e-11*scale.cpu()).all()
            errors[name] = float((error/scale).max())
    assert outputs[MODELS[0]].fim is None
    assert outputs[MODELS[1]].measurement is None
    torch.testing.assert_close(outputs[MODELS[0]].measurement, outputs[MODELS[2]].measurement, rtol=0, atol=0)
    torch.testing.assert_close(outputs[MODELS[1]].fim, outputs[MODELS[2]].fim, rtol=0, atol=0)
    (tmp_path / 'cuda-scalar-errors.json').write_text(json.dumps(errors, indent=2))


def test_cuda_normalized_gaussian_statistics(tmp_path, gpu_stream):
    layout, inputs = sensing_case(device='cuda:0')
    reference = scalar_sensing_reference(config=CONFIG, geometry_layout=layout['geometry'],
        network_spec=layout['network'], inputs=scalar_inputs(inputs))
    model = sensing_instance(MODELS[0], layout, config=CONFIG, device='cuda:0',
                             rng=WorldRng(53, layout['axes']['B'], 'cuda:0'))
    mean = torch.tensor(reference['h'], dtype=torch.float64, device='cuda:0')
    variance = torch.tensor(reference['variance'], dtype=torch.float64, device='cuda:0')
    n, x = 4096, 20
    first = model.evaluate(inputs).output
    valid = first.field_valid['measurement'].clone()
    assert valid.any()
    draws = [first.measurement[valid].clone()]
    for _ in range(n-1):
        draws.append(model.evaluate(inputs).output.measurement[valid].clone())
    residuals = (torch.stack(draws)-mean[valid])/variance[valid].sqrt()
    observed_mean, observed_variance = residuals.mean(0), residuals.var(0)
    assert observed_mean.abs().le(math.sqrt(2*x/n)).all()
    assert observed_variance.ge(1-2*math.sqrt(x/(n-1))).all()
    assert observed_variance.le(1+2*math.sqrt(x/(n-1))+2*x/(n-1)).all()
    (tmp_path / 'cuda-noise-statistics.json').write_text(json.dumps(dict(samples=n,
        normalized_mean=observed_mean.tolist(), normalized_variance=observed_variance.tolist()), indent=2))


def test_cuda_subset_reset_keeps_other_world_future_measurements(gpu_stream):
    layout, inputs = sensing_case(device='cuda:0')
    models = [sensing_instance(MODELS[2], layout, config=CONFIG, device='cuda:0',
        rng=WorldRng(71, layout['axes']['B'], 'cuda:0')) for _ in range(2)]
    for model in models:
        model.evaluate(inputs)
    model, control = models
    pointers = (model.output.measurement.data_ptr(), model.output.fim.data_ptr())
    before = model.output.measurement[1].clone()
    model.reset(ModelResetMask(torch.tensor([True, False], device='cuda:0')),
                inputs._replace(interval_s=torch.zeros_like(inputs.interval_s)))
    assert not model.output.field_valid['measurement'][0].any()
    assert not model.output.field_valid['fim'][0].any()
    torch.testing.assert_close(model.output.measurement[1], before, rtol=0, atol=0)
    actual, expected = model.evaluate(inputs).output, control.evaluate(inputs).output
    torch.testing.assert_close(actual.measurement[1], expected.measurement[1], rtol=0, atol=0)
    torch.testing.assert_close(actual.fim[1], expected.fim[1], rtol=0, atol=0)
    assert pointers == (model.output.measurement.data_ptr(), model.output.fim.data_ptr())


def test_cuda_static_budget_and_actual_tail_sample(tmp_path, gpu_stream):
    layout, inputs = sensing_case(device='cuda:0')
    full = sensing_instance(MODELS[1], layout, config=CONFIG, device='cuda:0').evaluate(
        inputs._replace(interval_s=torch.full_like(inputs.interval_s, .04))).output.fim
    tails = {}
    for k in (7, 3, 1):
        rng = WorldRng(89, layout['axes']['B'], 'cuda:0')
        expected_rng = WorldRng(89, layout['axes']['B'], 'cuda:0')
        model = sensing_instance(MODELS[2], layout, config=CONFIG, device='cuda:0', rng=rng)
        total = torch.zeros_like(full)
        for left in range(0, 20, k):
            length = min(k, 20-left)
            current = inputs._replace(interval_s=torch.full_like(inputs.interval_s, length*.002))
            output = model.evaluate(current).output
            total += output.fim
            noise = torch.stack([torch.randn(output.measurement.shape[1:], dtype=torch.float64,
                device='cuda:0', generator=expected_rng.generator(w, layout['instance_id'], 'measurement'))
                for w in range(layout['axes']['B'])])
        expected = scalar_sensing_reference(config=CONFIG, geometry_layout=layout['geometry'],
            network_spec=layout['network'], inputs=scalar_inputs(current))
        mask = torch.tensor(expected['measurement_valid'], device='cuda:0')
        variance = torch.tensor(expected['variance'], dtype=torch.float64, device='cuda:0')
        measurement = torch.where(mask, torch.tensor(expected['h'], dtype=torch.float64, device='cuda:0')
                                  + variance.sqrt()*noise, 0.)
        torch.testing.assert_close(output.measurement, measurement, rtol=1e-11, atol=0)
        assert (torch.linalg.matrix_norm(total-full) <= 1e-11*torch.linalg.matrix_norm(full).clamp_min(1.)).all()
        tails[k] = dict(ticks=length, variance=variance[mask].tolist())
    assert [tails[k]['ticks'] for k in (7, 3, 1)] == [6, 2, 1]
    torch.testing.assert_close(torch.tensor(tails[1]['variance'], dtype=torch.float64),
                              6*torch.tensor(tails[7]['variance'], dtype=torch.float64), rtol=1e-11, atol=0)
    (tmp_path / 'static-tail-budget.json').write_text(json.dumps(tails, indent=2))
