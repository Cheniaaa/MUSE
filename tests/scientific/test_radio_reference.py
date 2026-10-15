"""真实ModelInstance对照独立math oracle；不用产品传播helper生成预期。"""
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from muse_sdk.config import export_config_schema
from muse_sdk.descriptor import build_descriptor
from muse_sdk.geometry import ResolvedGeometry
from muse_sdk.models import ExecutionContext, ModelResetMask
from muse_sdk.network import network_model_input_fields
from muse_sdk.registration import TrainingPackageDefinition
from muse_runtime.model_execution import ModelInstance
from muse_runtime.rng import WorldRng
from radio_reference_helpers import radio_case, scalar_radio_reference


CONFIG = {'receiver_noise_temperature_k': 290., 'receiver_noise_figure_db': 7.}
LAWS = ('free_space_radio', 'indoor_office_radio')


def config_for(law, **updates):
    return {**CONFIG, **({'minimum_distance_m': 1.} if law == LAWS[0] else {}), **updates}


def plain_inputs(inputs):
    return {name: {key: value.tolist() for key, value in view.items()}
            for name, view in inputs._asdict().items() if isinstance(view, dict)}


def instance_for(law, layout, bound_slots, config=None, rng=None):
    from muse_builtins.models import registrations
    implementation = registrations()[law]
    descriptor = build_descriptor(TrainingPackageDefinition({}, {}, {}, {law: implementation}, {}),
        {}, 'a' * 64).radio_models[law]
    geometry = ResolvedGeometry(layout=layout['geometry'], endpoint_epsilon_m=1e-4, blockers=())
    available = tuple(field for phase in ('reset', 'post_physics')
        for field in network_model_input_fields(layout['network'], geometry=geometry,
            batch_size=layout['axes']['B'], phase=phase))
    return ModelInstance(implementation, descriptor=descriptor, config=config or config_for(law),
        layout=layout, context=ExecutionContext(str(bound_slots.device)), available_inputs=available,
        rng=rng or WorldRng(73, 2, str(bound_slots.device)), bound_slots=bound_slots)


def evaluate(law, inputs=None, layout=None, bound_slots=None, config=None):
    if inputs is None:
        layout, inputs, bound_slots = radio_case()
    instance = instance_for(law, layout, bound_slots, config)
    return instance.evaluate(inputs).output


def assert_reference(law, layout, inputs, bound_slots, config=None):
    config = config or config_for(law)
    expected = scalar_radio_reference(law=law, config=config, geometry_layout=layout['geometry'],
        network_spec=layout['network'], inputs=plain_inputs(inputs))
    output = evaluate(law, inputs, layout, bound_slots, config)
    for block_name, fields in expected.items():
        block = getattr(output, block_name)
        for field, values in fields.items():
            actual = getattr(block, field)
            target = torch.tensor(values, dtype=torch.bool if field == 'valid' else torch.float64,
                                  device=actual.device)
            torch.testing.assert_close(actual if field == 'valid' else actual.double(), target,
                                       rtol=1e-5, atol=0)
            if field != 'valid':
                assert (actual[target == 0] == 0).all()
    return output


def test_scalar_fixed_anchors():
    layout, inputs, _ = radio_case()
    inputs.simultaneous_transmissions['active'].zero_()
    inputs.allocations['link_power_w'].fill_(1.)
    inputs.allocations['link_active'].fill_(True)
    for law, los, loss in ((LAWS[0], True, 60.052008056),
                           (LAWS[1], True, 57.304224834), (LAWS[1], False, 65.067259919)):
        inputs.geometry['los'][0, 0] = los
        result = scalar_radio_reference(law=law, config=config_for(law),
            geometry_layout=layout['geometry'], network_spec=layout['network'], inputs=plain_inputs(inputs))
        gain = result['link']['path_gain_linear'][0][0][0]
        assert -10. * math.log10(gain) == pytest.approx(loss, abs=5e-10)
        sinr = result['interference']['sinr_linear'][0][0][0]
        assert gain / sinr == pytest.approx(1.0033472967343766e-13, rel=1e-14)


@pytest.mark.parametrize('law', LAWS)
def test_scalar_anchors_and_batch_outputs(law):
    layout, inputs, bound = radio_case()
    assert_reference(law, layout, inputs, bound)
    inputs.simultaneous_transmissions['active'].zero_()
    for los, loss in ((True, 60.052008056),) if law == LAWS[0] else (
            (True, 57.304224834), (False, 65.067259919)):
        inputs.geometry['los'][0, 0] = los
        output = assert_reference(law, layout, inputs, bound)
        assert -10. * math.log10(output.link.path_gain_linear[0, 0, 0].item()) == pytest.approx(loss, abs=1e-6)
        observed_noise = (output.link.received_power_w[0, 0, 0]
                          / output.interference.sinr_linear[0, 0, 0]).item()
        assert observed_noise == pytest.approx(1.0033472967343766e-13, rel=1e-6)


@pytest.mark.parametrize('law', LAWS)
def test_scaling_noise_and_model_domains(law):
    layout, inputs, bound = radio_case()
    inputs.simultaneous_transmissions['active'].zero_()
    baseline = assert_reference(law, layout, inputs, bound)
    signal = baseline.link.received_power_w[0, 0, 0].item()
    noise = signal / baseline.interference.sinr_linear[0, 0, 0].item()
    for update in ({'receiver_noise_temperature_k': 580.},
                   {'receiver_noise_figure_db': 7. + 10. * math.log10(2.)}):
        output = assert_reference(law, layout, inputs, bound, config_for(law, **update))
        assert signal / output.interference.sinr_linear[0, 0, 0].item() == pytest.approx(2. * noise, rel=1e-6)
    inputs.resources['bandwidth_hz'] *= 2.
    output = assert_reference(law, layout, inputs, bound)
    assert signal / output.interference.sinr_linear[0, 0, 0].item() == pytest.approx(2. * noise, rel=1e-6)
    assert output.rate.rate_bps[0, 0, 0] > baseline.rate.rate_bps[0, 0, 0]
    inputs.geometry['distance_m'][0, 0] *= 2.
    doubled = assert_reference(law, layout, inputs, bound)
    if law == LAWS[0]:
        assert doubled.link.path_gain_linear[0, 0, 0] == pytest.approx(
            baseline.link.path_gain_linear[0, 0, 0].item() / 4., rel=1e-6)
        inputs.geometry['distance_m'][0, 0] /= 2.
        inputs.resources['frequency_hz'] *= 2.
        assert assert_reference(law, layout, inputs, bound).link.path_gain_linear[0, 0, 0] == pytest.approx(
            baseline.link.path_gain_linear[0, 0, 0].item() / 4., rel=1e-6)
        inputs.geometry.pop('los')
        inputs.geometry['distance_m'][0, 0] = 2.
        assert not assert_reference(law, layout, inputs, bound, config_for(law, minimum_distance_m=3.)).link.valid[0, 0, 0]
        assert assert_reference(law, layout, inputs, bound, config_for(law, minimum_distance_m=2.)).link.valid[0, 0, 0]
    else:
        for distance, valid in ((1., True), (150., True),
                (torch.nextafter(torch.tensor(1.), torch.tensor(0.)).item(), False),
                (torch.nextafter(torch.tensor(150.), torch.tensor(float('inf'))).item(), False)):
            inputs.geometry['distance_m'][0, 0] = distance
            assert bool(assert_reference(law, layout, inputs, bound).link.valid[0, 0, 0]) == valid
        for los in (True, False):
            gains = []
            inputs.geometry['los'][0, 0] = los
            for distance in (1., 10., 150.):
                inputs.geometry['distance_m'][0, 0] = distance
                gains.append(assert_reference(law, layout, inputs, bound).link.path_gain_linear[0, 0, 0].item())
            assert gains[0] > gains[1] > gains[2]
        inputs.geometry['los'][0, 0] = True
        los_gain = assert_reference(law, layout, inputs, bound).link.path_gain_linear[0, 0, 0]
        assert gains[-1] <= los_gain


@pytest.mark.parametrize('law', LAWS)
def test_permuted_endpoint_path_resource_mapping(law):
    layout, inputs, bound = radio_case()
    original = assert_reference(law, layout, inputs, bound)
    permuted = assert_reference(law, *radio_case(permutation=True))
    for old_block, new_block in zip(original, permuted):
        for old, new in zip(old_block, new_block):
            torch.testing.assert_close(new, old[:, :, [2, 0, 1]], rtol=0, atol=0)
    # 同P的不同L只在自己的候选resource有效，宿主bound全True不能代替候选。
    assert layout['geometry'].desired[0].path == layout['geometry'].desired[2].path
    assert original.link.valid[0, 0].tolist() == [True, False, True]
    assert original.link.valid[0, 2].tolist() == [False, True, False]
    inputs.simultaneous_transmissions['power_w'][0, 2, 0] *= 3.
    changed = assert_reference(law, layout, inputs, bound)
    assert changed.interference.interference_power_w[0, 0, 0] > original.interference.interference_power_w[0, 0, 0]
    for old_block, new_block in zip(original, changed):
        for old, new in zip(old_block, new_block):
            torch.testing.assert_close(old[1], new[1], rtol=0, atol=0)


@pytest.mark.parametrize('law', LAWS)
def test_unknown_interferer_is_not_silently_zero(law):
    layout, inputs, bound = radio_case()
    baseline = assert_reference(law, layout, inputs, bound)
    inputs.geometry['valid'][0, 1] = False
    unknown = assert_reference(law, layout, inputs, bound)
    assert unknown.link.valid[0, 0, 0] and not unknown.interference.valid[0, 0, 0]
    assert not unknown.rate.valid[0, 0, 0]
    inputs.simultaneous_transmissions['active'][0, 2, 0] = False
    assert assert_reference(law, layout, inputs, bound).rate.valid[0, 0, 0]
    inputs.simultaneous_transmissions['active'][0, 2, 0] = True
    saved_power = inputs.simultaneous_transmissions['power_w'][0, 2, 0].item()
    inputs.simultaneous_transmissions['power_w'][0, 2, 0] = 0.
    assert assert_reference(law, layout, inputs, bound).rate.valid[0, 0, 0]
    inputs.simultaneous_transmissions['power_w'][0, 2, 0] = saved_power
    inputs.simultaneous_transmissions['active'][0, 2, 0] = False
    # 仍在另一resource实际发射，未知路径不污染F0。
    inputs.simultaneous_transmissions['active'][0, 2, 1] = True
    assert assert_reference(law, layout, inputs, bound).rate.valid[0, 0, 0]
    inputs.geometry['valid'][0, 1] = True
    inputs.simultaneous_transmissions['active'][0, 2, 0] = True
    saved_distance = inputs.geometry['distance_m'][0, 1].item()
    for outside in (0.,) if law == LAWS[0] else (0., 151.):
        inputs.geometry['distance_m'][0, 1] = outside
        outside_domain = assert_reference(law, layout, inputs, bound)
        assert outside_domain.link.valid[0, 0, 0] and not outside_domain.rate.valid[0, 0, 0]
        inputs.simultaneous_transmissions['active'][0, 2, 0] = False
        assert assert_reference(law, layout, inputs, bound).rate.valid[0, 0, 0]
        inputs.simultaneous_transmissions['active'][0, 2, 0] = True
    inputs.geometry['distance_m'][0, 1] = saved_distance
    inputs.geometry['los'].fill_(True)
    clear = assert_reference(law, layout, inputs, bound)
    inputs.geometry['los'][0, 0] = False
    desired_blocked = assert_reference(law, layout, inputs, bound)
    inputs.geometry['los'][0, 0] = True
    inputs.geometry['los'][0, 1] = False
    source_blocked = assert_reference(law, layout, inputs, bound)
    if law == LAWS[1]:
        assert desired_blocked.interference.sinr_linear[0, 0, 0] < clear.interference.sinr_linear[0, 0, 0]
        assert source_blocked.interference.sinr_linear[0, 0, 0] > clear.interference.sinr_linear[0, 0, 0]
    else:
        torch.testing.assert_close(desired_blocked.rate.rate_bps, clear.rate.rate_bps, rtol=0, atol=0)
        torch.testing.assert_close(source_blocked.rate.rate_bps, clear.rate.rate_bps, rtol=0, atol=0)
    assert baseline.link.valid[0, 0, 0]


@pytest.mark.parametrize('law', LAWS)
def test_validity_table_and_half_duplex(law):
    layout, inputs, bound = radio_case()
    output = assert_reference(law, layout, inputs, bound)
    assert not output.link.valid[0, 0, 1] and output.link.path_gain_linear[0, 0, 1] == 0
    assert output.rate.valid[0, 0, 2] and output.rate.rate_bps[0, 0, 2] == 0
    inputs.allocations['link_active'][0, 0, 0] = False
    output = assert_reference(law, layout, inputs, bound)
    assert output.link.valid[0, 0, 0] and output.rate.valid[0, 0, 0]
    assert output.link.received_power_w[0, 0, 0] == output.interference.sinr_linear[0, 0, 0] == 0
    inputs.allocations['link_active'][0, 0, 0] = True
    inputs.allocations['link_power_w'][0, 0, 0] = 0.
    assert assert_reference(law, layout, inputs, bound).rate.valid[0, 0, 0]
    inputs.allocations['link_power_w'][0, 0, 0] = .001
    incoming = assert_reference(law, layout, inputs, bound).link.received_power_w[0, 0, 0].item()
    inputs.simultaneous_transmissions['active'][0, 1, 0] = True
    inputs.simultaneous_transmissions['power_w'][0, 1, 0] = .001
    output = assert_reference(law, layout, inputs, bound)
    assert output.link.valid[0, 0, 0] and not output.interference.valid[0, 0, 0] and not output.rate.valid[0, 0, 0]
    assert output.link.received_power_w[0, 0, 0] == incoming > 0
    inputs.simultaneous_transmissions['power_w'][0, 1, 0] = 0.
    assert assert_reference(law, layout, inputs, bound).rate.valid[0, 0, 0]
    inputs.simultaneous_transmissions['power_w'][0, 1, 1] = .001
    inputs.simultaneous_transmissions['active'][0, 1, 1] = True
    assert assert_reference(law, layout, inputs, bound).rate.valid[0, 0, 0]
    for endpoint in (0, 1):
        inputs.endpoint_masks['present'][0, endpoint] = False
        output = assert_reference(law, layout, inputs, bound)
        assert not output.link.valid[0, 0, 0] and not output.interference.valid[0, 0, 0] and not output.rate.valid[0, 0, 0]
        inputs.endpoint_masks['present'][0, endpoint] = True
    inputs.geometry['valid'][0, 0] = False
    output = assert_reference(law, layout, inputs, bound)
    assert not output.link.valid[0, 0, 0] and not output.rate.valid[0, 0, 0]


@pytest.mark.parametrize('law', LAWS)
def test_invalid_distance_is_not_repaired_and_numeric_errors_surface(law):
    layout, inputs, bound = radio_case()
    for value in (0., float('nan')):
        inputs.geometry['valid'][0, 1] = False
        inputs.geometry['distance_m'][0, 1] = value
        inputs.simultaneous_transmissions['active'][0, 2, 0] = False
        output = assert_reference(law, layout, inputs, bound)
        assert output.rate.valid[0, 0, 0] and torch.isfinite(output.rate.rate_bps).all()
        inputs.simultaneous_transmissions['active'][0, 2, 0] = True
        assert not assert_reference(law, layout, inputs, bound).rate.valid[0, 0, 0]
    inputs.geometry['valid'][0, 0] = True
    inputs.geometry['distance_m'][0, 0] = float('nan')
    with pytest.raises(RuntimeError, match='finite|numeric|distance'):
        evaluate(law, inputs, layout, bound)
    layout, inputs, bound = radio_case()
    with pytest.raises((RuntimeError, OverflowError), match='noise|overflow|range'):
        evaluate(law, inputs, layout, bound, config_for(law, receiver_noise_figure_db=4000.))
    # 噪声因子本身有限，但乘积溢出不能经过SINR变成有效零rate。
    with pytest.raises(RuntimeError, match='noise'):
        evaluate(law, inputs, layout, bound, config_for(law, receiver_noise_temperature_k=1e308,
            receiver_noise_figure_db=1000.))
    # 最终float32有效输出溢出必须由宿主既有边界拒绝。
    inputs.resources['bandwidth_hz'].fill_(1e39)
    inputs.allocations['link_power_w'].fill_(3e38)
    with pytest.raises(RuntimeError, match='output.*finite'):
        evaluate(law, inputs, layout, bound)
    if law == LAWS[0]:
        layout, inputs, bound = radio_case()
        inputs.resources['frequency_hz'].fill_(1e-200)
        with pytest.raises(RuntimeError, match='output.link.*finite'):
            evaluate(law, inputs, layout, bound)


@pytest.mark.parametrize('law', LAWS)
def test_reset_cache_state_and_rng(law):
    layout, inputs, bound = radio_case()
    rng, untouched = WorldRng(73, 2, 'cpu'), WorldRng(73, 2, 'cpu')
    global_rng = torch.get_rng_state().clone()
    instance = instance_for(law, layout, bound, rng=rng)
    leaves = lambda: [*(value for block in instance.output for value in block), instance.sim_time, instance.interval_s]
    pointers = [value.data_ptr() for value in leaves()]
    reset_inputs = inputs._replace(interval_s=torch.zeros_like(inputs.interval_s))
    instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), reset_inputs)
    expected = evaluate(law, inputs, layout, bound)
    for actual, target in zip(instance.output.link, expected.link):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    for actual, target in zip(instance.output.interference, expected.interference):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    assert not instance.output.rate.valid.any() and not instance.output.rate.rate_bps.any()
    instance.evaluate(inputs._replace(sim_time=torch.tensor([1., 2.], dtype=torch.float64)))
    saved = [value[1].clone() for value in leaves()]
    inputs.geometry['distance_m'][0, 0] = 30.
    instance.reset(ModelResetMask(torch.tensor([True, False])), reset_inputs)
    assert not instance.output.rate.valid[0].any() and not instance.output.rate.rate_bps[0].any()
    changed = evaluate(law, inputs, layout, bound)
    for actual_block, expected_block in ((instance.output.link, changed.link),
                                          (instance.output.interference, changed.interference)):
        for actual, target in zip(actual_block, expected_block):
            torch.testing.assert_close(actual[0], target[0], rtol=0, atol=0)
    for actual, target in zip(leaves(), saved):
        torch.testing.assert_close(actual[1], target, rtol=0, atol=0)
    assert [value.data_ptr() for value in leaves()] == pointers
    assert instance.state == () and instance.diagnostics == {}
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    torch.testing.assert_close(rng.uniform(range(2), 'reference', 'probe', (3,)),
        untouched.uniform(range(2), 'reference', 'probe', (3,)), rtol=0, atol=0)


def test_lightweight_describe_and_metadata_registration():
    code = '''
import sys
from muse_builtins.models import registrations
for model in registrations().values():
    model.describe()
assert 'torch' not in sys.modules
assert 'warp' not in sys.modules
from pathlib import Path
from muse_builtins.metadata import export_metadata
metadata = export_metadata(Path('examples/basic_flight/controller.json'))
assert set(metadata.descriptor.radio_models) == set(registrations())
assert 'torch' not in sys.modules
assert 'warp' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
    from muse_builtins.metadata import export_metadata
    from muse_builtins.models import registrations
    metadata = export_metadata(Path('examples/basic_flight/controller.json'))
    assert set(metadata.descriptor.radio_models) == set(LAWS)
    for law, implementation in registrations().items():
        descriptor = metadata.descriptor.radio_models[law]
        schema = export_config_schema(implementation.describe()['config_schema'])
        assert descriptor.config_schema == schema
        assert set(schema['required']) == set(config_for(law))
        assert all('default' not in value for value in schema['properties'].values())
        by_phase = [{tuple(field.path) for field in descriptor.input_fields if field.phase == phase}
                    for phase in ('reset', 'post_physics')]
        assert by_phase[0] == by_phase[1]
        assert (('network', 'geometry', 'los') in by_phase[0]) == (law == LAWS[1])
    assert metadata.descriptor.radio_models[LAWS[1]].frequency_hz_schema == {
        'type': 'number', 'exclusiveMinimum': 5e8, 'exclusiveMaximum': 1e11}
