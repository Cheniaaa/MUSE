"""科学law的真实ModelInstance、独立math/Schur/统计CPU验收。"""
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from muse_sdk.models import ModelResetMask
from muse_runtime.rng import WorldRng
from sensing_reference_helpers import sensing_case, sensing_instance, scalar_sensing_reference


CONFIG = {'receiver_noise_temperature_k': 290., 'receiver_noise_figure_db': 7., 'target_rcs_m2': 1., 'minimum_distance_m': 1.}
LAWS = ('free_space_path_length_measurement', 'free_space_position_efim_2d', 'free_space_path_length_efim_2d')


def reference(layout, inputs, config=CONFIG):
    return scalar_sensing_reference(config=config, geometry_layout=layout['geometry'], network_spec=layout['network'], inputs=inputs)


def evaluate(layout, inputs, law=LAWS[2], config=CONFIG, rng=None):
    return sensing_instance(law, layout, config=config, rng=rng).evaluate(inputs).output


def assert_fim(actual, expected):
    target = torch.as_tensor(expected, dtype=torch.float64, device=actual.device)
    error = torch.linalg.matrix_norm(actual - target)
    assert (error <= 1e-11 * torch.linalg.matrix_norm(target).clamp_min(1)).all()
    assert (actual[target == 0] == 0).all()


def assert_reference(layout, inputs, config=CONFIG):
    expected, output = reference(layout, inputs, config), evaluate(layout, inputs, config=config)
    for field in ('measurement', 'fim'):
        assert torch.equal(output.field_valid[field], torch.tensor(expected[field + '_valid']))
    assert_fim(output.fim, expected['fim'])
    assert (output.measurement[~output.field_valid['measurement']] == 0).all()
    return output, expected


def test_scalar_fixed_anchors():
    # 设计§8字面量量级，而非oracle自证；额外全精度值由独立math标量表达核对。
    layout, inputs = sensing_case()
    inputs.simultaneous_transmissions['active'][:, 3] = False
    inputs = inputs._replace(interval_s=torch.full((2,), .012, dtype=torch.float64))
    expected = reference(layout, inputs)
    noise_psd = 1.380649e-23 * 290 * 10**.7
    echo = .001 * (299792458. / 2.412e9)**2 / ((4 * math.pi)**3 * 33**2)
    echo_b = .001 * (299792458. / 2.437e9)**2 / ((4 * math.pi)**3 * 33**2)
    variance = 299792458.**2 * noise_psd / (8 * math.pi**2 * (5e6**2 / 12) * echo * .012)
    variance_b = 299792458.**2 * noise_psd / (8 * math.pi**2 * (5e6**2 / 12) * echo_b * .012)
    assert expected['h'][0][0][0] == pytest.approx(2 * math.sqrt(33), rel=1e-6)
    assert expected['echo'][0][0][0] == pytest.approx(echo, rel=1e-6)
    assert expected['echo'][0][1][1] == pytest.approx(echo_b, rel=1e-6)
    assert expected['variance'][0][0][0] == pytest.approx(variance, rel=1e-6)
    assert expected['variance'][0][1][1] == pytest.approx(variance_b, rel=1e-6)
    assert expected['echo'][0][0][0] == pytest.approx(7.149e-12, rel=1e-4)
    assert expected['echo'][0][1][1] == pytest.approx(7.003e-12, rel=1e-4)
    assert math.sqrt(expected['variance'][0][0][0]) == pytest.approx(.01131, rel=5e-4)
    assert math.sqrt(expected['variance'][0][1][1]) == pytest.approx(.01142, rel=5e-4)
    budget_inputs = inputs._replace(interval_s=torch.full((2,), .04, dtype=torch.float64))
    quiet_budget = reference(layout, budget_inputs)
    quiet_j = [[sum(quiet_budget['fim'][0][s][f][i][j] for s in (0, 1) for f in range(3)) for j in (0, 1)] for i in (0, 1)]
    quiet_bound = math.sqrt((quiet_j[0][0] + quiet_j[1][1]) / (quiet_j[0][0] * quiet_j[1][1] - quiet_j[0][1]**2))
    nominal_bound = math.sqrt((variance + variance_b) * .012 / (2 * (64 / 33) * .04))
    assert quiet_bound == pytest.approx(nominal_bound, rel=1e-6)
    assert quiet_bound == pytest.approx(.004470, rel=5e-4)
    inputs.simultaneous_transmissions['active'][0, 3, 0] = True
    jammer_psd = .001 * (299792458. / (4 * math.pi * math.sqrt(80) * 2.412e9))**2 / 5e6
    jammed_variance = variance * (noise_psd + jammer_psd) / noise_psd
    jammed = reference(layout, inputs)
    assert jammed['variance'][0][0][0] == pytest.approx(jammed_variance, rel=1e-6)
    assert math.sqrt(jammed['variance'][0][0][0]) == pytest.approx(1.248, rel=5e-4)
    jammed_budget = reference(layout, budget_inputs)
    jammed_j = [[sum(jammed_budget['fim'][0][s][f][i][j] for s in (0, 1) for f in range(3)) for j in (0, 1)] for i in (0, 1)]
    jammed_bound = math.sqrt((jammed_j[0][0] + jammed_j[1][1]) / (jammed_j[0][0] * jammed_j[1][1] - jammed_j[0][1]**2))
    assert jammed_bound == pytest.approx(math.sqrt((jammed_variance + variance_b) * .012 / (2 * (64 / 33) * .04)), rel=1e-6)
    assert jammed_bound == pytest.approx(.3471, rel=5e-4)


@pytest.mark.parametrize('law', LAWS)
def test_scalar_and_model_three_capabilities(law):
    layout, inputs = sensing_case()
    output = evaluate(layout, inputs, law)
    expected = reference(layout, inputs)
    assert (output.measurement is None) == (law == LAWS[1])
    assert (output.fim is None) == (law == LAWS[0])
    assert output.time_semantics == ({'measurement': 'interval_sample'} if law == LAWS[0]
        else {'fim': 'increment'} if law == LAWS[1] else {'measurement': 'interval_sample', 'fim': 'increment'})
    for field in output.field_valid:
        assert torch.equal(output.field_valid[field], torch.tensor(expected[field + '_valid']))
        assert getattr(output, field).dtype == torch.float64
        assert getattr(output, field).shape == ((2, 4, 3) if field == 'measurement' else (2, 4, 3, 2, 2))
    if output.fim is not None:
        assert_fim(output.fim, expected['fim'])
        torch.testing.assert_close(output.fim, evaluate(layout, inputs, LAWS[2]).fim, rtol=0, atol=0)
    if output.measurement is not None:
        # 精确seed下scalar h/v重建，不依赖产品内部helper/diagnostics。
        rng = WorldRng(73, 2, 'cpu')
        noise = torch.stack([torch.randn((4, 3), dtype=torch.float64, generator=rng.generator(w, 'sensing_reference', 'measurement')) for w in range(2)])
        target = torch.tensor(expected['h'], dtype=torch.float64) + torch.sqrt(torch.tensor(expected['variance'], dtype=torch.float64)) * noise
        target = torch.where(output.field_valid['measurement'], target, 0.)
        torch.testing.assert_close(output.measurement, target, rtol=1e-11, atol=0)


def test_jacobian_and_nuisance_schur_reference():
    layout, inputs = sensing_case()
    # 整坐标、3-4-5距离，double有限差分避免float32不可表达的扰动。
    positions = inputs.geometry['positions_world']
    positions[0, 3] = torch.tensor([0., 0., 1.])
    positions[0, 1] = torch.tensor([3., 4., 1.])
    positions[0, 4] = torch.tensor([3., 4., 1.])
    positions[0, 5] = torch.tensor([6., 0., 1.])
    for p, path in enumerate(layout['geometry'].paths):
        inputs.geometry['distance_m'][:, p] = torch.linalg.vector_norm(positions[:, path.tx].double() - positions[:, path.rx].double(), dim=-1)
    output, expected = assert_reference(layout, inputs)
    s, f = 2, 0
    q, t, r = (positions[0, p].double().tolist() for p in (4, 3, 5))
    h = lambda xy: math.dist([*xy, q[2]], t) + math.dist([*xy, q[2]], r)
    delta = 1e-4
    gradient = [(h([q[0] + delta, q[1]]) - h([q[0] - delta, q[1]])) / (2 * delta),
                (h([q[0], q[1] + delta]) - h([q[0], q[1] - delta])) / (2 * delta)]
    torch.testing.assert_close(torch.tensor(gradient), torch.tensor(expected['gradient'][0][s][f]), rtol=1e-7, atol=1e-8)
    variance = expected['variance'][0][s][f]
    # E[ε²]=1, E[ε³]=0, E[(ε²-1)²]=2；mean-score/log(a)-score正交。
    joint = torch.zeros((3, 3), dtype=torch.float64)
    g = torch.tensor(expected['gradient'][0][s][f], dtype=torch.float64)
    joint[:2, :2] = torch.outer(g, g) / variance
    joint[2, 2] = .5
    schur = joint[:2, :2] - joint[:2, 2:] @ torch.linalg.solve(joint[2:, 2:], joint[2:, :2])
    assert_fim(output.fim[0, s, f], schur)
    # 单站沿r的Gaussian族：已知a=Kr^4多出8/r²，而每样本自由nuisance不含它。
    mono = expected['fim'][0][0][0]
    r0 = inputs.geometry['distance_m'][0, layout['geometry'].sensing[0].outbound].item()
    radial = positions[0, 1, :2].double() - positions[0, 3, :2].double()
    radial /= torch.linalg.vector_norm(radial)
    radial_g = sum(expected['gradient'][0][0][0][i] * radial[i].item() for i in (0, 1))
    actual = (radial @ torch.tensor(mono, dtype=torch.float64) @ radial).item()
    assert actual == pytest.approx(radial_g**2 / expected['variance'][0][0][0], rel=1e-11)
    assert radial_g == pytest.approx(2., rel=1e-11)
    k = expected['a'][0][0][0] / r0**4
    known = 4 * inputs.interval_s[0].item() / (k * r0**4) + 8 / r0**2
    assert actual == pytest.approx(4 * inputs.interval_s[0].item() / (k * r0**4), rel=1e-11)
    assert known > actual and known != pytest.approx(actual, rel=1e-5)


def test_information_scales_with_actual_energy():
    layout, inputs = sensing_case()
    original, ref = assert_reference(layout, inputs)
    doubled = inputs._replace(interval_s=2 * inputs.interval_s)
    output, _ = assert_reference(layout, doubled)
    torch.testing.assert_close(output.fim, 2 * original.fim, rtol=1e-11, atol=0)
    h, mask = torch.tensor(ref['h'], dtype=torch.float64), original.field_valid['measurement']
    torch.testing.assert_close((output.measurement - h)[mask], ((original.measurement - h) / math.sqrt(2))[mask], rtol=1e-11, atol=1e-14)
    budget = evaluate(layout, inputs._replace(interval_s=torch.full((2,), .04, dtype=torch.float64))).fim
    parts = [evaluate(layout, inputs._replace(interval_s=torch.full((2,), t, dtype=torch.float64))).fim for t in (.014, .014, .012)]
    torch.testing.assert_close(sum(parts), budget, rtol=1e-11, atol=0)
    # 三个物理标度分别由独立oracle核对，不把名义variance变化当附加FIM。
    for update, factor in (({'target_rcs_m2': 2.}, 2.),):
        changed, _ = assert_reference(layout, inputs, {**CONFIG, **update})
        torch.testing.assert_close(changed.fim, factor * original.fim, rtol=1e-11, atol=0)
    inputs.simultaneous_transmissions['active'][:, 3] = False
    quiet, _ = assert_reference(layout, inputs)
    hotter, _ = assert_reference(layout, inputs, {**CONFIG, 'receiver_noise_temperature_k': 580.})
    # S0/F0不受其他resource发射干扰。
    torch.testing.assert_close(hotter.fim[0, 0, 0], quiet.fim[0, 0, 0] / 2, rtol=1e-11, atol=0)
    noisier, _ = assert_reference(layout, inputs, {**CONFIG, 'receiver_noise_figure_db': 7 + 10 * math.log10(2)})
    torch.testing.assert_close(noisier.fim[0, 0, 0], quiet.fim[0, 0, 0] / 2, rtol=1e-11, atol=0)
    inputs.simultaneous_transmissions['power_w'][:, 0] *= 2
    stronger, _ = assert_reference(layout, inputs)
    torch.testing.assert_close(stronger.fim[0, 0, 0], quiet.fim[0, 0, 0] * 2, rtol=1e-11, atol=0)
    inputs.simultaneous_transmissions['power_w'][:, 0] /= 2
    inputs.resources['frequency_hz'] *= 2
    higher, _ = assert_reference(layout, inputs)
    torch.testing.assert_close(higher.fim[0, 0, 0], quiet.fim[0, 0, 0] / 4, rtol=1e-11, atol=0)
    inputs.resources['bandwidth_hz'] *= 2
    wider, _ = assert_reference(layout, inputs)
    torch.testing.assert_close(wider.fim[0, 0, 0], higher.fim[0, 0, 0] * 4, rtol=1e-11, atol=0)


def test_masks_rank_and_interference_ownership():
    layout, inputs = sensing_case()
    baseline, _ = assert_reference(layout, inputs)
    eig = torch.linalg.eigvalsh(baseline.fim[0, 0, 0])
    assert eig[1] > 0 and eig[0] <= 2 * torch.finfo(torch.float64).eps * eig[1]
    assert baseline.field_valid['fim'][0, 0, 0]
    relation = layout['geometry'].sensing[0]
    for field, value in (('active', False), ('power_w', 0.)):
        old = inputs.simultaneous_transmissions[field][0, 0, 0].item()
        inputs.simultaneous_transmissions[field][0, 0, 0] = value
        output, _ = assert_reference(layout, inputs)
        assert not output.field_valid['measurement'][0, 0, 0]
        assert output.field_valid['fim'][0, 0, 0] and not output.fim[0, 0, 0].any()
        inputs.simultaneous_transmissions[field][0, 0, 0] = old
    for endpoint in (0,):
        inputs.endpoint_masks['present'][0, endpoint] = False
        inputs.geometry['valid'][0, relation.outbound] = False
        output, _ = assert_reference(layout, inputs)
        assert output.field_valid['fim'][0, 0, 0] and not output.fim[0, 0, 0].any()
        inputs.endpoint_masks['present'][0, endpoint] = True
        assert not assert_reference(layout, inputs)[0].field_valid['fim'][0, 0, 0]
        inputs.geometry['valid'][0, relation.outbound] = True
    inputs.geometry['los'][0, relation.outbound] = False
    blocked, _ = assert_reference(layout, inputs)
    assert blocked.field_valid['fim'][0, 0, 0] and not blocked.fim[0, 0, 0].any()
    inputs.geometry['los'][0, relation.outbound] = True
    old_distance = inputs.geometry['distance_m'][0, relation.outbound].item()
    inputs.geometry['distance_m'][0, relation.outbound] = .5
    assert not assert_reference(layout, inputs)[0].field_valid['fim'][0, 0, 0]
    inputs.geometry['distance_m'][0, relation.outbound] = CONFIG['minimum_distance_m']
    assert assert_reference(layout, inputs)[0].field_valid['fim'][0, 0, 0]
    inputs.geometry['distance_m'][0, relation.outbound] = old_distance
    source = next(source for item in layout['geometry'].interference if item.consumer == 0 for source in item.sources if source.tx == 2)
    inputs.geometry['valid'][0, source.path] = False
    unknown, _ = assert_reference(layout, inputs)
    assert not unknown.field_valid['fim'][0, 0, 0]
    torch.testing.assert_close(unknown.fim[:, 1:], baseline.fim[:, 1:], rtol=0, atol=0)
    inputs.simultaneous_transmissions['active'][0, 3, 0] = False
    assert assert_reference(layout, inputs)[0].field_valid['fim'][0, 0, 0]
    inputs.simultaneous_transmissions['active'][0, 3, 0] = True
    old_power = inputs.simultaneous_transmissions['power_w'][0, 3, 0].item()
    inputs.simultaneous_transmissions['power_w'][0, 3, 0] = 0.
    assert assert_reference(layout, inputs)[0].field_valid['fim'][0, 0, 0]
    inputs.simultaneous_transmissions['power_w'][0, 3, 0] = old_power
    inputs.endpoint_masks['present'][0, 3] = False
    assert assert_reference(layout, inputs)[0].field_valid['fim'][0, 0, 0]
    inputs.endpoint_masks['present'][0, 3] = True
    # 未知路径不应污染非该source实际占用的resource。
    assert unknown.field_valid['fim'][0, 1, 1]
    inputs.geometry['valid'][0, source.path] = True
    inputs.geometry['los'][0, source.path] = False
    inputs.geometry['distance_m'][0, source.path] = .1
    suppressed, _ = assert_reference(layout, inputs)
    assert suppressed.field_valid['fim'][0, 0, 0]
    assert torch.linalg.matrix_norm(suppressed.fim[0, 0, 0]) > torch.linalg.matrix_norm(baseline.fim[0, 0, 0])
    inputs.geometry['los'][0, source.path] = True
    assert not assert_reference(layout, inputs)[0].field_valid['fim'][0, 0, 0]


def test_bistatic_zero_gradient_and_noncollinear_views():
    layout, inputs = sensing_case()
    # 两个非共线单站视角合计满rank；单关系rank1不等于invalid。
    output = evaluate(layout, inputs)
    combined = output.fim[0, :2].sum((0, 1))
    eig = torch.linalg.eigvalsh(combined)
    assert eig[0] > 2 * torch.finfo(torch.float64).eps * eig[1]
    positions = inputs.geometry['positions_world']
    positions[0, 3], positions[0, 4], positions[0, 5] = (
        torch.tensor(value) for value in ((0., 0., 1.), (2., 0., 1.), (4., 0., 1.)))
    for p, path in enumerate(layout['geometry'].paths):
        inputs.geometry['distance_m'][:, p] = torch.linalg.vector_norm(positions[:, path.tx].double() - positions[:, path.rx].double(), dim=-1)
    output, _ = assert_reference(layout, inputs)
    assert output.field_valid['measurement'][0, 2, 0] and output.field_valid['fim'][0, 2, 0]
    assert not output.fim[0, 2, 0].any()  # 目标位于两站连线内部。
    relation = layout['geometry'].sensing[2]
    inputs.geometry['valid'][0, relation.inbound] = False
    assert not assert_reference(layout, inputs)[0].field_valid['fim'][0, 2, 0]
    inputs.endpoint_masks['present'][0, 2] = False
    absent, _ = assert_reference(layout, inputs)
    assert absent.field_valid['fim'][0, 2, 0] and not absent.fim[0, 2, 0].any()


def test_duplicate_physical_observation_is_rejected():
    layout, _ = sensing_case()
    geometry = layout['geometry']
    duplicated = geometry.model_copy(update={'sensing': (geometry.sensing[0], geometry.sensing[0].model_copy(update={'sensing_id': 'other'}), *geometry.sensing[2:])})
    with pytest.raises(ValueError, match='duplicate|Duplicate|physical'):
        sensing_instance(LAWS[2], {**layout, 'geometry': duplicated}, config=CONFIG)


def test_permuted_mapping_and_target_isolation():
    layout, inputs = sensing_case()
    original, _ = assert_reference(layout, inputs)
    permuted, _ = assert_reference(*sensing_case(permutation=True))
    torch.testing.assert_close(permuted.fim, original.fim[:, [3, 1, 0, 2]][:, :, [2, 0, 1]], rtol=0, atol=0)
    # 仅第二目标的已量化geometry变化不得串入第一目标两个S。
    relation = layout['geometry'].sensing[2]
    inputs.geometry['distance_m'][:, relation.outbound] *= 1.5
    inputs.geometry['distance_m'][:, relation.inbound] *= 1.5
    changed, _ = assert_reference(layout, inputs)
    torch.testing.assert_close(original.fim[:, :2], changed.fim[:, :2], rtol=0, atol=0)
    assert not torch.equal(original.fim[:, 2], changed.fim[:, 2])


def test_numeric_faults_are_not_missing_measurements():
    layout, inputs = sensing_case()
    path = layout['geometry'].sensing[0].outbound
    inputs.geometry['distance_m'][0, path] = float('nan')
    with pytest.raises(RuntimeError, match='finite|numeric|distance'):
        evaluate(layout, inputs)
    inputs.geometry['valid'][0, path] = False
    assert not evaluate(layout, inputs).field_valid['measurement'][0, 0, 0]
    layout, inputs = sensing_case()
    for updates in ({'receiver_noise_temperature_k': 1e308, 'receiver_noise_figure_db': 1000.},
                    {'target_rcs_m2': 1e-320}, {'target_rcs_m2': 1e308}, {'receiver_noise_figure_db': 4000.}):
        with pytest.raises((RuntimeError, OverflowError), match='finite|numeric|variance|echo|noise|range'):
            evaluate(layout, inputs, config={**CONFIG, **updates})


def test_reset_and_rng_isolation():
    layout, inputs = sensing_case()
    rng = WorldRng(73, 2, 'cpu')
    instance = sensing_instance(LAWS[2], layout, config=CONFIG, rng=rng)
    control = sensing_instance(LAWS[2], layout, config=CONFIG)
    reset_inputs = inputs._replace(geometry={}, interval_s=torch.zeros_like(inputs.interval_s))
    global_state = torch.get_rng_state().clone()
    instance.reset(ModelResetMask(torch.tensor([True, True])), reset_inputs)
    assert not instance.output.measurement.any() and not instance.output.fim.any()
    assert not any(value.any() for value in instance.output.field_valid.values())
    first = instance.evaluate(inputs).output
    torch.testing.assert_close(first.measurement, control.evaluate(inputs).output.measurement, rtol=0, atol=0)
    saved = [value[1].clone() for value in (first.measurement, first.fim, *first.field_valid.values(), instance.sim_time, instance.interval_s)]
    pointers = [value.data_ptr() for value in (first.measurement, first.fim, *first.field_valid.values())]
    instance.reset(ModelResetMask(torch.tensor([True, False])), reset_inputs)
    for value, old in zip((first.measurement, first.fim, *first.field_valid.values(), instance.sim_time, instance.interval_s), saved):
        torch.testing.assert_close(value[1], old, rtol=0, atol=0)
    torch.testing.assert_close(instance.evaluate(inputs).output.measurement, control.evaluate(inputs).output.measurement, rtol=0, atol=0)
    assert pointers == [value.data_ptr() for value in (first.measurement, first.fim, *first.field_valid.values())]
    assert instance.state == () and instance.diagnostics == {}
    torch.testing.assert_close(torch.get_rng_state(), global_state, rtol=0, atol=0)
    # FIM-only不消耗同instance身份的measurement流。
    shared, fresh = WorldRng(73, 2, 'cpu'), WorldRng(73, 2, 'cpu')
    sensing_instance(LAWS[1], layout, config=CONFIG, rng=shared).evaluate(inputs)
    lhs = sensing_instance(LAWS[0], layout, config=CONFIG, rng=shared).evaluate(inputs).output
    rhs = sensing_instance(LAWS[0], layout, config=CONFIG, rng=fresh).evaluate(inputs).output
    torch.testing.assert_close(lhs.measurement, rhs.measurement, rtol=0, atol=0)
    other = {**layout, 'instance_id': 'other'}
    assert not torch.equal(sensing_instance(LAWS[0], other, config=CONFIG, rng=shared).evaluate(inputs).output.measurement, lhs.measurement)
    assert not torch.equal(lhs.measurement[0], lhs.measurement[1])


def test_normalized_measurement_statistics(tmp_path):
    layout, inputs = sensing_case()
    instance = sensing_instance(LAWS[0], layout, config=CONFIG)
    expected = reference(layout, inputs)
    h, v = torch.tensor(expected['h'], dtype=torch.float64), torch.tensor(expected['variance'], dtype=torch.float64)
    mask = torch.tensor(expected['measurement_valid'])
    n, x = 4096, 20.
    residuals = torch.stack([((instance.evaluate(inputs).output.measurement.clone() - h) / torch.sqrt(v))[mask] for _ in range(n)])
    means, variances = residuals.mean(0), residuals.var(0)
    assert (means.abs() <= math.sqrt(2 * x / n)).all()
    assert (variances >= 1 - 2 * math.sqrt(x / (n - 1))).all()
    assert (variances <= 1 + 2 * math.sqrt(x / (n - 1)) + 2 * x / (n - 1)).all()
    (tmp_path / 'statistics.txt').write_text(f'N={n}; x={x}; means={means.tolist()}; variances={variances.tolist()}\n')
    assert (residuals[:, 0] != residuals[:, 1]).any()
    # 合法低SNR Gaussian可为负，不加检测阈值或裁剪。
    tiny = sensing_instance(LAWS[0], layout, config={**CONFIG, 'target_rcs_m2': 1e-12})
    assert (torch.stack([tiny.evaluate(inputs).output.measurement.clone() for _ in range(16)])[..., mask[0]] < 0).any()


def test_lightweight_describe_and_metadata_registration():
    code = '''
import sys
from pathlib import Path
from muse_builtins.models import sensing_registrations
from muse_builtins.metadata import export_metadata
from muse_sdk.descriptor import parse_descriptor_json
metadata = export_metadata(Path('examples/basic_flight/controller.json'))
descriptor = parse_descriptor_json(metadata.descriptor.model_dump_json())
assert set(descriptor.sensing_models) == set(sensing_registrations())
for name, model in sensing_registrations().items():
    assert descriptor.sensing_models[name].assumptions == model.describe()['assumptions']
assert 'torch' not in sys.modules
assert 'warp' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
    from muse_builtins.metadata import export_metadata
    from muse_builtins.models import sensing_registrations
    metadata = export_metadata(Path('examples/basic_flight/controller.json'))
    assert set(metadata.descriptor.sensing_models) == set(LAWS)
    for law in LAWS:
        descriptor = metadata.descriptor.sensing_models[law]
        schema = descriptor.config_schema
        assert set(schema['required']) == set(CONFIG)
        assert all('default' not in value for value in schema['properties'].values())
        assert not any(item.phase == 'reset' for item in descriptor.input_fields)
        assert any(item.path == ['network', 'interval_s'] for item in descriptor.input_fields)
        uses_position = any(item.path == ['network', 'geometry', 'positions_world'] for item in descriptor.input_fields)
        assert uses_position == (law != LAWS[0])
        for field, units, frame in (('measurement', 'm', 'none'), ('fim', 'm^-2', 'world_z_up')):
            spec = getattr(descriptor.output_spec, field)
            if spec is not None:
                assert spec.units == units and spec.frame == frame and spec.dtype == 'float64'
