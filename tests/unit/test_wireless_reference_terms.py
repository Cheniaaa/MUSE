"""共享无线消费者：冻结映射、局部可见性与轻量登记。"""
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from muse_sdk.geometry import PropagationPathLayout
from muse_sdk.network import NetworkSpec, SensingFieldView


def feedback_env():
    # Q、E、L、role slot均不同序；resource顺序不是channel ID顺序。
    layout = PropagationPathLayout(points=tuple(dict(group_id=group, slot=slot,
        device_id=device, body='root', position_m=(0., 0., 0.), assumption='test')
        for group, slot, device in [('targets', 0, None), ('uavs', 1, 'radio'),
            ('stations', 0, 'radio'), ('others', 0, 'radio'), ('uavs', 0, 'radio'),
            ('stations', 1, 'radio')]),
        paths=tuple(dict(path_id=f'p{i}', tx=tx, rx=rx) for i, (tx, rx) in enumerate(
            [(3, 2), (1, 5), (4, 2), (1, 2)])),
        desired=tuple(dict(link_id=name, path=path) for name, path in
            [('other_up', 0), ('upload', 2), ('alternate', 1), ('upload', 3)]),
        sensing=(), interference=())
    endpoints = (2, 4, 3, 1, 5)
    network = NetworkSpec(endpoints=tuple(layout.points[i] for i in endpoints),
        endpoint_sets={}, endpoint_to_point=endpoints,
        resources=tuple(dict(resource_id=f'band_{i}', channel_id=channel,
            center_frequency_hz=2.412e9+i*25e6, bandwidth_hz=5e6, sharing='reuse')
            for i, channel in enumerate([19, 7, -3])),
        endpoint_limits=tuple(dict(resource_indices=(0, 1, 2), max_power_w=.001) for _ in endpoints),
        role_controls=(dict(role_id='observer', endpoint_indices=(3, 1)),
                       dict(role_id='other', endpoint_indices=(2,))),
        background_emitters=(), passive_endpoints=(0, 4), communication_allocations=(), services=(),
        epoch_offsets=(0, 7, 14, 20), physics_dt=.002,
        link_count=4, sensing_count=0, point_count=6, path_count=4)
    rate = SimpleNamespace(rate_bps=torch.full((2, 4, 3), float('nan')),
                           valid=torch.zeros(2, 4, 3, dtype=torch.bool))
    transmissions = SimpleNamespace(active=torch.zeros(2, 5, 3, dtype=torch.bool),
                                     power_w=torch.zeros(2, 5, 3))
    allocations = SimpleNamespace(link_active=torch.zeros(2, 4, 3, dtype=torch.bool))
    for world, link, endpoint, resource, power, value, valid in (
            (0, 3, 3, 2, .00025, 2e7, True), (0, 1, 1, 0, .0005, 4e7, True),
            (1, 3, 3, 1, .00075, 6e7, True), (1, 1, 1, 2, .00025, float('nan'), False)):
        transmissions.active[world, endpoint, resource] = True
        transmissions.power_w[world, endpoint, resource] = power
        allocations.link_active[world, link, resource] = True
        rate.rate_bps[world, link, resource] = value
        rate.valid[world, link, resource] = valid
    return SimpleNamespace(device='cpu', experiment=SimpleNamespace(
        geometry=SimpleNamespace(layout=layout), network=network,
        bindings={'observer': SimpleNamespace(group_id='uavs', role_slot_to_group_slot=[1, 0])},
        scenario=SimpleNamespace(scene=SimpleNamespace(control_substeps=20))),
        physics_outputs=SimpleNamespace(episode_decision=torch.tensor([1, 3])),
        network=SimpleNamespace(radio=SimpleNamespace(rate=rate, sample_tick=torch.tensor([14, 54])),
            simultaneous_transmissions=transmissions, allocations=allocations))


def feedback_term(env):
    from muse_builtins.wireless_terms import SelectedLinkFeedback
    params = dict(role_id='observer', link_id='upload', rate_scale_bps=1e8, power_scale_w=.001)
    return SelectedLinkFeedback(SimpleNamespace(params=params), env), params


def test_selected_feedback_maps_local_endpoint_link_resource_and_masks_nan():
    env = feedback_env()
    term, params = feedback_term(env)
    expected = torch.tensor([
        [[[0., 0., 0., 0.], [0., 0., 0., 0.], [.2, 1., .25, .3]],
         [[.4, 1., .5, .3], [0., 0., 0., 0.], [0., 0., 0., 0.]]],
        [[[0., 0., 0., 0.], [.6, 1., .75, .3], [0., 0., 0., 0.]],
         [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., .25, 0.]]]])
    result = term(env, **params)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected)
    # 不得从其他role、未选频道、target真值或FIM推导本机反馈。
    env.network.radio.rate.rate_bps[:, 0] = 9e9
    env.network.radio.rate.valid[:, 0] = True
    env.network.simultaneous_transmissions.power_w[:, 2] = .001
    env.network.simultaneous_transmissions.active[:, 2] = True
    env.network.radio.rate.rate_bps[0, 3, :2] = 8e9
    env.network.radio.rate.valid[0, 3, :2] = True
    env.role_positions_world = {'target': torch.full((2, 1, 3), 999.)}
    env.sensing_outputs = {'fim': 'must not read'}
    torch.testing.assert_close(term(env, **params), expected)


def test_feedback_invalid_preserves_power_zero_power_and_subset_reset_are_empty():
    env = feedback_env()
    term, params = feedback_term(env)
    env.network.radio.rate.valid[0, 3, 2] = False
    env.network.radio.rate.rate_bps[0, 3, 2] = float('nan')
    torch.testing.assert_close(term(env, **params)[0, 0, 2], torch.tensor([0., 0., .25, 0.]))
    # 实际0W的既有runtime契约：active/link_active同时清零。
    env.network.simultaneous_transmissions.active[0, 3] = False
    env.network.simultaneous_transmissions.power_w[0, 3] = 0
    env.network.allocations.link_active[0, 3] = False
    assert torch.equal(term(env, **params)[0, 0], torch.zeros(3, 4))
    before = term(env, **params)[1].clone()
    env.physics_outputs.episode_decision[0] = 0
    env.network.radio.sample_tick[0] = 0
    env.network.radio.rate.valid[0] = False
    env.network.simultaneous_transmissions.active[0] = False
    env.network.simultaneous_transmissions.power_w[0] = 0
    env.network.allocations.link_active[0] = False
    assert torch.equal(term(env, **params)[0], torch.zeros(2, 3, 4))
    torch.testing.assert_close(term(env, **params)[1], before)


def test_feedback_age_uses_full_decision_clock_and_does_not_clip_or_normalize_clip():
    env = feedback_env()
    term, params = feedback_term(env)
    env.network.radio.sample_tick[1] = 14
    env.network.radio.rate.rate_bps[1, 3, 1] = 2e8
    torch.testing.assert_close(term(env, **params)[1, 0, 1], torch.tensor([2., 1., .75, 2.3]))


def test_feedback_requires_actual_transmission_and_link_allocation():
    env = feedback_env()
    term, params = feedback_term(env)
    env.network.simultaneous_transmissions.active[0, 3, 2] = False
    env.network.allocations.link_active[0, 1, 0] = False
    assert torch.equal(term(env, **params)[0], torch.zeros(2, 3, 4))


@pytest.mark.parametrize('mode', ['missing', 'multiple'])
def test_feedback_rejects_non_unique_logical_link_per_owned_role_slot(mode):
    env = feedback_env()
    layout = env.experiment.geometry.layout
    desired = tuple(item.model_copy(update={'link_id': 'upload' if mode == 'multiple' else 'missing'})
                    if i == (2 if mode == 'multiple' else 3) else item
                    for i, item in enumerate(layout.desired))
    env.experiment.geometry.layout = layout.model_copy(update={'desired': desired})
    with pytest.raises(ValueError, match='exactly one.*link.*role slot'):
        feedback_term(env)


def test_wireless_registration_descriptor_is_cpu_light_and_declares_real_dependencies():
    code = '''import sys
from muse_builtins.wireless_terms import registrations
from muse_sdk.descriptor import build_descriptor, parse_descriptor_json
from muse_sdk.registration import TrainingPackageDefinition
terms=registrations()
descriptor=build_descriptor(TrainingPackageDefinition(terms, {}, {}, {}, {}), {}, 'a'*64, default_composition_id=None)
restored=parse_descriptor_json(descriptor.model_dump_json())
assert set(restored.terms)=={'local_path_length','position_bound_violation','selected_link_feedback'}
assert not {'torch','warp'} & sys.modules.keys()
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    from muse_builtins.wireless_terms import registrations
    registration = registrations()['selected_link_feedback']
    assert registration.output_spec['shape'] == ['B', 'N_role', 'F', 4]
    assert {tuple(field['path']) for field in registration.input_fields} == {
        ('network', 'radio', 'rate', 'rate_bps'), ('network', 'radio', 'rate', 'valid'),
        ('network', 'radio', 'sample_tick'), ('network', 'allocations', 'link_active'),
        ('network', 'simultaneous_transmissions', 'active'),
        ('network', 'simultaneous_transmissions', 'power_w'), ('physics_outputs', 'episode_decision')}


def test_builtin_lookup_exposes_common_wireless_consumers():
    from muse_builtins.terms import registrations
    assert {'local_path_length', 'position_bound_violation', 'selected_link_feedback'} <= registrations().keys()


@pytest.mark.parametrize('field,value', [('link_id', ''), ('rate_scale_bps', 0), ('power_scale_w', -1)])
def test_feedback_config_rejects_invalid_user_values(field, value):
    from muse_builtins.wireless_terms import SelectedLinkFeedbackConfig
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        SelectedLinkFeedbackConfig(**(dict(link_id='upload', rate_scale_bps=1e8, power_scale_w=.001)
                                     | {field: value}))


def test_position_bound_helpers_group_by_target_point_and_share_term_semantics():
    from muse_builtins.wireless_terms import (PositionBoundViolation, position_bound_violation,
                                              position_target_groups)
    from test_sensing_reference_terms import term_env
    env = term_env()
    groups = position_target_groups(env.experiment.geometry.layout, device='cpu')
    assert [indices.tolist() for indices in groups] == [[0, 1], [2, 3]]
    data = torch.zeros(3, 4, 2, 2, 2, dtype=torch.float64)
    data[:, :, :, 0, 0] = 100.
    data[:, :, :, 1, 1] = 100.
    data[1, 2:, :, 1, 1] = 0.
    valid = torch.ones(3, 4, 2, dtype=torch.bool)
    valid[2, 0, 0] = False
    data[2, 0, 0] = float('nan')
    view = SensingFieldView(data, valid, None, None)
    env.sensing_outputs = {'fim': view}
    expected = torch.tensor([0., 1., 1.])
    torch.testing.assert_close(position_bound_violation(view, groups, limit_m=.25), expected)
    term = PositionBoundViolation(SimpleNamespace(params={}), env)
    torch.testing.assert_close(term(env, position_bound_limit_m=.25), expected)
