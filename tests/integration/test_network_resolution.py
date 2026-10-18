"""纯descriptor网络解析及物理归属；不构造任何模型实现。"""
from copy import deepcopy

import pytest
from muse_sdk.descriptor import InputField
from muse_sdk.experiment import learning_contract
from muse_server.resolver import resolve_experiment
from wireless_helpers import resource, wireless_request


def resolve(request, snapshot):
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors
    return report.resolved_experiment


def rejected(request, snapshot, message):
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert any(message in e['message'] or message in e['field'] for e in report.errors), report.errors


@pytest.mark.parametrize('k,expected', [(4, (0, 4, 8, 10)), (20, (0, 10))])
def test_epoch_offsets_are_bounded_and_clip_tail(pipeline, k, expected):
    request, snapshot = wireless_request(pipeline)
    request['scenario']['network']['epoch_max_ticks'] = k
    exp = resolve(request, snapshot)
    assert exp.network.epoch_offsets == expected
    assert [p.slot for p in exp.network.endpoints] == [1, 0, 0]
    assert exp.model_bindings['radio'].config == {'scale': 1.}
    assert exp.model_bindings['radio'].input_fields[0].tensor.shape == [2, 2, 1]


def test_resolve_activation_matrix(pipeline):
    request, snapshot = wireless_request(pipeline, service=False)
    # 模型候选+设备候选不执行；显式emitter独立于Radio。
    exp = resolve(request, snapshot)
    assert exp.network is not None and exp.model_bindings == {}
    assert 'radio' not in exp.activation
    expected_layout = [
        dict(role_id='scout', field='motion', start=0, stop=4, shape=[2, 1, 4]),
        dict(role_id='guard', field='motion', start=4, stop=12, shape=[2, 2, 4])]
    assert [entry.model_dump() for entry in exp.action_layout] == expected_layout
    request['scenario']['network']['background_emitters'] = []
    request['scenario']['network']['passive_endpoints'] = ['tx', 'rx']
    inactive = resolve(request, snapshot)
    assert inactive.network is None
    assert [entry.model_dump() for entry in inactive.action_layout] == expected_layout
    request['scenario']['network']['services'] = [dict(service_id='payload', link_id='up', resource_ids=['band'], initial_bits=0.)]
    assert 'radio' in resolve(request, snapshot).model_bindings
    package = next(x for x in snapshot.resources if x.kind == 'training_package')
    package.metadata['radio_models']['oracle']['output_spec']['output_blocks'] = ['link']
    rejected(request, snapshot, 'rate')


def test_background_without_paths_or_rf_semantics(pipeline):
    request, snapshot = wireless_request(pipeline, service=False, radio=False)
    request['scenario']['radio_geometry']['links'] = []
    request['scenario']['network']['communication_allocations'] = []
    environment = next(x for x in snapshot.resources if x.kind == 'environment')
    environment.metadata['rf_geometry'] = None
    environment.metadata.pop('geometry_summary', None)
    exp = resolve(request, snapshot)
    assert exp.geometry is None and exp.network.link_count == 0 and exp.model_bindings == {}


def test_physical_rf_and_service_owners_are_unique(pipeline):
    request, snapshot = wireless_request(pipeline)
    duplicate = deepcopy(request)
    duplicate['scenario']['radio_geometry']['links'].append(dict(link_id='alias', tx='tx', rx='rx'))
    duplicate['scenario']['network']['communication_allocations'].append(dict(link_id='alias', resource_ids=['band']))
    rejected(duplicate, snapshot, 'physical tx/resource')
    duplicate = deepcopy(request)
    duplicate['scenario']['network']['services'].append(dict(service_id='again', link_id='up', resource_ids=['band'], initial_bits=1.))
    rejected(duplicate, snapshot, 'service')
    duplicate = deepcopy(request)
    duplicate['scenario']['network']['role_controls'] = {'guard': 'tx'}
    rejected(duplicate, snapshot, 'RF owner')


def test_channels_and_radio_role_keep_declared_order(pipeline):
    request, snapshot = wireless_request(pipeline, service=False)
    network = request['scenario']['network']
    network['resources'] = [resource('a', 7), resource('b', -3, 2e9), resource('c', 19, 3e9)]
    network['endpoint_limits']['tx']['resource_ids'] = ['a', 'b', 'c']
    network['endpoint_limits']['rx']['resource_ids'] = ['a']
    network['background_emitters'] = []
    network['role_controls'] = {'guard': 'tx'}
    network['communication_allocations'][0]['resource_ids'] = ['a', 'b', 'c']
    package = next(x for x in snapshot.resources if x.kind == 'training_package')
    package.metadata['compositions']['default']['agent_roles']['guard']['action_interfaces']['radio'] = 'power_channel'
    snapshot.builtins.descriptor.algorithms['constant_velocity'].action_requirements.append('power_channel')
    exp = resolve(request, snapshot)
    assert exp.action_specs['guard'].radio.channel_ids == [7, -3, 19]
    assert exp.network.role_controls[0].endpoint_indices == (0, 1)
    assert [entry.model_dump() for entry in exp.action_layout] == [
        dict(role_id='scout', field='motion', start=0, stop=4, shape=[2, 1, 4]),
        dict(role_id='guard', field='motion', start=4, stop=12, shape=[2, 2, 4]),
        dict(role_id='guard', field='radio', start=12, stop=16, shape=[2, 2, 2])]


def test_overlapping_spectrum_and_short_dwell_rejected(pipeline):
    request, snapshot = wireless_request(pipeline)
    request['scenario']['network']['resources'].append(resource('duplicate', -3))
    rejected(request, snapshot, 'overlap')
    request, snapshot = wireless_request(pipeline)
    request['scenario']['network']['background_emitters'][0].update(behavior='periodic',
        config=dict(on_ticks=3, off_ticks=4, phase_ticks=0, power_w=[1., 1.], channel_id=[7, 7]))
    rejected(request, snapshot, 'dwell')


def test_network_contract_changes_preserve_only_batch_invariance(pipeline):
    request, snapshot = wireless_request(pipeline)
    original = learning_contract(resolve(request, snapshot))
    changed = deepcopy(request)
    changed['topology']['world_count'] = 7
    assert learning_contract(resolve(changed, snapshot)) == original
    mutations = [lambda n: n.update(epoch_max_ticks=5),
        lambda n: n['endpoint_limits']['tx'].update(max_power_w=3.),
        lambda n: n['services'][0].update(initial_bits=9.),
        lambda n: n['background_emitters'][0]['config'].update(power_w=[.5, 1.])]
    for mutate in mutations:
        changed = deepcopy(request)
        mutate(changed['scenario']['network'])
        assert learning_contract(resolve(changed, snapshot)) != original
    changed = deepcopy(request)
    changed['model_config'] = {'radio': {'scale': 2.}}
    assert learning_contract(resolve(changed, snapshot)) != original


@pytest.mark.parametrize('frequency,valid', [
    (5e8, False), (5e8 + 1, True), (1e11 - 1, True), (1e11, False),
])
@pytest.mark.parametrize('resource_index', [0, 1])
@pytest.mark.parametrize('builtin', [False, True])
def test_active_radio_frequency_domain_uses_canonical_resources(pipeline, frequency, valid, resource_index, builtin):
    from muse_sdk.descriptor import RadioDescriptor
    request, snapshot = wireless_request(pipeline)
    package = next(x for x in snapshot.resources if x.kind == 'training_package')
    schema = {'type': 'number', 'exclusiveMinimum': 5e8, 'exclusiveMaximum': 1e11}
    package.metadata['radio_models']['oracle']['frequency_hz_schema'] = schema
    if builtin:
        snapshot.builtins.descriptor.radio_models['oracle'] = RadioDescriptor.model_validate(
            package.metadata['radio_models']['oracle'])
        package.metadata['compositions']['default']['model_bindings']['radio'] = {
            'default': 'builtin:oracle', 'options': {'builtin:oracle': {}}}
    resources = request['scenario']['network']['resources']
    resources.append(resource('other', -3, 2e9))
    # 第二个resource无需被allocation使用，科学域仍须覆盖全部canonical资源。
    resources[resource_index]['center_frequency_hz'] = frequency
    if valid:
        exp = resolve(request, snapshot)
        assert exp.network.resources[resource_index].center_frequency_hz == frequency
        assert exp.model_bindings['radio'].descriptor.frequency_hz_schema == schema
    else:
        report = resolve_experiment(request, snapshot)
        assert not report.valid
        assert [error['field'] for error in report.errors] == [
            f'scenario.network.resources.{resource_index}.center_frequency_hz']


def test_inactive_radio_domain_does_not_restrict_other_experiments(pipeline):
    request, snapshot = wireless_request(pipeline)
    package = next(x for x in snapshot.resources if x.kind == 'training_package')
    package.metadata['radio_models']['oracle']['frequency_hz_schema'] = {
        'type': 'number', 'exclusiveMinimum': 5e8, 'exclusiveMaximum': 1e11}
    request['scenario']['network']['resources'][0]['center_frequency_hz'] = 5e8
    request['scenario']['network']['services'] = []
    exp = resolve(request, snapshot)
    assert exp.network is not None and exp.model_bindings == {}
    assert 'radio' not in exp.activation

    # 同一场景激活另一个无科学频率域的包模型，不受候选模型限制。
    package.metadata['radio_models']['unrestricted'] = {
        **deepcopy(package.metadata['radio_models']['oracle']),
        'model_id': 'unrestricted', 'frequency_hz_schema': None}
    package.metadata['compositions']['default']['model_bindings']['radio'] = {
        'default': 'unrestricted', 'options': {'unrestricted': {}}}
    request['scenario']['network']['services'] = [
        dict(service_id='payload', link_id='up', resource_ids=['band'], initial_bits=7.)]
    exp = resolve(request, snapshot)
    assert exp.model_bindings['radio'].reference == 'unrestricted'
    assert exp.model_bindings['radio'].descriptor.frequency_hz_schema is None

    package.metadata['compositions']['default']['model_bindings']['radio'] = {
        'default': 'oracle', 'options': {'oracle': {}}}
    request['scenario']['network']['services'] = []
    request['scenario']['network']['background_emitters'] = []
    request['scenario']['network']['passive_endpoints'] = ['tx', 'rx']
    exp = resolve(request, snapshot)
    assert exp.network is None and exp.model_bindings == {}


def test_frequency_domain_is_frozen_in_learning_contract(pipeline):
    request, snapshot = wireless_request(pipeline)
    package = next(x for x in snapshot.resources if x.kind == 'training_package')
    schema = {'type': 'number', 'exclusiveMinimum': 5e8, 'exclusiveMaximum': 1e11}
    package.metadata['radio_models']['oracle']['frequency_hz_schema'] = deepcopy(schema)
    exp = resolve(request, snapshot)
    original = learning_contract(exp)
    assert original['model_bindings']['radio']['descriptor']['frequency_hz_schema'] == schema
    changed = deepcopy(request)
    changed['topology']['world_count'] = 7
    assert learning_contract(resolve(changed, snapshot)) == original
    package.metadata['radio_models']['oracle']['frequency_hz_schema']['exclusiveMinimum'] = 6e8
    assert learning_contract(resolve(request, snapshot)) != original
    assert exp.model_bindings['radio'].descriptor.frequency_hz_schema == schema


def sensing_descriptor(*, radio_input=False, shape=None):
    return dict(model_id='sensor', config_schema=dict(type='object', properties={}, additionalProperties=False),
        input_fields=[dict(path=['network', 'radio', 'rate', 'rate_bps'], phase='post_physics',
            tensor=dict(shape=['B', 'L', 'F'], dtype='float32', units='bit/s', frame='none', rate='radio_epoch'))] if radio_input else [],
        state_scope='world', state_schema={}, diagnostics_schema={}, assumptions={'synthetic': True},
        output_spec={'measurement': dict(shape=shape or ['B', 'S', 2], dtype='float32', units='m',
            frame='none', rate='radio_epoch', time_semantics='snapshot')})


@pytest.mark.parametrize('kind,radio_input', [('radio', False), ('sensing', False), ('sensing', True)])
def test_algorithm_only_model_requirements_activate_their_dependencies(pipeline, kind, radio_input):
    request, snapshot = wireless_request(pipeline, service=False)
    package = next(x for x in snapshot.resources if x.kind == 'training_package')
    if kind == 'sensing':
        package.metadata['sensing_models']['sensor'] = sensing_descriptor(radio_input=radio_input, shape=['B', 'L', 2])
        package.metadata['compositions']['default']['model_bindings']['sensing'] = {
            'default': 'sensor', 'options': {'sensor': {}}}
    baseline = resolve(request, snapshot)
    assert baseline.model_bindings == {}
    requirement = InputField(
        path=['network', 'radio', 'rate', 'rate_bps'] if kind == 'radio' else ['sensing_outputs', 'measurement', 'data'],
        phase='post_physics', tensor=dict(shape=['B', 'L', 'F'] if kind == 'radio' else ['B', 'L', 2],
            dtype='float32', units='bit/s' if kind == 'radio' else 'm', frame='none', rate='radio_epoch'))
    snapshot.builtins.descriptor.algorithms['constant_velocity'].observation_requirements = [requirement]
    exp = resolve(request, snapshot)
    assert set(exp.model_bindings) == ({'radio', 'sensing'} if radio_input else {kind})
    assert exp.activation[kind] == ['algorithm:builtin:constant_velocity']
    assert exp.algorithm_inputs.observations[0].path == requirement.path
    assert exp.algorithm_inputs.observations[0].tensor.shape == ([2, 2, 1] if kind == 'radio' else [2, 2, 2])
    assert exp.algorithm_inputs.actors == baseline.algorithm_inputs.actors
    assert exp.algorithm_inputs.critics == baseline.algorithm_inputs.critics
    del package.metadata['compositions']['default']['model_bindings'][kind]
    rejected(request, snapshot, 'Required model binding')


def test_algorithm_requirement_does_not_grant_actor_direct_radio_access(pipeline):
    request, snapshot = wireless_request(pipeline, service=False)
    field = InputField(path=['network', 'radio', 'rate', 'rate_bps'], phase='post_physics',
        tensor=dict(shape=['B', 'L', 'F'], dtype='float32', units='bit/s', frame='none', rate='radio_epoch'))
    algorithm = snapshot.builtins.descriptor.algorithms['constant_velocity']
    algorithm.observation_requirements = [field]
    algorithm.actor_observation = {'guard': [field]}
    rejected(request, snapshot, 'Actor must only read its own observations')


def add_sensing_consumer(snapshot, *, radio_input=False, shape=None):
    package = next(x for x in snapshot.resources if x.kind == 'training_package')
    package.metadata['sensing_models']['sensor'] = sensing_descriptor(radio_input=radio_input, shape=shape)
    package.metadata['compositions']['default']['model_bindings']['sensing'] = {
        'default': 'sensor', 'options': {'sensor': {}}}
    package.metadata['terms']['measurement_score'] = dict(term_kind='reward', scope='world',
        config_schema=dict(type='object', properties={}, additionalProperties=False), input_fields=[dict(
            path=['sensing_outputs', 'measurement', 'data'], phase='post_physics',
            tensor=dict(shape=shape or ['B', 'S', 2], dtype='float32', units='m', frame='none', rate='radio_epoch'))],
        output_spec=dict(shape=['B'], dtype='float32', units='1', frame='none', rate='decision'))
    package.metadata['compositions']['default']['world_reward_terms'] = [dict(selection_id='measurement', term='measurement_score')]


def test_sensing_only_and_radio_dependency_are_explicit(pipeline):
    request, snapshot = wireless_request(pipeline, service=False, radio=False)
    request['scenario']['radio_geometry']['links'] = []
    request['scenario']['radio_geometry']['sensing'] = [dict(sensing_id='echo', tx='tx', rx='rx',
        target=dict(group_id='scout_group', slots=[0]))]
    request['scenario']['network']['communication_allocations'] = []
    add_sensing_consumer(snapshot)
    exp = resolve(request, snapshot)
    assert set(exp.model_bindings) == {'sensing'} and exp.network.link_count == 0
    assert exp.network.sensing_count == 2
    from muse_sdk.network import network_model_input_fields, network_public_fields
    inputs = network_model_input_fields(exp.network, geometry=exp.geometry, batch_size=2, phase='reset')
    assert not any(item.path[:2] == ['network', 'allocations'] for item in inputs)
    public = {tuple(item.path): item.tensor.shape for item in network_public_fields(exp.network, exp.model_bindings, batch_size=2)}
    assert public[('sensing_outputs', 'measurement', 'data')] == [2, 2, 2]
    assert public[('sensing_outputs', 'measurement', 'interval_end_tick')] == [2]
    assert not any(path[:2] == ('network', 'radio') for path in public)
    package = next(x for x in snapshot.resources if x.kind == 'training_package')
    package.metadata['sensing_models']['sensor'] = sensing_descriptor(radio_input=True)
    rejected(request, snapshot, 'Required model binding')


def test_radio_upstream_fields_bind_both_real_phases(pipeline):
    request, snapshot = wireless_request(pipeline)
    add_sensing_consumer(snapshot, radio_input=True, shape=['B', 'L', 2])
    package = next(x for x in snapshot.resources if x.kind == 'training_package')
    package.metadata['sensing_models']['sensor']['input_fields'] *= 2
    package.metadata['sensing_models']['sensor']['input_fields'][0] = deepcopy(package.metadata['sensing_models']['sensor']['input_fields'][0])
    package.metadata['sensing_models']['sensor']['input_fields'][0]['phase'] = 'reset'
    exp = resolve(request, snapshot)
    assert set(exp.model_bindings) == {'radio', 'sensing'}
    assert [item.phase for item in exp.model_bindings['sensing'].input_fields] == ['reset', 'post_physics']
    assert exp.activation['radio'] == ['service:payload', 'model:sensing']


def test_model_geometry_and_unknown_axes_require_real_layout(pipeline):
    request, snapshot = wireless_request(pipeline, service=False)
    add_sensing_consumer(snapshot, shape=['B', 'missing_axis', 2])
    rejected(request, snapshot, 'missing_axis')
    package = next(x for x in snapshot.resources if x.kind == 'training_package')
    package.metadata['sensing_models']['sensor'] = sensing_descriptor(shape=['B', 'E', 2])
    package.metadata['terms']['measurement_score']['input_fields'][0]['tensor']['shape'] = ['B', 'E', 2]
    package.metadata['sensing_models']['sensor']['input_fields'] = [dict(
        path=['network', 'geometry', 'distance_m'], phase='post_physics',
        tensor=dict(shape=['B', 'P'], dtype='float32', units='m', frame='none', rate='radio_epoch'))]
    request['scenario']['radio_geometry']['links'] = []
    request['scenario']['network']['communication_allocations'] = []
    rejected(request, snapshot, 'P')


def test_service_fields_bind_the_specific_service_link_axis(pipeline):
    request, snapshot = wireless_request(pipeline)
    package = next(x for x in snapshot.resources if x.kind == 'training_package')
    package.metadata['terms']['service_score'] = dict(term_kind='reward', scope='world',
        config_schema=dict(type='object', properties={}, additionalProperties=False), input_fields=[dict(
            path=['network', 'services', 'payload', 'remaining_bits'], phase='post_physics',
            tensor=dict(shape=['B', 'N_service_links'], dtype='float64', units='bit', frame='none', rate='decision'))],
        output_spec=dict(shape=['B'], dtype='float32', units='1', frame='none', rate='decision'))
    package.metadata['compositions']['default']['world_reward_terms'] = [dict(selection_id='remaining', term='service_score')]
    exp = resolve(request, snapshot)
    selection = next(s for s in exp.selections if s.selection_id == 'remaining')
    assert selection.input_fields[0].tensor.shape == [2, 2]
    package.metadata['terms']['service_score']['input_fields'][0]['path'][2] = 'unknown'
    rejected(request, snapshot, 'Unavailable public field')


@pytest.mark.parametrize('mutation,message', [
    (lambda n: n['background_emitters'][0]['config'].update(power_w=[1.]), 'vectors'),
    (lambda n: n['background_emitters'][0]['config'].update(power_w=[3., 1.]), 'budget'),
    (lambda n: n['background_emitters'][0]['config'].update(channel_id=[-3, 7]), 'budget'),
    (lambda n: n.update(passive_endpoints=[]), 'explicit RF owner'),
    (lambda n: n['services'][0].update(resource_ids=['unknown']), 'resource'),
])
def test_static_network_boundaries_reject_invalid_declarations(pipeline, mutation, message):
    request, snapshot = wireless_request(pipeline)
    mutation(request['scenario']['network'])
    rejected(request, snapshot, message)


def test_same_tx_different_resources_and_receivers_are_legal(pipeline):
    request, snapshot = wireless_request(pipeline, service=False)
    network = request['scenario']['network']
    network['resources'].append(resource('other', -3, 2e9))
    network['endpoint_limits']['tx']['resource_ids'].append('other')
    request['scenario']['radio_geometry']['endpoints']['rx2'] = dict(group_id='anchor_group', slots=[1], device_id='ideal_point_radio')
    request['scenario']['radio_geometry']['links'].append(dict(link_id='other_link', tx='tx', rx='rx2'))
    network['endpoint_limits']['rx2'] = dict(resource_ids=['other'], max_power_w=2.)
    network['passive_endpoints'].append('rx2')
    network['communication_allocations'].append(dict(link_id='other_link', resource_ids=['other']))
    exp = resolve(request, snapshot)
    assert [(a.tx, a.rx, a.resource_indices) for a in exp.network.communication_allocations] == [
        (0, 2, (0,)), (1, 2, (0,)), (0, 3, (1,)), (1, 3, (1,))]


def test_active_network_resolver_cannot_import_execution_stack(pipeline, tmp_path):
    import json
    import subprocess
    import sys
    request, snapshot = wireless_request(pipeline)
    package = next(x for x in snapshot.resources if x.kind == 'training_package')
    package.metadata['radio_models']['oracle']['frequency_hz_schema'] = {
        'type': 'number', 'exclusiveMinimum': 5e8, 'exclusiveMaximum': 1e11}
    request_path, snapshot_path = tmp_path / 'request.json', tmp_path / 'snapshot.json'
    request_path.write_text(json.dumps(request))
    snapshot_path.write_text(snapshot.model_dump_json())
    script = '''
import importlib.abc, json, sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'muse_runtime','muse_builtins','torch','mujoco','mjlab','warp'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from muse_sdk.experiment import CatalogSnapshot
from muse_server.resolver import resolve_experiment
snapshot = CatalogSnapshot.model_validate_json(Path(sys.argv[2]).read_text())
report = resolve_experiment(json.loads(Path(sys.argv[1]).read_text()), snapshot)
assert report.valid, report.errors
assert report.resolved_experiment.model_bindings['radio'].reference == 'oracle'
'''
    result = subprocess.run([sys.executable, '-c', script, str(request_path), str(snapshot_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_network_contract_keeps_resource_axis_and_markov_parameters(pipeline):
    request, snapshot = wireless_request(pipeline)
    network = request['scenario']['network']
    network['resources'].append(resource('other', -3, 2e9))
    network['endpoint_limits']['tx']['resource_ids'].append('other')
    network['communication_allocations'][0]['resource_ids'].append('other')
    config = dict(states=[dict(active=[True, False], power_w=[1., 0.], channel_id=[7, -3]),
                          dict(active=[False, True], power_w=[0., 1.], channel_id=[-3, 7])],
                  transition_matrix=[[0., 1.], [1., 0.]], dwell_ticks=4, initial_state=0)
    network['background_emitters'][0].update(behavior='markov', config=config)
    package = next(x for x in snapshot.resources if x.kind == 'training_package')
    package.metadata['radio_models']['oracle']['input_fields'].append(dict(
        path=['network', 'resources', 'frequency_hz'], phase='post_physics',
        tensor=dict(shape=['F'], dtype='float64', units='Hz', frame='none', rate='radio_epoch')))
    exp = resolve(request, snapshot)
    contract = learning_contract(exp)
    assert contract['model_bindings']['radio']['input_fields'][-1]['tensor']['shape'] == [2]
    other = deepcopy(request)
    other['topology']['world_count'] = 9
    assert learning_contract(resolve(other, snapshot)) == contract
    other = deepcopy(request)
    other['scenario']['network']['resources'].reverse()
    assert learning_contract(resolve(other, snapshot)) != contract
    other = deepcopy(request)
    other['scenario']['network']['background_emitters'][0]['config']['transition_matrix'] = [[.5, .5], [1., 0.]]
    assert learning_contract(resolve(other, snapshot)) != contract


def test_unused_model_config_still_rejects_unknown_parameters(pipeline):
    request, snapshot = wireless_request(pipeline, service=False)
    request['model_config'] = {'radio': {'unknown': 1.}}
    rejected(request, snapshot, 'unknown')


def test_disjoint_service_resources_and_builtin_model_reference(pipeline):
    from muse_sdk.descriptor import RadioDescriptor
    request, snapshot = wireless_request(pipeline)
    network = request['scenario']['network']
    network['resources'].append(resource('other', -3, 2e9))
    network['endpoint_limits']['tx']['resource_ids'].append('other')
    network['communication_allocations'][0]['resource_ids'].append('other')
    network['services'].append(dict(service_id='second_payload', link_id='up', resource_ids=['other'], initial_bits=11.))
    package = next(x for x in snapshot.resources if x.kind == 'training_package')
    snapshot.builtins.descriptor.radio_models['oracle'] = RadioDescriptor.model_validate(package.metadata['radio_models']['oracle'])
    package.metadata['compositions']['default']['model_bindings']['radio'] = {
        'default': 'builtin:oracle', 'options': {'builtin:oracle': {}}}
    request['model_config'] = {'radio': {'scale': 3.}}
    exp = resolve(request, snapshot)
    assert [s.resource_indices for s in exp.network.services] == [(0,), (1,)]
    assert exp.model_bindings['radio'].reference == 'builtin:oracle'
    assert exp.model_bindings['radio'].config == {'scale': 3.}


def test_noncontinuous_slots_do_not_require_learning_rf_owners(pipeline):
    request, snapshot = wireless_request(pipeline)
    group = next(g for g in request['scenario']['entity_groups'] if g['group_id'] == 'guard_group')
    group['count'] = 3
    placements = request['scenario']['placements'][group['placement_id']]
    placements.append(dict(position=[8., 2., 1.], orientation_wxyz=[1., 0., 0., 0.]))
    request['role_config']['guard']['slots'] = [2, 0, 1]
    request['scenario']['command_targets_world']['guard'].append([1., 1., 1.])
    request['scenario']['radio_geometry']['endpoints']['tx']['slots'] = [2, 0]
    exp = resolve(request, snapshot)
    assert [point.slot for point in exp.network.endpoints[:2]] == [2, 0]
    assert exp.action_specs['guard'].radio is None
    assert exp.bindings['guard'].role_slot_to_group_slot == [2, 0, 1]


def test_fixed_source_to_moving_receiver_has_no_upload_direction_branch(pipeline):
    request, snapshot = wireless_request(pipeline)
    request['scenario']['radio_geometry']['endpoints'] = {
        'tx': dict(group_id='anchor_group', slots=[1, 0], device_id='ideal_point_radio'),
        'rx': dict(group_id='guard_group', slots=[1], device_id='ideal_point_radio')}
    exp = resolve(request, snapshot)
    assert [p.group_id for p in exp.network.endpoints] == ['anchor_group', 'anchor_group', 'guard_group']
    assert [(allocation.tx, allocation.rx) for allocation in exp.network.communication_allocations] == [(0, 2), (1, 2)]


def test_only_active_geometry_consumers_create_query_activation(pipeline):
    request, snapshot = wireless_request(pipeline)
    exp = resolve(request, snapshot)
    assert exp.geometry is not None and 'geometry' not in exp.activation
    package = next(x for x in snapshot.resources if x.kind == 'training_package')
    package.metadata['radio_models']['oracle']['input_fields'].append(dict(
        path=['network', 'geometry', 'distance_m'], phase='post_physics', tensor=dict(
            shape=['B', 'P'], dtype='float32', units='m', frame='none', rate='radio_epoch')))
    exp = resolve(request, snapshot)
    assert exp.activation['geometry'] == ['model:radio']
    request['scenario']['network']['services'] = []
    exp = resolve(request, snapshot)
    assert exp.model_bindings == {} and 'geometry' not in exp.activation
    package.metadata['terms']['geometry_score'] = dict(term_kind='reward', scope='world',
        config_schema=dict(type='object', properties={}, additionalProperties=False), input_fields=[dict(
            path=['network', 'geometry', 'sample_tick'], phase='post_physics',
            tensor=dict(shape=['B'], dtype='int64', units='tick', frame='none', rate='radio_epoch'))],
        output_spec=dict(shape=['B'], dtype='float32', units='1', frame='none', rate='decision'))
    package.metadata['compositions']['default']['world_reward_terms'] = [dict(selection_id='tick', term='geometry_score')]
    exp = resolve(request, snapshot)
    assert exp.activation['geometry'] == ['world.reward.tick']
    assert exp.model_bindings == {}
