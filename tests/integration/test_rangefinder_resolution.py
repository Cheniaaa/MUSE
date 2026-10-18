"""组合测距 profile 的 CPU 输入、逐设备时序与学习契约；不创建 GPU 环境。"""
import copy
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from muse_sdk.experiment import learning_contract
from muse_server.resolver import resolve_experiment

from test_imu_resolution import imu_case, resolved


TERMS = ('rangefinder_distance', 'rangefinder_valid', 'rangefinder_age')


def range_case(pipeline, *, enabled=True, profile=True, shared=False):
    request, snapshot, record = imu_case(pipeline, enabled=False, profile=False, shared=shared)
    for role in record.metadata['compositions']['default']['agent_roles'].values():
        role['observation_terms'].extend(
            {'selection_id': name, 'term': 'builtin:' + name, 'default_enabled': enabled}
            for name in TERMS)
    if profile:
        for group in request['scenario']['entity_groups']:
            group.update(sensor_profile_id='cf2_imu_rangefinder_down', sensor_config={
                'imu_update_rate_hz': 100., 'rangefinder_update_rate_hz': 5.})
    return request, snapshot, record


def range_input(group, field='data'):
    return {'path': ['sensor_outputs', group, 'range', field], 'phase': 'post_physics',
            'tensor': {'shape': ['B', 'N_group', *([1] if field == 'data' else [])],
                       'dtype': 'int64' if field == 'sample_tick' else 'float32',
                       'units': 'm' if field == 'data' else '1', 'frame': 'none', 'rate': 'decision'}}


def test_combined_profile_freezes_nominal_devices_and_independent_periods(pipeline):
    request, snapshot, record = range_case(pipeline)
    for role in record.metadata['compositions']['default']['agent_roles'].values():
        for selection in role['observation_terms']:
            if selection['selection_id'] in ('imu_gyro', 'imu_specific_force'):
                selection['default_enabled'] = True
    original = copy.deepcopy(request)
    exp = resolved(request, snapshot)
    assert request == original
    for group in exp.scenario.entity_groups:
        profile = exp.sensors[group.group_id]
        assert profile['sensor_profile_id'] == 'cf2_imu_rangefinder_down'
        assert 'mount' not in profile and 'period_ticks' not in profile
        assert profile['config'] == group.sensor_config == {
            'imu_update_rate_hz': 100., 'rangefinder_update_rate_hz': 5.}
        assert profile['outputs'] == {
            'gyro': {'sensor_name': 'body_gyro', 'units': 'rad/s', 'frame': 'imu_flu',
                     'width': 3, 'sensor_type': 'gyro', 'mount': 'imu',
                     'rate_key': 'imu_update_rate_hz', 'period_ticks': 5},
            'specific_force': {'sensor_name': 'body_linacc', 'units': 'm/s²', 'frame': 'imu_flu',
                               'width': 3, 'sensor_type': 'accelerometer', 'mount': 'imu',
                               'rate_key': 'imu_update_rate_hz', 'period_ticks': 5},
            'range': {'sensor_name': 'rangefinder_down', 'units': 'm', 'frame': 'none',
                      'width': 1, 'sensor_type': 'rangefinder', 'mount': 'rangefinder_down',
                      'rate_key': 'rangefinder_update_rate_hz', 'period_ticks': 100,
                      'position': [0., 0., -.01], 'orientation_wxyz': [0., 1., 0., 0.],
                      'min_range_m': .04, 'max_range_m': 4., 'minimum_period_s': .144}}
    for selection in exp.selections:
        if selection.enabled and selection.selection_id in ('imu_gyro', 'imu_specific_force', *TERMS):
            width = 3 if selection.selection_id.startswith('imu_') else 1
            count = exp.bindings[selection.binding_role_id].role_slot_count
            assert selection.output_spec.shape == [2, count, width]
            if selection.selection_id in ('imu_gyro', 'imu_specific_force', 'rangefinder_distance'):
                assert selection.input_fields[0].tensor.shape == [2, count, width]


@pytest.mark.parametrize('rate_key', ['imu_update_rate_hz', 'rangefinder_update_rate_hz'])
@pytest.mark.parametrize('rate', [0., -1., float('nan'), float('inf'), -float('inf'),
                                  501., 40., 120., 100.0001, '5', True, 1e-320, 1e-18, 10**400])
def test_each_rate_uses_cpu_finite_positive_grid_and_int64_boundary(pipeline, rate_key, rate):
    request, snapshot, _ = range_case(pipeline)
    request['scenario']['entity_groups'][0]['sensor_config'][rate_key] = rate
    report = resolve_experiment(request, snapshot)
    assert not report.valid and report.resolved_experiment is None
    assert any('sensor_config.' + rate_key in error['field'] for error in report.errors), report.errors


@pytest.mark.parametrize('period_s', [.144, .144 * (1 + 1e-10), .144 * (1 - 1e-10), .142, .01])
def test_range_budget_rejects_canonical_boundary_even_with_grid_tolerance(pipeline, period_s):
    request, snapshot, _ = range_case(pipeline)
    request['scenario']['entity_groups'][0]['sensor_config']['rangefinder_update_rate_hz'] = 1 / period_s
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert any(error['field'] == 'scenario.entity_groups[0].sensor_config.rangefinder_update_rate_hz'
               and 'period' in error['message'].lower() for error in report.errors), report.errors


@pytest.mark.parametrize('period_s,period_ticks,valid', [(.144, 48, False), (.147, 49, True)])
def test_range_budget_compares_declared_time_exactly_on_three_ms_grid(pipeline, period_s, period_ticks, valid):
    request, snapshot, _ = range_case(pipeline)
    request['scenario']['scene']['physics_dt'] = .003
    for group in request['scenario']['entity_groups']:
        group['sensor_config'] = {
            'imu_update_rate_hz': 1 / .003, 'rangefinder_update_rate_hz': 1 / period_s}
    report = resolve_experiment(request, snapshot)
    assert report.valid is valid, report.errors
    if valid:
        for profile in report.resolved_experiment.sensors.values():
            outputs = profile['outputs']
            assert outputs['gyro']['period_ticks'] == outputs['specific_force']['period_ticks'] == 1
            assert outputs['range']['period_ticks'] == period_ticks
    else:
        assert report.resolved_experiment is None
        assert any(error['field'] == 'scenario.entity_groups[0].sensor_config.rangefinder_update_rate_hz'
                   and 'period' in error['message'].lower() for error in report.errors), report.errors


def test_shortest_hardware_legal_grid_is_73_ticks_and_imu_remains_independent(pipeline):
    request, snapshot, _ = range_case(pipeline)
    request['scenario']['entity_groups'][0]['sensor_config'].update(
        imu_update_rate_hz=50., rangefinder_update_rate_hz=500 / 73)
    exp = resolved(request, snapshot)
    outputs = exp.sensors['scout_group']['outputs']
    assert [outputs[name]['period_ticks'] for name in ('gyro', 'specific_force', 'range')] == [10, 10, 73]
    assert exp.sensors['guard_group']['outputs']['range']['period_ticks'] == 100


@pytest.mark.parametrize('key,value', [
    ('mount', 'custom'), ('position', [0., 0., 0.]), ('orientation_wxyz', [1., 0., 0., 0.]),
    ('min_range_m', .1), ('max_range_m', 10.), ('width', 2), ('outputs', {}),
    ('minimum_period_s', 0.), ('update_rate_hz', 100.)])
def test_device_definition_cannot_be_overridden_by_users(pipeline, key, value):
    request, snapshot, _ = range_case(pipeline)
    request['scenario']['entity_groups'][0]['sensor_config'][key] = value
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert any('sensor_config' in error['field'] for error in report.errors), report.errors


@pytest.mark.parametrize('missing', ['imu_update_rate_hz', 'rangefinder_update_rate_hz'])
def test_both_device_rates_are_required(pipeline, missing):
    request, snapshot, _ = range_case(pipeline)
    del request['scenario']['entity_groups'][0]['sensor_config'][missing]
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert any('sensor_config' in error['field'] for error in report.errors), report.errors


@pytest.mark.parametrize('model,controller', [('ugv_diff_drive', 'diff_drive_controller'),
                                           ('radio_base_station', 'none')])
def test_combined_profile_rejects_incompatible_models(pipeline, model, controller):
    request, snapshot, _ = range_case(pipeline)
    request['scenario']['entity_groups'][0].update(model_id=model, controller_id=controller)
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert any('sensor_profile_id' in error['field'] for error in report.errors), report.errors


@pytest.mark.parametrize('enabled,profile,valid', [(True, False, False), (False, False, True),
                                                (False, True, True)])
def test_only_enabled_consumers_require_and_activate_profile(pipeline, enabled, profile, valid):
    request, snapshot, _ = range_case(pipeline, enabled=enabled, profile=profile)
    report = resolve_experiment(request, snapshot)
    assert report.valid is valid, report.errors
    if valid:
        assert not any(key.startswith('sensor:') for key in report.resolved_experiment.activation)
    else:
        assert any('Unavailable public field' in error['message'] for error in report.errors)


def test_range_consumer_requires_range_not_just_imu_profile(pipeline):
    request, snapshot, _ = range_case(pipeline)
    for group in request['scenario']['entity_groups']:
        group.update(sensor_profile_id='cf2_imu', sensor_config={'update_rate_hz': 100.})
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert any('Unavailable public field' in error['message'] for error in report.errors)


def test_split_reordered_roles_keep_group_input_width_and_scalar_role_outputs(pipeline):
    request, snapshot, _ = range_case(pipeline, shared=True)
    exp = resolved(request, snapshot)
    assert exp.bindings['scout'].role_slot_to_group_slot == [1]
    assert exp.bindings['guard'].role_slot_to_group_slot == [2, 0]
    for selection in exp.selections:
        if selection.selection_id in TERMS:
            count = exp.bindings[selection.binding_role_id].role_slot_count
            assert selection.output_spec.shape == [2, count, 1]
            item = selection.input_fields[0]
            assert item.path[:3] == ['sensor_outputs', 'scout_group', 'range']
            assert item.tensor.shape == [2, 3, *([1] if selection.selection_id == TERMS[0] else [])]
    assert len(exp.activation['sensor:scout_group']) == 6
    assert 'sensor:guard_group' not in exp.activation


def test_task_only_range_consumer_activates_whole_profile(pipeline):
    request, snapshot, record = range_case(pipeline, enabled=False, shared=True)
    record.metadata['tasks']['range_reader'] = {
        'task_id': 'range_reader', 'config_schema': {'type': 'object', 'properties': {}, 'additionalProperties': False},
        'input_fields': [range_input('scout_group'), range_input('scout_group', 'sample_tick')],
        'published_fields': {}, 'role_status_bindings': {}, 'learning_done_roles': []}
    record.metadata['compositions']['default']['task'] = 'range_reader'
    exp = resolved(request, snapshot)
    assert [item.tensor.shape for item in exp.task_spec.input_fields] == [[2, 3, 1], [2, 3]]
    assert exp.activation['sensor:scout_group'] == ['task:range_reader']
    contract = learning_contract(exp)
    assert set(contract['sensors']['groups'][0]['outputs']) == {'gyro', 'specific_force', 'range'}


@pytest.mark.parametrize('signal,key,value', [
    ('range', 'mount', 'other'), ('range', 'position', [0., 0., -.02]),
    ('range', 'orientation_wxyz', [1., 0., 0., 0.]), ('range', 'width', 2),
    ('range', 'min_range_m', .1), ('range', 'max_range_m', 3.), ('range', 'minimum_period_s', .14),
    ('range', 'sensor_name', 'other'), ('range', 'sensor_type', 'other'),
    ('range', 'units', 'other'), ('range', 'frame', 'other'), ('range', 'rate_key', 'other'),
    ('range', 'period_ticks', 73), ('gyro', 'period_ticks', 10), ('specific_force', 'period_ticks', 10)])
def test_learning_contract_freezes_every_output_definition_and_period(pipeline, signal, key, value):
    request, snapshot, _ = range_case(pipeline)
    exp = resolved(request, snapshot)
    changed = exp.model_copy(deep=True)
    changed.sensors['scout_group']['outputs'][signal][key] = value
    assert learning_contract(exp) != learning_contract(changed)


def test_contract_keeps_only_active_devices_and_canonical_periods_not_batch_or_raw_rate(pipeline):
    request, snapshot, record = range_case(pipeline, enabled=False)
    record.metadata['compositions']['default']['agent_roles']['scout']['observation_terms'][-3]['default_enabled'] = True
    first = resolved(request, snapshot)
    request['topology']['world_count'] = 1
    request['scenario']['entity_groups'][0]['sensor_config'] = {
        'imu_update_rate_hz': 100. * (1 + 1e-10), 'rangefinder_update_rate_hz': 5. * (1 - 1e-10)}
    request['scenario']['entity_groups'][1]['sensor_config']['rangefinder_update_rate_hz'] = 2.
    second = resolved(request, snapshot)
    contract = learning_contract(first)
    assert contract == learning_contract(second)
    assert contract['sensors'] == {'physics_dt': .002, 'control_substeps': 20, 'groups': [{
        'group_id': 'scout_group', 'sensor_profile_id': 'cf2_imu_rangefinder_down',
        'outputs': first.sensors['scout_group']['outputs']}]}
    assert first.scenario.entity_groups[0].sensor_config == second.scenario.entity_groups[0].sensor_config
    assert first.sensors['scout_group'] == second.sensors['scout_group']


def test_no_consumer_has_no_learning_sensor_contract(pipeline):
    request, snapshot, _ = range_case(pipeline, enabled=False)
    first = resolved(request, snapshot)
    request['scenario']['entity_groups'][0]['sensor_config']['rangefinder_update_rate_hz'] = 2.
    assert learning_contract(first) == learning_contract(resolved(request, snapshot))
    assert learning_contract(first)['sensors'] == {}


def test_real_cpu_checkpoint_accepts_equivalent_rates_and_rejects_changed_device_semantics(pipeline):
    from muse_runtime.checkpoint import load_checkpoint, publish_checkpoint
    catalog, _ = pipeline
    request, snapshot, _ = range_case(pipeline)
    first = resolved(request, snapshot)
    plan = catalog.create_run_plan(catalog.save_experiment(first), mode='EVALUATE', device='cpu')
    artifact = publish_checkpoint(plan, SimpleNamespace(learning_state_dict=lambda: {'updates': 7}), steps=4, updates=7)
    record = {'run_id': plan.run_id, 'artifact_directory': plan.artifact_directory,
              'plan_path': str(Path(plan.artifact_directory) / 'resolved_run_plan.json'),
              'artifacts': [artifact.model_dump(mode='json')]}
    with sqlite3.connect(catalog.database) as db:
        db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, record TEXT NOT NULL)')
        db.execute('INSERT INTO runs VALUES (?, ?)', (plan.run_id, json.dumps(record)))
    request['topology']['world_count'] = 1
    for key in ('imu_update_rate_hz', 'rangefinder_update_rate_hz'):
        request['scenario']['entity_groups'][0]['sensor_config'][key] *= 1 + 1e-10
    equivalent = resolved(request, snapshot)
    reference = catalog.resolve_checkpoint(artifact.id, equivalent)
    loaded = []
    load_checkpoint(plan.model_copy(update={'experiment': equivalent, 'checkpoint': reference}),
                    SimpleNamespace(load_learning_state_dict=loaded.append))
    assert loaded == [{'updates': 7}]
    for signal, key, value in [('range', 'position', [0., 0., -.02]), ('range', 'width', 2),
                               ('range', 'max_range_m', 3.), ('range', 'period_ticks', 73),
                               ('gyro', 'period_ticks', 10)]:
        changed = equivalent.model_copy(deep=True)
        changed.sensors['scout_group']['outputs'][signal][key] = value
        with pytest.raises(ValueError, match='Incompatible checkpoint'):
            catalog.resolve_checkpoint(artifact.id, changed)
        with pytest.raises(ValueError, match='contract'):
            load_checkpoint(plan.model_copy(update={'experiment': changed, 'checkpoint': reference}),
                            SimpleNamespace(load_learning_state_dict=loaded.append))
    assert loaded == [{'updates': 7}]
