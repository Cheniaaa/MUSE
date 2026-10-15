"""理想 IMU 的 CPU 输入、激活与 checkpoint 语义；不创建 GPU 环境。"""
import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest

from muse_server.resolver import resolve_experiment


SIGNALS = ('gyro', 'specific_force')
TERMS = ('imu_gyro', 'imu_specific_force', 'imu_valid', 'imu_age')


def imu_case(pipeline, *, enabled=True, profile=True, shared=False):
    catalog, source = pipeline
    request = copy.deepcopy(source)
    snapshot = catalog.snapshot()
    record = next(r for r in snapshot.resources if r.resource_id == source['training_package_id'])
    for role in record.metadata['composition']['agent_roles'].values():
        role['observation_terms'].extend(
            {'selection_id': name, 'term': 'builtin:' + name, 'default_enabled': enabled}
            for name in TERMS)
    if profile:
        for group in request['scenario']['entity_groups']:
            group.update(sensor_profile_id='cf2_imu', sensor_config={'update_rate_hz': 100.})
    if shared:
        group = request['scenario']['entity_groups'][0]
        group['count'] = 3
        request['scenario']['entity_groups'] = [group]
        request['scenario']['placements'][group['placement_id']] *= 3
        request['role_config']['scout'] = {'group_id': group['group_id'], 'slots': [1]}
        request['role_config']['guard'] = {'group_id': group['group_id'], 'slots': [2, 0]}
    return request, snapshot, record


def resolved(request, snapshot):
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors
    return report.resolved_experiment


def sensor_input(group=None, *, signal='gyro', field='data'):
    width = [3] if field == 'data' else []
    return {'path': ['sensor_outputs', group or {'binding': 'group_id'}, signal, field],
            'phase': 'post_physics', 'tensor': {
                'shape': ['B', 'N_group', *width],
                'dtype': 'bool' if field == 'valid' else 'int64' if field == 'sample_tick' else 'float32',
                'units': 'rad/s' if field == 'data' else 's' if field == 'age' else '1',
                'frame': 'imu_flu' if field == 'data' else 'none', 'rate': 'decision'}}


def add_task(request, record, inputs):
    record.metadata['tasks']['imu_reader'] = {
        'task_id': 'imu_reader', 'config_schema': {'type': 'object', 'properties': {}, 'additionalProperties': False},
        'input_fields': inputs, 'published_fields': {}, 'role_status_bindings': {}, 'learning_done_roles': []}
    record.metadata['composition']['task'] = 'imu_reader'
    request['task_id'] = 'imu_reader'


def test_profile_freezes_integer_grid_and_native_output_contract(pipeline):
    request, snapshot, _ = imu_case(pipeline)
    original = copy.deepcopy(request)
    exp = resolved(request, snapshot)
    assert request == original
    for group in exp.scenario.entity_groups:
        profile = exp.sensors[group.group_id]
        assert profile['sensor_profile_id'] == 'cf2_imu'
        assert 'mount' not in profile and 'period_ticks' not in profile
        assert profile['config'] == group.sensor_config == {'update_rate_hz': 100.}
        assert profile['outputs'] == {
            'gyro': {'sensor_name': 'body_gyro', 'units': 'rad/s', 'frame': 'imu_flu',
                     'width': 3, 'sensor_type': 'gyro', 'mount': 'imu',
                     'rate_key': 'update_rate_hz', 'period_ticks': 5},
            'specific_force': {'sensor_name': 'body_linacc', 'units': 'm/s²', 'frame': 'imu_flu',
                               'width': 3, 'sensor_type': 'accelerometer', 'mount': 'imu',
                               'rate_key': 'update_rate_hz', 'period_ticks': 5}}
        assert exp.activation['sensor:' + group.group_id] == [
            f'roles.{role}.observation.{name}'
            for role, binding in exp.bindings.items() if binding.group_id == group.group_id for name in TERMS]


@pytest.mark.parametrize('rate', [0., -1., float('nan'), float('inf'), -float('inf'), 501., 120.,
                                  100.0001, '100', True, 1e-320])
def test_invalid_rates_rejected_at_cpu_boundary(pipeline, rate):
    request, snapshot, _ = imu_case(pipeline)
    request['scenario']['entity_groups'][0]['sensor_config']['update_rate_hz'] = rate
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert any('sensor_config' in error['field'] for error in report.errors), report.errors


def test_oversized_json_integer_rate_returns_field_error(pipeline):
    request, snapshot, _ = imu_case(pipeline)
    request['scenario']['entity_groups'][0]['sensor_config']['update_rate_hz'] = 10**400
    report = resolve_experiment(json.loads(json.dumps(request)), snapshot)
    assert not report.valid and report.resolved_experiment is None
    assert any(error['field'] == 'scenario.entity_groups[0].sensor_config.update_rate_hz'
               for error in report.errors), report.errors


def test_period_exceeding_int64_rejected_at_cpu_boundary(pipeline):
    request, snapshot, _ = imu_case(pipeline)
    request['scenario']['entity_groups'][0]['sensor_config']['update_rate_hz'] = 1e-18
    report = resolve_experiment(request, snapshot)
    assert not report.valid and report.resolved_experiment is None
    assert any(error['field'] == 'scenario.entity_groups[0].sensor_config.update_rate_hz'
               for error in report.errors), report.errors


@pytest.mark.parametrize('config', [{}, {'update_rate_hz': 100., 'noise': .1},
                                   {'update_rate_hz': 100., 'latency': .1},
                                   {'update_rate_hz': 100., 'outputs': {}}])
def test_profile_requires_rate_and_forbids_unimplemented_fields(pipeline, config):
    request, snapshot, _ = imu_case(pipeline)
    request['scenario']['entity_groups'][0]['sensor_config'] = config
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert any('sensor_config' in error['field'] for error in report.errors), report.errors


def test_none_rejects_nonempty_config(pipeline):
    catalog, source = pipeline
    request = copy.deepcopy(source)
    request['scenario']['entity_groups'][0]['sensor_config'] = {'update_rate_hz': 100.}
    report = resolve_experiment(request, catalog.snapshot())
    assert not report.valid
    assert any('sensor_config' in error['field'] for error in report.errors), report.errors


@pytest.mark.parametrize('model,controller', [('ugv_diff_drive', 'diff_drive_controller'),
                                           ('radio_base_station', 'none')])
def test_cf2_profile_rejected_for_other_models(pipeline, model, controller):
    request, snapshot, _ = imu_case(pipeline)
    request['scenario']['entity_groups'][0].update(model_id=model, controller_id=controller)
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert any('sensor_profile_id' in error['field'] for error in report.errors), report.errors


def test_group_binding_uses_group_shape_not_split_or_reordered_role_shape(pipeline):
    request, snapshot, _ = imu_case(pipeline, shared=True)
    exp = resolved(request, snapshot)
    assert exp.bindings['scout'].role_slot_to_group_slot == [1]
    assert exp.bindings['guard'].role_slot_to_group_slot == [2, 0]
    for selection in exp.selections:
        if selection.selection_id not in TERMS:
            continue
        count = exp.bindings[selection.binding_role_id].role_slot_count
        assert selection.output_spec.shape == [2, count, 3 if selection.selection_id in TERMS[:2] else 2]
        assert all(item.path[:2] == ['sensor_outputs', 'scout_group'] for item in selection.input_fields)
        assert all(item.tensor.shape[:2] == [2, 3] for item in selection.input_fields)
    assert len(exp.activation['sensor:scout_group']) == 8
    assert 'sensor:guard_group' not in exp.activation


def test_task_only_consumer_binds_concrete_group_and_sample_tick(pipeline):
    request, snapshot, record = imu_case(pipeline, enabled=False, shared=True)
    add_task(request, record, [sensor_input('scout_group'), sensor_input('scout_group', field='sample_tick')])
    exp = resolved(request, snapshot)
    assert exp.task_spec.input_fields[0].tensor.shape == [2, 3, 3]
    assert exp.task_spec.input_fields[1].tensor.shape == [2, 3]
    assert exp.activation['sensor:scout_group'] == ['task:imu_reader']


def test_unselected_task_does_not_activate_sensor(pipeline):
    request, snapshot, record = imu_case(pipeline, enabled=False)
    add_task(request, record, [sensor_input('scout_group')])
    record.metadata['composition']['task'] = None
    request['task_id'] = None
    exp = resolved(request, snapshot)
    assert not any(key.startswith('sensor:') for key in exp.activation)


def test_abstract_group_binding_requires_role_context(pipeline):
    request, snapshot, record = imu_case(pipeline, enabled=False)
    add_task(request, record, [sensor_input()])
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert any('binding' in error['message'].lower() for error in report.errors), report.errors


@pytest.mark.parametrize('enabled,profile,valid', [(True, False, False), (False, False, True),
                                               (False, True, True)])
def test_only_enabled_consumers_require_and_activate_profile(pipeline, enabled, profile, valid):
    request, snapshot, _ = imu_case(pipeline, enabled=enabled, profile=profile)
    report = resolve_experiment(request, snapshot)
    assert report.valid is valid, report.errors
    if valid:
        assert not any(key.startswith('sensor:') for key in report.resolved_experiment.activation)
    else:
        assert any('Unavailable public field' in error['message'] for error in report.errors)


@pytest.mark.parametrize('owner', ['role', 'algorithm'])
def test_algorithm_abi_rejects_raw_sensor_requirements(pipeline, owner):
    from muse_sdk.descriptor import InputField
    request, snapshot, record = imu_case(pipeline, enabled=False)
    if owner == 'role':
        record.metadata['composition']['agent_roles']['scout']['input_requirements'] = [sensor_input()]
    else:
        snapshot.builtins.descriptor.algorithms['constant_velocity'].observation_requirements = [
            InputField.model_validate(sensor_input('scout_group'))]
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert any('observation Term' in error['message'] for error in report.errors), report.errors


@pytest.mark.parametrize('mutation', ['period', 'dt', 'decimation', 'native', 'units', 'frame'])
def test_learning_contract_distinguishes_active_sampling_semantics(pipeline, mutation):
    from muse_sdk.experiment import learning_contract
    request, snapshot, _ = imu_case(pipeline)
    exp = resolved(request, snapshot)
    if mutation == 'period':
        request['scenario']['entity_groups'][0]['sensor_config']['update_rate_hz'] = 50.
    elif mutation == 'dt':
        request['scenario']['scene']['physics_dt'] = .001
    elif mutation == 'decimation':
        request['scenario']['scene']['control_substeps'] = 10
    else:
        key = {'native': 'sensor_name', 'units': 'units', 'frame': 'frame'}[mutation]
        changed = exp.model_copy(deep=True)
        changed.sensors['scout_group']['outputs']['gyro'][key] = 'changed'
        assert learning_contract(exp) != learning_contract(changed)
        return
    assert learning_contract(exp) != learning_contract(resolved(request, snapshot))


def test_contract_projects_active_groups_without_raw_rate_or_batch(pipeline):
    from muse_sdk.experiment import learning_contract
    request, snapshot, record = imu_case(pipeline, enabled=False)
    record.metadata['composition']['agent_roles']['scout']['observation_terms'][-4]['default_enabled'] = True
    first = resolved(request, snapshot)
    request['topology']['world_count'] = 1
    request['scenario']['entity_groups'][0]['sensor_config']['update_rate_hz'] = 100. * (1 + 1e-10)
    request['scenario']['entity_groups'][1]['sensor_config']['update_rate_hz'] = 50.
    second = resolved(request, snapshot)
    contract = learning_contract(first)
    assert contract == learning_contract(second)
    assert contract['sensors'] == {
        'physics_dt': .002, 'control_substeps': 20,
        'groups': [{'group_id': 'scout_group', 'sensor_profile_id': 'cf2_imu',
                    'outputs': first.sensors['scout_group']['outputs']}]}
    assert first.scenario.entity_groups[0].sensor_config == second.scenario.entity_groups[0].sensor_config
    assert first.sensors['scout_group'] == second.sensors['scout_group']


def test_no_active_sensor_contract_is_empty_despite_profile_config_change(pipeline):
    from muse_sdk.experiment import learning_contract
    request, snapshot, _ = imu_case(pipeline, enabled=False)
    first = resolved(request, snapshot)
    request['scenario']['entity_groups'][0]['sensor_config']['update_rate_hz'] = 50.
    second = resolved(request, snapshot)
    assert learning_contract(first) == learning_contract(second)
    assert learning_contract(first)['sensors'] == {}


def test_canonical_rates_pass_real_catalog_and_worker_checkpoint_checks(pipeline):
    from muse_runtime.checkpoint import load_checkpoint, publish_checkpoint
    catalog, _ = pipeline
    request, snapshot, _ = imu_case(pipeline)
    first = resolved(request, snapshot)
    identity = catalog.save_experiment(first)
    plan = catalog.create_run_plan(identity, mode='EVALUATE', device='cpu')
    artifact = publish_checkpoint(plan, SimpleNamespace(learning_state_dict=lambda: {'updates': 7}), steps=4, updates=7)
    record = {'run_id': plan.run_id, 'artifact_directory': plan.artifact_directory,
              'plan_path': str(Path(plan.artifact_directory) / 'resolved_run_plan.json'),
              'artifacts': [artifact.model_dump(mode='json')]}
    with sqlite3.connect(catalog.database) as db:
        db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, record TEXT NOT NULL)')
        db.execute('INSERT INTO runs VALUES (?, ?)', (plan.run_id, json.dumps(record)))
    request['topology']['world_count'] = 1
    request['scenario']['entity_groups'][0]['sensor_config']['update_rate_hz'] *= 1 + 1e-10
    equivalent = resolved(request, snapshot)
    reference = catalog.resolve_checkpoint(artifact.id, equivalent)
    loaded = []
    load_checkpoint(plan.model_copy(update={'experiment': equivalent, 'checkpoint': reference}),
                    SimpleNamespace(load_learning_state_dict=loaded.append))
    assert loaded == [{'updates': 7}]


def test_inactive_config_remains_subject_to_catalog_resource_comparison(pipeline):
    from muse_runtime.checkpoint import publish_checkpoint
    from muse_sdk.experiment import learning_contract
    catalog, _ = pipeline
    request, snapshot, _ = imu_case(pipeline, enabled=False)
    first = resolved(request, snapshot)
    plan = catalog.create_run_plan(catalog.save_experiment(first), mode='EVALUATE', device='cpu')
    artifact = publish_checkpoint(plan, SimpleNamespace(learning_state_dict=lambda: {}), steps=0, updates=0)
    record = {'run_id': plan.run_id, 'artifact_directory': plan.artifact_directory,
              'plan_path': str(Path(plan.artifact_directory) / 'resolved_run_plan.json'),
              'artifacts': [artifact.model_dump(mode='json')]}
    with sqlite3.connect(catalog.database) as db:
        db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, record TEXT NOT NULL)')
        db.execute('INSERT INTO runs VALUES (?, ?)', (plan.run_id, json.dumps(record)))
    request['scenario']['entity_groups'][0]['sensor_config']['update_rate_hz'] = 50.
    changed = resolved(request, snapshot)
    assert learning_contract(first) == learning_contract(changed)
    with pytest.raises(ValueError, match='Incompatible checkpoint scenario'):
        catalog.resolve_checkpoint(artifact.id, changed)


def test_active_imu_resolution_and_sdk_container_have_no_gpu_imports(pipeline, tmp_path):
    request, snapshot, _ = imu_case(pipeline)
    source = tmp_path / 'input.json'
    source.write_text(json.dumps({'request': request, 'snapshot': snapshot.model_dump(mode='json')}))
    script = '''
import importlib.abc, json, sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','mujoco','mjlab','warp','muse_runtime','muse_builtins'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from muse_sdk.runtime_types import SensorSample
from muse_sdk.experiment import CatalogSnapshot
from muse_server.resolver import resolve_experiment
data = json.loads(Path(sys.argv[1]).read_text())
snapshot = CatalogSnapshot.model_validate_json(json.dumps(data['snapshot']))
report = resolve_experiment(data['request'], snapshot)
assert report.valid, report.errors
assert 'sensor:scout_group' in report.resolved_experiment.activation
assert SensorSample(1, 2, 3, 4).sample_tick == 4
'''
    result = subprocess.run([sys.executable, '-c', script, str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
