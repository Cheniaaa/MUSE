"""编辑检查与完整解析共享值来源，缺依赖时不制造可运行Experiment。"""
from copy import deepcopy

import pytest


def test_partial_inspection_preserves_missing_null_and_origins(pipeline):
    from muse_server.inspection import inspect_experiment
    catalog, request = pipeline
    snapshot = catalog.snapshot().model_copy(deep=True)
    algorithm = snapshot.builtins.descriptor.algorithms['constant_velocity']
    algorithm.config_schema = {'type': 'object', 'properties': {
        'motion': {'type': 'array', 'items': {'type': 'number'}},
        'label': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
        'channels': {'type': 'array', 'items': {'type': 'integer'}, 'default': [0]},
    }, 'required': ['motion', 'label'], 'additionalProperties': False}
    package = next(r for r in snapshot.resources if r.resource_id == request['training_package_id']
                   and r.revision == request['training_package_revision'])
    package.metadata['compositions']['default']['algorithm_config']['channels'] = [1, 2]
    partial = {key: request[key] for key in ('training_package_id', 'training_package_revision', 'composition_id')}
    partial['algorithm_config'] = {'channels': [3]}
    missing = inspect_experiment(partial, snapshot)
    section = next(s for s in missing.sections if s.path == ['algorithm_config'])
    assert missing.resolved_experiment is None
    assert section.effective['channels'] == [3]
    assert section.origins['/channels'] == 'override' and section.origins['/motion'] == 'package'
    assert 'label' not in section.effective and '/label' in section.missing
    partial['algorithm_config']['label'] = None
    present = inspect_experiment(partial, snapshot)
    section = next(s for s in present.sections if s.path == ['algorithm_config'])
    assert section.effective['label'] is None and '/label' not in section.missing
    partial['algorithm_config']['channels'] = ['unfinished']
    invalid = inspect_experiment(partial, snapshot)
    section = next(s for s in invalid.sections if s.path == ['algorithm_config'])
    assert section.effective['channels'] == ['unfinished']
    assert section.errors and invalid.resolved_experiment is None


def test_inspect_matches_complete_resolver(pipeline):
    from muse_server.inspection import inspect_experiment
    from muse_server.resolver import resolve_experiment
    catalog, source = pipeline
    request = deepcopy(source)
    request['term_config'] = {'roles': {
        'scout': {'reward': {'distance': {'config': {'scale': 3.}}}},
        'guard': {'reward': {'distance': {'config': {'scale': 7.}}}},
    }}
    original = deepcopy(request)
    snapshot = catalog.snapshot()
    checked = resolve_experiment(request, snapshot)
    inspected = inspect_experiment(request, snapshot)
    assert checked.valid, checked.errors
    assert inspected.resolved_experiment == checked.resolved_experiment
    assert not inspected.errors
    for section in inspected.sections:
        if section.path == ['algorithm_config']:
            assert section.effective == checked.resolved_experiment.algorithm_config
        if section.path == ['term_config', 'roles', 'scout', 'reward', 'distance', 'config']:
            assert section.effective == {'scale': 3.}
            assert section.origins == {'/scale': 'override'}
    assert request == original


def test_inspect_disabled_selection_keeps_partial_values(pipeline):
    from muse_server.inspection import inspect_experiment
    catalog, request = pipeline
    snapshot = catalog.snapshot()
    package = next(r for r in snapshot.resources if r.resource_id == request['training_package_id']
                   and r.revision == request['training_package_revision'])
    selection = package.metadata['compositions']['default']['agent_roles']['scout']['reward_terms'][0]
    selection.update(required=False, default_enabled=False, config={})
    partial = {key: request[key] for key in ('training_package_id', 'training_package_revision', 'composition_id')}
    partial['term_config'] = {'roles': {'scout': {'reward': {selection['selection_id']: {'config': {'scale': 7.}}}}}}
    inspected = inspect_experiment(partial, snapshot)
    section = next(s for s in inspected.sections if s.path[-2:] == [selection['selection_id'], 'config'])
    assert section.effective == {'scale': 7.} and not section.missing
    partial['term_config']['roles']['scout']['reward'][selection['selection_id']]['config']['scale'] = 'bad'
    invalid = inspect_experiment(partial, snapshot)
    section = next(s for s in invalid.sections if s.path[-2:] == [selection['selection_id'], 'config'])
    assert section.errors and section.effective['scale'] == 'bad'


@pytest.mark.parametrize('partial', [{}, {'scenario': {'entity_groups': []}}, {'algorithm_config': None}])
def test_inspect_incomplete_inputs_never_constructs_default_scene(pipeline, partial):
    from muse_server.inspection import inspect_experiment
    catalog, _ = pipeline
    result = inspect_experiment(partial, catalog.snapshot())
    assert result.resolved_experiment is None
    assert result.errors


@pytest.mark.parametrize('field,value', [('composition_id', []), ('training_package_id', {}),
                                       ('training_package_revision', True), ('model_choices', {'radio': []})])
def test_inspection_reports_malformed_partial_selectors(pipeline, field, value):
    from muse_server.inspection import inspect_experiment
    catalog, request = pipeline
    snapshot = catalog.snapshot()
    if field == 'model_choices':
        from wireless_helpers import wireless_request
        request, snapshot = wireless_request(pipeline)
    partial = {key: request[key] for key in ('training_package_id', 'training_package_revision', 'composition_id')}
    partial[field] = value
    result = inspect_experiment(partial, snapshot)
    assert result.resolved_experiment is None
    assert any(error['field'].startswith(field) for error in result.errors)


def test_inspection_sensor_effective_rate_matches_canonical_runtime_config(pipeline):
    from muse_server.inspection import inspect_experiment
    catalog, source = pipeline
    request = deepcopy(source)
    request['scenario']['entity_groups'][0].update(sensor_profile_id='cf2_imu',
                                                sensor_config={'update_rate_hz': 50.000000001})
    report = inspect_experiment(request, catalog.snapshot())
    assert report.resolved_experiment is not None, report.errors
    section = next(s for s in report.sections if s.path == ['scenario', 'entity_groups', 0, 'sensor_config'])
    assert section.effective == report.resolved_experiment.sensors['scout_group']['config']
    assert section.origins['/update_rate_hz'] == 'scenario'


def test_inspection_and_schema_http_are_cpu_catalog_views(pipeline):
    from fastapi.testclient import TestClient
    from muse_server.app import create_app
    catalog, request = pipeline
    with TestClient(create_app(catalog.directory)) as client:
        schema = client.get('/catalog/schema')
        assert schema.status_code == 200
        schemas = schema.json()
        assert schemas['api_revision'] == 2
        assert schemas['experiment_request']['additionalProperties'] is False
        assert 'composition_id' in schemas['experiment_request']['required']
        assert 'task_id' not in schemas['experiment_request']['properties']
        inspected = client.post('/experiments/inspect', json=request)
        checked = client.post('/experiments/validate', json=request)
        assert inspected.status_code == 200
        assert inspected.json()['resolved_experiment'] == checked.json()['resolved_experiment']
        networks = client.get('/catalog/networks').json()
        assert any(item['reference'] == 'builtin:free_space_radio' and item['kind'] == 'radio'
                   for item in networks['items'])
        assert 'network' in schemas['scenario']['properties']
