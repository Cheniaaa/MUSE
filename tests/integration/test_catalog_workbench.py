"""工作台来源、分页、资源更新与真实参考导入的持久边界。"""
from copy import deepcopy
from datetime import datetime
import shutil

from fastapi.testclient import TestClient
import pytest

from flight_helpers import EXAMPLE, archive
from muse_sdk.config import ConfigError
from muse_server.app import bootstrap, create_app
from muse_server.catalog import Catalog
from muse_server.resolver import resolve_experiment


def test_source_roundtrip_does_not_expand_defaults(pipeline, tmp_path):
    original, source = pipeline
    catalog = Catalog(tmp_path / 'catalog', original.builtin_path)
    request = deepcopy(source)
    request.pop('term_config')
    request.pop('algorithm_config')
    checked = resolve_experiment(request, original.snapshot())
    assert checked.valid, checked.errors
    resolved = checked.resolved_experiment
    assert resolved is not None and resolved.algorithm_config['motion']
    identity = catalog.save_experiment(resolved, source=request, label='对照')
    expected = deepcopy(request)
    request['root_seed'] = 99
    reopened = Catalog(catalog.directory, catalog.builtin_path)
    stored = reopened.get_experiment_source(identity)
    assert stored == expected
    assert 'term_config' not in stored and 'algorithm_config' not in stored
    assert reopened.get_experiment(identity) == resolved
    page = reopened.list_experiments(limit=1)
    assert page['next_offset'] is None
    assert page['items'][0]['id'] == identity and page['items'][0]['label'] == '对照'
    assert page['items'][0]['composition_id'] == request['composition_id']
    assert datetime.fromisoformat(page['items'][0]['created_at']).tzinfo is not None
    assert 'definition' not in page['items'][0] and 'source' not in page['items'][0]


def test_catalog_identity_survives_reopen_and_rejects_legacy_without_writes(pipeline, tmp_path):
    catalog, _ = pipeline
    reopened = Catalog(catalog.directory, catalog.builtin_path)
    assert reopened.catalog_id == catalog.catalog_id
    other = Catalog(tmp_path / 'new', catalog.builtin_path)
    assert other.catalog_id != catalog.catalog_id
    old = tmp_path / 'old'
    old.mkdir()
    (old / 'catalog.sqlite3').write_bytes(b'legacy data must remain untouched')
    before = {p.name: p.read_bytes() for p in old.iterdir()}
    with pytest.raises(ConfigError, match='[Nn]ew|[Ll]egacy|revision'):
        Catalog(old, catalog.builtin_path)
    with pytest.raises(ConfigError):
        bootstrap(old, EXAMPLE / 'controller.json')
    assert {p.name: p.read_bytes() for p in old.iterdir()} == before


def test_revision_failure_preserves_previous_resource(pipeline, tmp_path):
    catalog, request = pipeline
    record = next(r for r in catalog.snapshot().resources if r.resource_id == request['environment_id'])
    content = tmp_path / 'env'
    shutil.copytree(record.path, content)
    scene = content / record.metadata['scene']
    scene.write_text('<mujoco><worldbody><geom type="box" size="1 2 3" pos="0 0 3"/></worldbody></mujoco>')
    with TestClient(create_app(catalog.directory)) as client:
        updated = client.post('/catalog/environments', params={'resource_id': record.resource_id},
            content=archive(content, tmp_path / 'updated.zip').read_bytes(), headers={'Content-Type': 'application/zip'})
        assert updated.status_code == 201, updated.text
        assert updated.json()['revision'] == record.revision + 1
        assert updated.json()['resource_id'] == record.resource_id
        scene.write_text('<mujoco><worldbody>broken')
        failed = client.post('/catalog/environments', params={'resource_id': record.resource_id},
            content=archive(content, tmp_path / 'failed.zip').read_bytes(), headers={'Content-Type': 'application/zip'})
        assert failed.status_code == 422
        revisions = [r for r in catalog.snapshot().resources if r.resource_id == record.resource_id]
        assert [r.revision for r in revisions] == [record.revision, record.revision + 1]
        assert revisions[0] == record
        assert not (catalog.directory / 'resources' / f'{record.resource_id}-{record.revision + 2}').exists()


def test_http_sources_and_bounded_catalog_pages(pipeline):
    source_catalog, request = pipeline
    with TestClient(create_app(source_catalog.directory)) as client:
        identities = [client.post('/experiments', json=request, params={'label': f'试验{index}'}).json()['experiment_id']
                      for index in range(3)]
        first = client.get('/experiments', params={'limit': 2}).json()
        second = client.get('/experiments', params={'limit': 2, 'offset': first['next_offset']}).json()
        assert [r['id'] for r in first['items'] + second['items']] == identities
        assert second['next_offset'] is None
        assert client.get(f'/experiments/{identities[0]}/source').json() == request
        assert client.get('/experiments/missing/source').status_code == 404
        health = client.get('/health').json()
        assert health['api_revision'] == 2 and health['catalog_id'] == source_catalog.catalog_id
        for route in ('environments', 'training-packages', 'entity-models', 'controllers', 'sensors', 'networks'):
            first = client.get('/catalog/' + route, params={'limit': 1}).json()
            assert isinstance(first['items'], list) and len(first['items']) == 1
            if first['next_offset'] is not None:
                second = client.get('/catalog/' + route, params={'limit': 1, 'offset': first['next_offset']}).json()
                assert second['items'] and second['items'] != first['items']
            assert client.get('/catalog/' + route, params={'offset': -1}).status_code == 422
            assert client.get('/catalog/' + route, params={'limit': 201}).status_code == 422
            assert client.get('/catalog/' + route, params={'limit': 0}).status_code == 422


@pytest.mark.parametrize('example_id,algorithm', [('basic_flight', 'builtin:ippo'),
    ('isac_cost', 'builtin:mappo'), ('hybrid_history_markov', 'builtin:ippo')])
def test_examples_import_real_refs(pipeline, example_id, algorithm):
    from muse_server.examples import import_example
    catalog, _ = pipeline
    imported = import_example(catalog, example_id)
    checked = resolve_experiment(imported['request'], catalog.snapshot())
    assert checked.valid, checked.errors
    assert checked.resolved_experiment.algorithm_id == algorithm
    assert imported['run_options']['mode'] == 'TRAIN'
    assert len(imported['resources']) == 2
    if example_id == 'isac_cost':
        assert imported['research_status']['learning'] == 'not_met'
    elif example_id == 'hybrid_history_markov':
        assert imported['research_status']['learning'] == 'incomplete'


def test_example_routes_reject_unknown_and_offer_real_empty_ground(pipeline):
    catalog, _ = pipeline
    with TestClient(create_app(catalog.directory)) as client:
        directory = client.get('/catalog/examples').json()
        assert {item['id'] for item in directory['items']} == {
            'basic_flight', 'isac_cost', 'hybrid_history_markov', 'empty_environment'}
        imported = client.post('/catalog/examples/empty_environment/import')
        assert imported.status_code == 201, imported.text
        body = imported.json()
        assert 'training_package_id' not in body['request'] and len(body['resources']) == 1
        ref = body['resources'][0]
        record = next(r for r in catalog.snapshot().resources if r.resource_id == ref['resource_id'])
        assert record.metadata['geometry_summary']['geoms']
        assert client.post('/catalog/examples/not-a-template/import').status_code == 422
