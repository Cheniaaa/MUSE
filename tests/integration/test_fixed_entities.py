"""固定端点不占学习动作；可控实体的动作归属仍完整。"""
import copy
import json
import os
from pathlib import Path

import pytest

from flight_helpers import add_fixed_anchors
from test_runs_api import real_http_client, zip_body
from test_supervisor import wait_for


@pytest.mark.parametrize('index', [0, 1, 2])
def test_fixed_entities_preserve_action_layout_and_freeze_resources(pipeline, index):
    from muse_sdk.content import hash_package
    from muse_server.resolver import resolve_experiment
    catalog, source = pipeline
    original = resolve_experiment(source, catalog.snapshot()).resolved_experiment
    request = copy.deepcopy(source)
    add_fixed_anchors(request, index=index)
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    exp = report.resolved_experiment
    assert exp.action_specs == original.action_specs
    assert exp.action_layout == original.action_layout
    assert exp.algorithm_inputs == original.algorithm_inputs
    assert exp.activation == original.activation
    assert exp.controllers['anchor_group'] == {'id': 'none'}
    assert set(exp.bindings) == {'scout', 'guard'}
    identity = catalog.save_experiment(exp, source=request)
    plan = catalog.create_run_plan(identity, mode='EVALUATE', device='cpu', max_decision_steps=2)
    model = plan.experiment.entity_models['radio_base_station']
    asset = Path(plan.resource_paths['entity_model:radio_base_station'])
    assert (asset / model.physics_asset).is_file()
    assert hash_package(asset) == model.sha256
    assert plan.experiment.scenario.placements['anchor_start'] == exp.scenario.placements['anchor_start']


@pytest.mark.parametrize('change,field', [
    ('unowned', 'role_config'), ('active_none', 'controller_id'),
    ('passive_active', 'controller_id'), ('passive_role', 'role_config.scout'),
])
def test_fixed_entities_do_not_relax_active_ownership(pipeline, change, field):
    from muse_server.resolver import resolve_experiment
    catalog, source = pipeline
    request = copy.deepcopy(source)
    add_fixed_anchors(request)
    if change == 'unowned':
        group = request['scenario']['entity_groups'][0]
        group['count'] += 1
        request['scenario']['placements'][group['placement_id']].append(
            dict(position=[3., 3., 1.], orientation_wxyz=[1., 0., 0., 0.]))
    elif change == 'active_none':
        request['scenario']['entity_groups'][0]['controller_id'] = 'none'
    elif change == 'passive_active':
        request['scenario']['entity_groups'][-1]['controller_id'] = 'cf2_body_velocity_v1'
    else:
        request['role_config']['scout'] = dict(group_id='anchor_group', slots=[0])
    report = resolve_experiment(request, catalog.snapshot())
    assert not report.valid
    assert any(field in error['field'] for error in report.errors), report.errors


@pytest.mark.skipif(os.environ.get('MUSE_TEST_GPU') != '1', reason='Explicit serial GPU acceptance')
def test_fixed_entities_real_http_training(real_http_client):
    client = real_http_client
    example = Path(__file__).resolve().parents[2] / 'examples/cooperative_mappo'
    request = json.loads((example / 'request_train.json').read_text())
    for kind, route in [('environment', 'environments'), ('training_package', 'training-packages')]:
        response = client.post('/catalog/' + route, content=zip_body(example / kind),
                               headers={'Content-Type': 'application/zip'})
        assert response.status_code == 201, response.text
        resource = response.json()
        request[kind + '_id'], request[kind + '_revision'] = resource['resource_id'], resource['revision']
    add_fixed_anchors(request, index=0)
    saved = client.post('/experiments', json=request)
    assert saved.status_code == 201, saved.text
    created = client.post('/runs', json=dict(experiment_id=saved.json()['experiment_id'],
                                           mode='TRAIN', max_decision_steps=8))
    assert created.status_code == 201, created.text
    run_id = created.json()['run_id']
    result = wait_for(lambda: (r if (r := client.get(f'/runs/{run_id}').json())['state']
        in ('COMPLETED', 'FAILED', 'LOST', 'CANCELLED') else None), timeout=180)
    directory = Path(result['artifact_directory'])
    assert result['state'] == 'COMPLETED', (result, (directory / 'logs/worker.log').read_text())
    assert result['exit_code'] == 0
    assert result['metrics']['steps'] == 8 and result['metrics']['updates'] == 2
    frozen = json.loads((directory / 'resolved_run_plan.json').read_text())['experiment']
    assert frozen['controllers']['anchor_group'] == {'id': 'none'}
    assert set(frozen['action_specs']) == {'scout', 'rover'}
    assert 'controller:anchor_group' not in frozen['activation']
    assert any(artifact['kind'] == 'checkpoint' for artifact in result['artifacts'])
