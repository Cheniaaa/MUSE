"""真实合作 cost 示例的数值、注册与 HTTP 恢复验收。"""
import json
import os
from types import SimpleNamespace

import pytest

from mappo_helpers import EXAMPLE, cooperative_pipeline, example_definition
from test_runs_api import real_http_client, zip_body
from test_supervisor import wait_for


def cost_request(request):
    configured = json.loads((EXAMPLE / 'request_cost.json').read_text())
    for key in ('environment_id', 'environment_revision', 'training_package_id', 'training_package_revision'):
        configured[key] = request[key]
    return configured


def test_real_cost_terms_use_squared_action_target_error():
    import torch
    definition = example_definition()
    assert {'team_squared_error', 'scout_squared_error'} <= definition.terms.keys()
    positions = {'scout': torch.tensor([[[3., 0., 0.]], [[0., 0., 0.]]]),
                 'rover': torch.tensor([[[0., 4., 0.], [0., 0., 5.]], [[0., 0., 0.], [0., 0., 0.]]])}
    targets = {role: torch.full_like(value, 2.) for role, value in positions.items()}
    positions = {role: value + targets[role] for role, value in positions.items()}
    env = SimpleNamespace(role_positions_world=positions, act_commands=SimpleNamespace(target_world=targets),
                          commands=SimpleNamespace(target_world={role: value + 99 for role, value in targets.items()}))
    team = definition.terms['team_squared_error']
    scout = definition.terms['scout_squared_error']
    torch.testing.assert_close(team.implementation(env), torch.tensor([50. / 3., 0.]))
    torch.testing.assert_close(scout.implementation(env, role_id='scout'), torch.tensor([[9.], [0.]]))


def test_cost_example_opt_in_preserves_cost_free_baseline(pipeline, tmp_path):
    from muse_server.resolver import resolve_experiment
    catalog, request = cooperative_pipeline(pipeline, tmp_path)
    baseline = resolve_experiment(request, catalog.snapshot())
    assert baseline.valid, baseline.errors
    assert baseline.resolved_experiment.costs.model_dump() == {'roles': {}, 'world': {}}
    assert (EXAMPLE / 'request_cost.json').is_file(), 'Missing explicit cost request'
    report = resolve_experiment(cost_request(request), catalog.snapshot())
    assert report.valid, report.errors
    costs = report.resolved_experiment.costs
    assert set(costs.world) == {'team_squared_error'}
    assert set(costs.roles) == {'scout'}
    assert set(costs.roles['scout']) == {'scout_squared_error'}
    for item, scope, boundary in [(costs.world['team_squared_error'], 'world', 'world'),
                                  (costs.roles['scout']['scout_squared_error'], 'role', 'world_or_learning_done')]:
        assert item.scope == scope and item.boundary == boundary
        assert item.units == 'm^2' and item.statistic == 'episode_discounted_sum'
        assert item.budget == .01 and item.discount == .9


@pytest.mark.skipif(os.environ.get('MUSE_TEST_GPU') != '1', reason='Explicit serial GPU HTTP acceptance')
def test_real_cost_http_three_runs(real_http_client):
    import hashlib
    import io
    from pathlib import Path
    import torch

    client = real_http_client
    assert (EXAMPLE / 'request_cost.json').is_file(), 'Missing real cost example'
    request = json.loads((EXAMPLE / 'request_cost.json').read_text())
    request['algorithm_config'].update(share_actor_parameters=False, recurrent={'type': 'gru', 'hidden_size': 16})
    for kind, route in [('environment', 'environments'), ('training_package', 'training-packages')]:
        response = client.post('/catalog/' + route, content=zip_body(EXAMPLE / kind),
                               headers={'Content-Type': 'application/zip'})
        assert response.status_code == 201, response.text
        item = response.json()
        request[kind + '_id'], request[kind + '_revision'] = item['resource_id'], item['revision']
    response = client.post('/experiments', json=request)
    assert response.status_code == 201, response.text
    identity = response.json()['experiment_id']
    reference, previous = None, None
    run_ids, artifacts = [], []
    for mode, steps, updates in [('TRAIN', 12, 3), ('TRAIN', 8, 5), ('EVALUATE', 9, 5)]:
        response = client.post('/runs', json=dict(experiment_id=identity, mode=mode,
            max_decision_steps=steps, checkpoint_artifact_id=reference))
        assert response.status_code == 201, response.text
        run_id = response.json()['run_id']
        run_ids.append(run_id)
        result = wait_for(lambda: (r if (r := client.get(f'/runs/{run_id}').json())['state']
            in ('COMPLETED', 'FAILED', 'LOST', 'CANCELLED') else None), timeout=180)
        directory = Path(result['artifact_directory'])
        assert result['state'] == 'COMPLETED', (result, (directory / 'logs/worker.log').read_text())
        assert result['exit_code'] == 0
        assert result['metrics']['steps'] == steps
        assert result['metrics']['updates'] == (steps // 4 if mode == 'TRAIN' else 0)
        plan = json.loads((directory / 'resolved_run_plan.json').read_text())
        assert (plan['checkpoint']['artifact_id'] if plan['checkpoint'] else None) == reference
        items = client.get(f'/runs/{run_id}/artifacts')
        assert items.status_code == 200
        checkpoints = [a for a in items.json() if a['kind'] == 'checkpoint']
        print('COST_API_RESULT', json.dumps(dict(run_id=run_id, mode=mode, directory=str(directory),
              steps=steps, updates=result['metrics']['updates'], exit_code=result['exit_code'])))
        if mode == 'EVALUATE':
            assert checkpoints == []
            continue
        assert len(checkpoints) == 1
        item = checkpoints[0]
        downloaded = client.get(f"/runs/{run_id}/artifacts/{item['id']}")
        assert downloaded.status_code == 200
        assert len(downloaded.content) == item['size']
        assert hashlib.sha256(downloaded.content).hexdigest() == item['sha256']
        envelope = torch.load(io.BytesIO(downloaded.content), weights_only=True, map_location='cpu')
        assert envelope['run_id'] == run_id and envelope['experiment_id'] == identity
        state = envelope['learning_state']
        assert state['updates'] == updates
        assert set(state['roles']) == {'scout/slot/0', 'rover/slot/0', 'rover/slot/1'}
        assert len(state['costs']) == 2
        assert {(c['metadata']['scope'], c['metadata']['cost_id']) for c in state['costs'].values()} == {
            ('role', 'scout_squared_error'), ('world', 'team_squared_error')}
        for key, cost in state['costs'].items():
            assert cost['metadata']['units'] == 'm^2' and cost['metadata']['budget'] == .01
            assert torch.isfinite(cost['lambda']) and cost['lambda'] >= 0
            assert cost['lambda'].item() != pytest.approx(cost['metadata']['lambda_initial'])
            assert cost['optimizer']['state']
            assert all(torch.isfinite(t).all() for t in cost['critic'].values())
            if previous is not None:
                old = previous['costs'][key]
                assert not torch.equal(old['lambda'], cost['lambda'])
                assert any(not torch.equal(t, cost['critic'][name]) for name, t in old['critic'].items())
                assert all(v['step'] > old['optimizer']['state'][p]['step']
                           for p, v in cost['optimizer']['state'].items())
        if previous is not None:
            for owner, old in previous['roles'].items():
                assert any(not torch.equal(t, state['roles'][owner]['actor'][name]) for name, t in old['actor'].items())
        previous, reference = state, item['id']
        artifacts.append(reference)
    assert len(set(run_ids)) == 3 and len(set(artifacts)) == 2
