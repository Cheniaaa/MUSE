"""两个参考包的真实HTTP训练/恢复/评价与有界显示验证。"""
import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from test_runs_api import real_http_client, zip_body
from test_supervisor import wait_for


pytestmark = pytest.mark.skipif(os.environ.get('MUSE_TEST_GPU') != '1',
                              reason='Explicit serial GPU HTTP acceptance')
EXAMPLES = Path(__file__).resolve().parents[2] / 'examples'


def register_reference(client, scenario, *, horizon=4):
    source = EXAMPLES / scenario
    request = json.loads((source / 'request_train.json').read_text())
    request['topology']['world_count'] = 2
    request['algorithm_config'].update(collection_steps=horizon, hidden_dims=[16],
                                        num_learning_epochs=1, num_mini_batches=1)
    for kind, route in [('environment', 'environments'), ('training_package', 'training-packages')]:
        response = client.post('/catalog/' + route, content=zip_body(source / kind),
                               headers={'Content-Type': 'application/zip'})
        assert response.status_code == 201, response.text
        item = response.json()
        request[kind + '_id'], request[kind + '_revision'] = item['resource_id'], item['revision']
    validated = client.post('/experiments/validate', json=request)
    assert validated.status_code == 200 and validated.json()['valid'], validated.text
    response = client.post('/experiments', json=request)
    assert response.status_code == 201, response.text
    return response.json()['experiment_id'], request


def run_completed(client, identity, mode, steps, checkpoint=None):
    response = client.post('/runs', json=dict(experiment_id=identity, mode=mode,
        max_decision_steps=steps, checkpoint_artifact_id=checkpoint))
    assert response.status_code == 201, response.text
    run_id = response.json()['run_id']
    result = wait_for(lambda: (r if (r := client.get(f'/runs/{run_id}').json())['state']
        in ('COMPLETED', 'FAILED', 'LOST', 'CANCELLED') else None), timeout=300)
    directory = Path(result['artifact_directory'])
    assert result['state'] == 'COMPLETED', (result, (directory / 'logs/worker.log').read_text())
    assert result['exit_code'] == 0 and result['metrics']['steps'] == steps
    artifacts = client.get(f'/runs/{run_id}/artifacts')
    assert artifacts.status_code == 200, artifacts.text
    downloaded = {}
    for item in artifacts.json():
        response = client.get(f"/runs/{run_id}/artifacts/{item['id']}")
        assert response.status_code == 200, response.text
        assert len(response.content) == item['size']
        assert hashlib.sha256(response.content).hexdigest() == item['sha256']
        downloaded[item['kind']] = response.content
    events = [json.loads(line) for line in downloaded['metrics'].splitlines()]
    rows = [event['payload']['diagnostics']['task'] for event in events
            if event['payload']['source'] == 'step']
    assert len(rows) == steps
    assert all(len(row['episode_decision']) == len(row['task_reward']) for row in rows)
    assert all(all(row['service_valid']) for row in rows)
    assert all(any(power > 0 for power in row['rf_energy_j']) for row in rows)
    plan = json.loads(downloaded['run_plan'])
    assert (plan['checkpoint']['artifact_id'] if plan['checkpoint'] else None) == checkpoint
    print('REFERENCE_HTTP_RESULT', json.dumps(dict(run_id=run_id, mode=mode,
          directory=str(directory), metrics=result['metrics'])))
    return result, artifacts.json(), downloaded


@pytest.mark.parametrize('scenario', ['isac_cost', 'hybrid_history_markov'])
def test_reference_train_resume_and_evaluate(real_http_client, scenario):
    import torch
    client = real_http_client
    identity, request = register_reference(client, scenario)
    reference, previous = None, None
    for mode, steps, updates, total in [('TRAIN', 12, 3, 3), ('TRAIN', 8, 2, 5), ('EVALUATE', 9, 0, 5)]:
        result, artifacts, downloaded = run_completed(client, identity, mode, steps, reference)
        assert result['metrics']['updates'] == updates
        checkpoints = [item for item in artifacts if item['kind'] == 'checkpoint']
        if mode == 'EVALUATE':
            assert checkpoints == []
            continue
        assert len(checkpoints) == 1
        state = torch.load(io.BytesIO(downloaded['checkpoint']), weights_only=True, map_location='cpu')['learning_state']
        assert state['updates'] == total and state['revision'] == 2
        assert ('costs' in state) == (scenario == 'isac_cost')
        for role, owner in state['roles'].items():
            channel_ids = next(value for name, value in owner['actor'].items() if name.endswith('channel_ids'))
            assert channel_ids.tolist() == ([7, -3] if scenario == 'isac_cost' else [7, -3, 19])
            assert owner['optimizer']['state']
            if previous is not None:
                assert any(not torch.equal(value, previous['roles'][role]['actor'][name])
                           for name, value in owner['actor'].items())
        if scenario == 'isac_cost':
            cost, = state['costs'].values()
            assert cost['metadata']['cost_id'] == 'mission_position_violation'
            assert cost['optimizer']['state'] and torch.isfinite(cost['lambda'])
        reference, previous = checkpoints[0]['id'], state


def test_zero_update_checkpoint_batch_seed_change_and_viser(real_http_client):
    import torch
    client = real_http_client
    identity, request = register_reference(client, 'hybrid_history_markov', horizon=32)
    result, artifacts, downloaded = run_completed(client, identity, 'TRAIN', 1)
    assert result['metrics']['updates'] == 0
    checkpoint, = [item for item in artifacts if item['kind'] == 'checkpoint']
    state = torch.load(io.BytesIO(downloaded['checkpoint']), weights_only=True, map_location='cpu')['learning_state']
    assert state['updates'] == 0
    request['topology']['world_count'], request['root_seed'] = 3, 101
    response = client.post('/experiments', json=request)
    assert response.status_code == 201, response.text
    evaluation_id = response.json()['experiment_id']
    evaluated, _, _ = run_completed(client, evaluation_id, 'EVALUATE', 3, checkpoint['id'])
    assert evaluated['metrics']['updates'] == 0
    response = client.post('/runs', json=dict(experiment_id=evaluation_id, mode='INTERACTIVE',
        max_decision_steps=100000, checkpoint_artifact_id=checkpoint['id']))
    assert response.status_code == 201, response.text
    path = '/runs/' + response.json()['run_id']
    wait_for(lambda: client.get(path).json()['state'] in ('PREPARING', 'COMPILING', 'RUNNING'), timeout=30)
    pause = client.post(path + '/commands', json=dict(command_seq=1, command='pause'))
    assert pause.status_code == 200, pause.text
    paused = wait_for(lambda: (i['result'] if (i := client.get(path).json()['interaction'])['result']
        and i['result']['command_seq'] == 1 else None), timeout=180)
    assert paused['status'] == 'applied' and paused['paused']
    response = client.patch(path + '/display', json={'enabled': True})
    assert response.status_code == 200, response.text
    first = wait_for(lambda: (s if (s := client.get(path + '/display').json())['status'] == 'ready' else None), timeout=60)
    assert client.get(first['viewer_url']).is_success
    response = client.post(path + '/commands', json=dict(command_seq=2, command='step'))
    assert response.status_code == 200, response.text
    stepped = wait_for(lambda: (i['result'] if (i := client.get(path).json()['interaction'])['result']
        and i['result']['command_seq'] == 2 else None), timeout=30)
    assert stepped['steps'] == paused['steps'] + 1 and stepped['updates'] == 0
    last = wait_for(lambda: (s if (s := client.get(path + '/display').json())['frame']['tick']
                            > first['frame']['tick'] else None), timeout=30)
    assert last['frame']['decision'] == first['frame']['decision'] + 1
    assert client.post(path + '/cancel').status_code == 200
    terminal = wait_for(lambda: (r if (r := client.get(path).json())['state']
        in ('CANCELLED', 'COMPLETED', 'FAILED', 'LOST') else None), timeout=30)
    assert terminal['state'] == 'CANCELLED' and terminal['exit_code'] == 0
    assert not terminal['forced']
    print('REFERENCE_VISER_RESULT', json.dumps(dict(run_id=terminal['run_id'], first=first['frame'], last=last['frame'])))
