"""真实HTTP冻结策略交互：ACK、重试、暂停心跳、指定world reset及取消。"""
import json
import os
from pathlib import Path

import pytest

from mappo_helpers import EXAMPLE
from test_runs_api import real_http_client, zip_body
from test_supervisor import wait_for


@pytest.mark.skipif(os.environ.get('MUSE_TEST_GPU') != '1', reason='Explicit serial GPU HTTP acceptance')
def test_real_interactive_http_checkpoint_commands_and_cancel(real_http_client):
    client = real_http_client
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
    response = client.post('/runs', json=dict(experiment_id=identity, mode='TRAIN', max_decision_steps=4))
    assert response.status_code == 201, response.text
    training_id = response.json()['run_id']
    trained = wait_for(lambda: (r if (r := client.get(f'/runs/{training_id}').json())['state']
        in ('COMPLETED', 'FAILED', 'LOST', 'CANCELLED') else None), timeout=180)
    assert trained['state'] == 'COMPLETED', trained
    reference = next(a['id'] for a in trained['artifacts'] if a['kind'] == 'checkpoint')
    response = client.post('/runs', json=dict(experiment_id=identity, mode='INTERACTIVE',
        max_decision_steps=100000, checkpoint_artifact_id=reference))
    assert response.status_code == 201, response.text
    run_id = response.json()['run_id']
    path = f'/runs/{run_id}'
    wait_for(lambda: client.get(path).json()['state'] in ('PREPARING', 'COMPILING', 'RUNNING'), timeout=30)
    sequence = 0

    def execute(name, world_id=None):
        nonlocal sequence
        sequence += 1
        command = dict(command_seq=sequence, command=name)
        if world_id is not None:
            command['world_id'] = world_id
        response = client.post(path + '/commands', json=command)
        assert response.status_code == 200, response.text
        acknowledged = wait_for(lambda: (i['result'] if (i := client.get(path).json()['interaction'])['result']
            and i['result']['command_seq'] == sequence else None), timeout=180)
        assert acknowledged['status'] == 'applied', acknowledged
        assert acknowledged['updates'] == 0
        # 同请求重试返回确认结果，不再次推进或reset。
        assert client.post(path + '/commands', json=command).json()['result'] == acknowledged
        return acknowledged

    paused = execute('pause')
    assert paused['paused']
    directory = Path(client.get(path).json()['artifact_directory'])
    plan_before = (directory / 'resolved_run_plan.json').read_bytes()
    heartbeat = client.get(path).json()['heartbeat']
    alive = wait_for(lambda: (r if (r := client.get(path).json())['heartbeat'] != heartbeat else None), timeout=10)
    assert alive['interaction']['result'] == paused and alive['state'] == 'RUNNING'
    one = execute('step')
    assert one['paused'] and one['steps'] == paused['steps'] + 1
    reset = execute('reset_world', 1)
    assert reset['world_id'] == 1 and reset['paused'] and reset['steps'] == one['steps']
    after = execute('step')
    assert after['steps'] == one['steps'] + 1
    resumed = execute('resume')
    assert not resumed['paused']
    paused_again = execute('pause')
    assert paused_again['steps'] > resumed['steps'] and paused_again['paused']
    assert (directory / 'resolved_run_plan.json').read_bytes() == plan_before
    assert client.post(path + '/cancel').status_code == 200
    terminal = wait_for(lambda: (r if (r := client.get(path).json())['state']
        in ('COMPLETED', 'FAILED', 'LOST', 'CANCELLED') else None), timeout=30)
    assert terminal['state'] == 'CANCELLED', (terminal, (directory / 'logs/worker.log').read_text())
    assert terminal['exit_code'] == 0 and not terminal['forced']
    assert terminal['metrics']['steps'] == paused_again['steps'] and terminal['metrics']['updates'] == 0
    assert not [a for a in terminal['artifacts'] if a['kind'] == 'checkpoint']
    print('INTERACTIVE_API_RESULT', json.dumps(dict(training_run=training_id, interactive_run=run_id,
        commands=sequence, steps=terminal['metrics']['steps'], updates=0, state=terminal['state'])))
