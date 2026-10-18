import io
import json
import os
from pathlib import Path
import time
import zipfile
import re
import subprocess
import sys
import socket

import pytest

from test_supervisor import ROOT, command, wait_for

EXAMPLE = ROOT / 'examples/basic_flight'


def zip_body(directory):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.write(path, path.relative_to(directory))
    return buffer.getvalue()


def register_experiment(client, *, train=False, local_algorithm=False):
    env = client.post('/catalog/environments', content=zip_body(EXAMPLE / 'environment'),
                      headers={'content-type': 'application/zip'})
    assert env.status_code == 201, env.text
    package = zip_body(EXAMPLE / ('training_package_ippo' if train else 'training_package'))
    if local_algorithm:
        buffer = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(package)) as source, zipfile.ZipFile(buffer, 'w') as target:
            for name in source.namelist():
                content = source.read(name)
                if name == '__init__.py':
                    content = b'''from muse_sdk.registration import TrainingPackageDefinition
from muse_builtins.algorithms.ippo import IPPO
class CustomIPPO(IPPO):
    @classmethod
    def describe(cls):
        return {**super().describe(), 'algorithm_id': 'custom_ippo'}
def build_training_package():
    return TrainingPackageDefinition({}, {}, {'custom_ippo': CustomIPPO}, {}, {})
'''
                elif name == 'training_package.yaml':
                    content = content.replace(b'algorithm: builtin:ippo', b'algorithm: custom_ippo')
                target.writestr(name, content)
        package = buffer.getvalue()
    pkg = client.post('/catalog/training-packages', content=package,
                      headers={'content-type': 'application/zip'})
    assert pkg.status_code == 201, pkg.text
    request = json.loads((EXAMPLE / ('request_train.json' if train else 'request.json')).read_text())
    request.update(environment_id=env.json()['resource_id'], environment_revision=env.json()['revision'],
                   training_package_id=pkg.json()['resource_id'], training_package_revision=pkg.json()['revision'])
    if train:
        request['algorithm_config'].update(collection_steps=4, num_learning_epochs=1, num_mini_batches=1)
        request['term_config'] = {'world': {'termination': {'episode_limit': {'config': {'episode_max_steps': 4}}}}}
    validated = client.post('/experiments/validate', json=request)
    assert validated.status_code == 200 and validated.json()['valid'], validated.text
    saved = client.post('/experiments', json=request)
    assert saved.status_code == 201, saved.text
    return saved.json()['experiment_id'], validated.json()['resolved_experiment']


@pytest.fixture(autouse=True)
def source_environment(monkeypatch):
    monkeypatch.setenv('PYTHONPATH', str(ROOT / 'src'))
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient
    from muse_server.app import create_app
    with TestClient(create_app(tmp_path / 'data', worker_command=command('success'))) as client:
        yield client


@pytest.fixture
def real_http_client(tmp_path):
    """复用真实 Server/Worker HTTP 入口；退出时关闭服务及其子进程。"""
    import httpx2
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    with (tmp_path / 'server.log').open('wb') as log:
        server = subprocess.Popen([sys.executable, '-m', 'muse_server.app', '--port', str(port),
                                   '--data-dir', str(tmp_path / 'gpu-data')], stdout=log, stderr=log)
        try:
            with httpx2.Client(base_url=f'http://127.0.0.1:{port}', timeout=30.) as connection:
                def healthy():
                    try:
                        return connection.get('/health').is_success
                    except httpx2.HTTPError:
                        return False
                wait_for(healthy, timeout=30)
                yield connection
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=15)
            assert server.poll() is not None


def test_real_api_registration_resolution_run_events_and_artifacts(client):
    assert client.get('/health').json()['status'] == 'ok'
    identity, exp = register_experiment(client)
    assert client.get(f'/experiments/{identity}').json() == exp
    for name in ('environments', 'training-packages', 'entity-models', 'controllers', 'sensors'):
        page = client.get('/catalog/' + name).json()
        assert page['items'] and page['next_offset'] is None
    created = client.post('/runs', json=dict(experiment_id=identity, mode='EVALUATE', max_decision_steps=8))
    assert created.status_code == 201, created.text
    run_id = created.json()['run_id']
    with client.websocket_connect(f'/streams/runs/{run_id}/events') as ws:
        while (event := ws.receive_json())['kind'] != 'terminal':
            assert event['run_id'] == run_id
        assert event['payload']['state'] == 'COMPLETED'
    result = client.get(f'/runs/{run_id}').json()
    assert result['exit_code'] == 0
    assert client.get('/runs').json()[0]['run_id'] == run_id
    artifacts = client.get(f'/runs/{run_id}/artifacts').json()
    log = next(a for a in artifacts if a['kind'] == 'log')
    response = client.get(f"/runs/{run_id}/artifacts/{log['id']}")
    assert response.status_code == 200 and 'native stdout' in response.text
    assert log['size'] == len(response.content)
    plan = client.app.state.supervisor.get_plan(run_id)
    assert plan.max_decision_steps == 8
    assert plan.experiment.model_dump(mode='json') == exp


def test_run_budget_and_unsupported_mode_boundary(client):
    identity, _ = register_experiment(client, train=True)
    for budget in (None, 0, -1, True, 1.5, '8'):
        payload = dict(experiment_id=identity, mode='TRAIN', max_decision_steps=budget)
        response = client.post('/runs', json=payload)
        assert response.status_code == 422, response.text
    for mode in ('EVALUATE', 'INTERACTIVE'):
        response = client.post('/runs', json=dict(experiment_id=identity, mode=mode, max_decision_steps=8))
        assert response.status_code == 422, response.text
    response = client.post('/runs', json=dict(experiment_id=identity, mode='TRAIN', max_decision_steps=8,
                                            resource_paths={'environment': '/tmp/user-path'}))
    assert response.status_code == 422
    for budget in (1, 6, 8):
        response = client.post('/runs', json=dict(experiment_id=identity, mode='TRAIN', max_decision_steps=budget))
        assert response.status_code == 201, response.text
        plan = client.app.state.supervisor.get_plan(response.json()['run_id'])
        assert plan.max_decision_steps == budget
        assert plan.experiment.training_schedule.collection_steps == 4


def test_save_resolves_input_and_does_not_accept_frozen_forgery(client):
    identity, exp = register_experiment(client)
    assert client.post('/experiments', json=exp).status_code == 422
    assert client.get('/experiments/missing').status_code == 404
    assert client.get('/runs/missing').status_code == 404
    assert client.post('/runs', json=dict(experiment_id=identity, mode='TRAIN', max_decision_steps=8)).status_code == 422


def test_interactive_rule_admission_and_learning_checkpoint_required(client):
    identity, _ = register_experiment(client)
    response = client.post('/runs', json=dict(experiment_id=identity, mode='INTERACTIVE', max_decision_steps=8))
    assert response.status_code == 201, response.text
    assert response.json()['interaction'] == {'command': None, 'result': None}
    identity, _ = register_experiment(client, train=True)
    response = client.post('/runs', json=dict(experiment_id=identity, mode='INTERACTIVE', max_decision_steps=8))
    assert response.status_code == 422
    assert response.json()['errors'][0]['field'] == 'checkpoint_artifact_id'
    from test_checkpoint import checkpoint_command
    supervisor = client.app.state.supervisor
    supervisor.worker_command = checkpoint_command('success')
    training = client.post('/runs', json=dict(experiment_id=identity, mode='TRAIN', max_decision_steps=8)).json()
    trained = wait_for(lambda: (r if (r := client.get(f"/runs/{training['run_id']}").json())['state'] == 'COMPLETED' else None))
    artifact = next(a for a in trained['artifacts'] if a['kind'] == 'checkpoint')
    response = client.post('/runs', json=dict(experiment_id=identity, mode='INTERACTIVE',
        max_decision_steps=3, checkpoint_artifact_id=artifact['id']))
    assert response.status_code == 201, response.text
    plan = supervisor.get_plan(response.json()['run_id'])
    assert plan.mode == 'INTERACTIVE' and plan.checkpoint.artifact_id == artifact['id']
    assert plan.checkpoint_interval_updates is None
    assert client.post('/runs', json=dict(experiment_id=identity, mode='INTERACTIVE',
        max_decision_steps=3, checkpoint_artifact_id=artifact['id'], checkpoint_interval_updates=1)).status_code == 422


def test_commands_api_pending_get_and_errors(tmp_path):
    from fastapi.testclient import TestClient
    from muse_server.app import create_app
    from test_interactive_supervisor import CHILD
    with TestClient(create_app(tmp_path / 'commands', worker_command=lambda p: [sys.executable, '-c', CHILD, p.run_id])) as client:
        identity, _ = register_experiment(client)
        response = client.post('/runs', json=dict(experiment_id=identity, mode='INTERACTIVE', max_decision_steps=8))
        assert response.status_code == 201, response.text
        run_id = response.json()['run_id']
        wait_for(lambda: client.get(f'/runs/{run_id}').json()['state'] == 'RUNNING')
        path = f'/runs/{run_id}/commands'
        request = dict(command_seq=1, command='pause')
        assert client.post(path, json=request).json()['result'] is None
        assert client.post(path, json=request).status_code == 200
        result = wait_for(lambda: client.get(f'/runs/{run_id}').json()['interaction']['result'])
        assert result['status'] == 'applied' and result['paused']
        assert client.post(path, json=request).json()['result'] == result
        for invalid in (dict(command_seq=1, command='resume'), dict(command_seq=3, command='pause'),
                        dict(command_seq=2, command='reset_world', world_id=2),
                        dict(command_seq=2, command='pause', world_id=0), dict(command_seq=0, command='pause')):
            assert client.post(path, json=invalid).status_code == 422
        assert client.post('/runs/missing/commands', json=request).status_code == 404
        client.post(f'/runs/{run_id}/cancel')
        assert client.post(path, json=request).status_code == 422


def test_upload_is_zip_body_and_traversal_rejected(client):
    assert client.post('/catalog/environments', content=b'bad',
                       headers={'content-type': 'text/plain'}).status_code == 415
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('../escape', 'x')
    assert client.post('/catalog/environments', content=buffer.getvalue(),
                       headers={'content-type': 'application/zip'}).status_code == 422


@pytest.mark.parametrize('name,content', [('environment.yaml', 'scene: 42\n'), ('bad.json', '{bad-json')])
def test_malformed_archive_metadata_is_client_error(client, name, content):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr(name, content)
    response = client.post('/catalog/environments', content=buffer.getvalue(),
                           headers={'content-type': 'application/zip'})
    assert response.status_code == 422
    assert response.json()['errors']


@pytest.mark.skipif(os.environ.get('MUSE_TEST_GPU') != '1', reason='Explicit final serial GPU acceptance')
@pytest.mark.parametrize('local_algorithm', [False, True], ids=['builtin', 'package'])
def test_real_gpu_api_train_two_blocks(real_http_client, local_algorithm):
    client = real_http_client
    identity, exp = register_experiment(client, train=True, local_algorithm=local_algorithm)
    response = client.post('/runs', json=dict(experiment_id=identity, mode='TRAIN', max_decision_steps=8))
    assert response.status_code == 201, response.text
    run_id = response.json()['run_id']
    result = wait_for(lambda: (r if (r := client.get(f'/runs/{run_id}').json())['state']
                              in ('COMPLETED', 'FAILED', 'LOST') else None), timeout=180)
    directory = Path(result['artifact_directory'])
    assert result['state'] == 'COMPLETED', (result, (directory / 'logs/worker.log').read_text())
    assert result['exit_code'] == 0
    assert result['metrics']['steps'] == 8 and result['metrics']['updates'] == 2
    metrics = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    updates = [e for e in metrics if e['payload']['source'] == 'update']
    assert len(updates) == 2
    assert set(updates[-1]['payload']['metrics']) == {'scout', 'guard'}
    plan = json.loads((directory / 'resolved_run_plan.json').read_text())
    assert plan['experiment'] == exp and plan['max_decision_steps'] == 8
    print('GPU_API_RESULT', json.dumps(dict(run_id=run_id, directory=str(directory),
          steps=result['metrics']['steps'], updates=result['metrics']['updates'], exit_code=result['exit_code'])))


@pytest.mark.skipif(os.environ.get('MUSE_VERIFY_LOCK') != '1', reason='Explicit network lock verification')
def test_lock_regeneration_matches_versions_and_hashes(tmp_path):
    destination = tmp_path / 'requirements.lock'
    subprocess.run(['uv', 'pip', 'compile', 'pyproject.toml', '--extra', 'test', '--python', '.myenv/bin/python',
                    '--torch-backend', 'cu128', '--generate-hashes', '--emit-index-url', '--emit-index-annotation',
                    '--format', 'requirements.txt', '--output-file', str(destination), '--quiet'], cwd=ROOT, check=True)
    def packages(path):
        result = {}
        current = None
        for line in path.read_text().splitlines():
            if match := re.match(r'^([a-zA-Z0-9_.-]+)==([^\s\\]+)', line):
                current = (match[1].lower().replace('_', '-'), match[2])
                result[current] = set()
            elif '--hash=sha256:' in line:
                result[current].add(line.strip().split()[0])
        return result
    expected = packages(ROOT / 'requirements.lock')
    assert expected == packages(destination)
    assert ('torch', '2.11.0+cu128') in expected
    assert all(expected.values())
    print('LOCK_VERIFIED', len(expected), 'packages, identical versions and hashes')


@pytest.mark.skipif(os.environ.get('MUSE_VERIFY_LOCK') != '1', reason='Explicit deployment wheel verification')
def test_deployment_wheel_contains_assets_and_controller(tmp_path):
    subprocess.run(['uv', 'build', '--wheel', '--no-build-isolation', '--python', '.myenv/bin/python',
                    '--out-dir', str(tmp_path), '--no-create-gitignore', '--no-build-logs'], cwd=ROOT, check=True)
    with zipfile.ZipFile(next(tmp_path.glob('muse-*.whl'))) as wheel:
        names = set(wheel.namelist())
        asset = ROOT / 'src/muse_builtins/assets'
        expected = {str(path.relative_to(ROOT / 'src')) for path in asset.rglob('*') if path.is_file()}
        assert expected <= names
        controller = next(name for name in names if name.endswith('/share/muse/controller.json'))
        assert wheel.read(controller) == (EXAMPLE / 'controller.json').read_bytes()
        assert 'muse_server/app.py' in names and 'muse_runtime/worker.py' in names
