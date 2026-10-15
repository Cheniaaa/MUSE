"""checkpoint 的 CPU 登记、引用与下载边界；真实学习由 GPU 验收覆盖。"""
import copy
import json
from pathlib import Path
import sqlite3

import pytest

from test_runs_api import EXAMPLE, register_experiment
from test_supervisor import command, make_plan, wait_for


PUBLISH = r'''
import hashlib
from uuid import uuid4
root = Path(plan.artifact_directory)
(root / 'checkpoints').mkdir()
for content in (b'first checkpoint', b'second checkpoint'):
    token = uuid4().hex
    relative = 'checkpoints/' + token + '.pt'
    path = root / relative
    path.write_bytes(content)
    payload = dict(id=plan.run_id + ':checkpoint:' + token, run=plan.run_id,
                   kind='checkpoint', path=relative, size=len(content),
                   sha256=hashlib.sha256(content).hexdigest())
    fault = sys.argv[2]
    if fault == 'tmp': payload['path'] += '.tmp'; path.rename(root / payload['path'])
    if fault == 'outside': payload['path'] = '../outside.pt'; path.rename(root / payload['path'])
    if fault == 'absolute': payload['path'] = str(path)
    if fault == 'run': payload['run'] = 'another-run'
    if fault == 'id': payload['id'] = 'another-run:checkpoint:' + token
    if fault == 'malformed_id': payload['id'] = plan.run_id + ':checkpoint:not-uuid'
    if fault == 'kind': payload['kind'] = 'log'
    if fault == 'payload': del payload['sha256']
    if fault == 'size': payload['size'] += 1
    if fault == 'hash': payload['sha256'] = '0' * 64
    if fault == 'missing_file': path.unlink()
    if fault == 'directory': path.unlink(); path.mkdir()
    if fault == 'symlink':
        outside = root.parent / (token + '.pt')
        path.rename(outside)
        path.symlink_to(outside)
    if fault != 'orphan': p.emit('checkpoint', payload)
    if fault not in ('cooperative', 'failure'): break
'''


def checkpoint_command(behavior):
    def build(plan):
        args = command(behavior)(plan)
        args[2] = args[2].replace('behavior = sys.argv[2]', PUBLISH + '\nbehavior = sys.argv[2]')
        return args
    return build


@pytest.fixture
def published(tmp_path):
    from fastapi.testclient import TestClient
    from muse_server.app import create_app
    app = create_app(tmp_path / 'data', worker_command=checkpoint_command('cooperative'))
    with TestClient(app) as client:
        identity, exp = register_experiment(client, train=True, local_algorithm=True)
        response = client.post('/runs', json=dict(experiment_id=identity, mode='TRAIN',
                               max_decision_steps=8, checkpoint_interval_updates=1))
        assert response.status_code == 201, response.text
        run_id = response.json()['run_id']
        wait_for(lambda: client.get(f'/runs/{run_id}').json()['state'] == 'RUNNING')
        items = client.get(f'/runs/{run_id}/artifacts').json()
        checkpoints = [a for a in items if a['kind'] == 'checkpoint']
        assert len(checkpoints) == 2
        client.post(f'/runs/{run_id}/cancel')
        wait_for(lambda: client.get(f'/runs/{run_id}').json()['state'] == 'CANCELLED')
        app.state.supervisor.worker_command = command('success')
        yield client, identity, exp, run_id, checkpoints


def test_checkpoint_downloads_and_frozen_references(published):
    from muse_server.catalog import Catalog
    client, identity, exp, run_id, checkpoints = published
    for item, content in zip(checkpoints, (b'first checkpoint', b'second checkpoint')):
        response = client.get(f"/runs/{run_id}/artifacts/{item['id']}")
        assert response.status_code == 200 and response.content == content
        for mode, budget in (('TRAIN', 8), ('EVALUATE', 3)):
            created = client.post('/runs', json=dict(experiment_id=identity, mode=mode,
                max_decision_steps=budget, checkpoint_artifact_id=item['id']))
            assert created.status_code == 201, created.text
            plan = client.app.state.supervisor.get_plan(created.json()['run_id'])
            assert plan.checkpoint.artifact_id == item['id']
            assert Path(plan.checkpoint.path).read_bytes() == content
            assert (plan.checkpoint.size, plan.checkpoint.sha256) == (item['size'], item['sha256'])
    source = client.app.state.supervisor.get_plan(run_id)
    assert source.checkpoint_interval_updates == 1
    # 独立 Catalog 实例必须只靠已提交的 SQLite 记录恢复引用。
    catalog = client.app.state.catalog
    reopened = Catalog(catalog.directory, catalog.builtin_path)
    assert reopened.resolve_checkpoint(checkpoints[1]['id'], source.experiment).artifact_id == checkpoints[1]['id']
    assert client.get(f'/runs/{run_id}/artifacts/checkpoint').status_code == 404
    assert client.get(f'/runs/{run_id}/artifacts/log').status_code == 404
    other = client.post('/runs', json=dict(experiment_id=identity, mode='TRAIN', max_decision_steps=8)).json()['run_id']
    assert client.get(f"/runs/{other}/artifacts/{checkpoints[0]['id']}").status_code == 404


def test_checkpoint_reference_rejects_unregistered_wrong_kind_and_changed_files(published):
    client, identity, _, run_id, checkpoints = published
    root = Path(client.get(f'/runs/{run_id}').json()['artifact_directory'])
    orphan = root / 'checkpoints' / ('a' * 32 + '.pt')
    orphan.write_bytes(b'unregistered')
    for artifact_id in ('unknown:checkpoint:' + 'a' * 32, f'{run_id}:checkpoint:' + 'a' * 32,
                        f'{run_id}:log', str(orphan)):
        response = client.post('/runs', json=dict(experiment_id=identity, mode='TRAIN',
            max_decision_steps=8, checkpoint_artifact_id=artifact_id))
        assert response.status_code == 422, response.text
    item = checkpoints[0]
    path = root / item['path']
    for content in (b'x', b'x' * item['size'], None):
        if content is None:
            path.unlink()
        else:
            path.write_bytes(content)
        response = client.post('/runs', json=dict(experiment_id=identity, mode='TRAIN',
            max_decision_steps=8, checkpoint_artifact_id=item['id']))
        assert response.status_code == 422, response.text


def test_checkpoint_compatibility_and_mode_boundaries(published):
    client, identity, exp, _, checkpoints = published
    catalog = client.app.state.catalog
    artifact_id = checkpoints[0]['id']
    for field in ('algorithm_config', 'environment', 'entity_models', 'scenario', 'controllers', 'sensors',
                  'environment_scene', 'environment_metadata', 'algorithm_inputs'):
        target = catalog.get_experiment(identity)
        if field == 'algorithm_config': target.algorithm_config['num_learning_epochs'] += 1
        elif field == 'environment': target.environment.revision += 1
        elif field == 'entity_models': next(iter(target.entity_models.values())).revision += 1
        elif field == 'scenario': next(iter(target.scenario.placements.values()))[0].position[0] += 1
        elif field == 'environment_scene': target.environment_scene = 'another.xml'
        elif field == 'algorithm_inputs': next(iter(target.algorithm_inputs.actors.values())).reverse()
        else: getattr(target, field)['changed'] = {}
        target_id = catalog.save_experiment(target)
        response = client.post('/runs', json=dict(experiment_id=target_id, mode='EVALUATE',
            max_decision_steps=3, checkpoint_artifact_id=artifact_id))
        assert response.status_code == 422, (field, response.text)
    # 用正式 resolver 创建 B/root_seed/preview 不同但学习兼容的目标。
    request = json.loads((EXAMPLE / 'request_train.json').read_text())
    request.update(environment_id=exp['environment']['resource_id'], environment_revision=exp['environment']['revision'],
                   training_package_id=exp['training_package']['resource_id'],
                   training_package_revision=exp['training_package']['revision'], algorithm_id='custom_ippo')
    request['algorithm_config'].update(collection_steps=4, num_learning_epochs=1, num_mini_batches=1)
    request['term_config'] = {'world': {'termination': {'episode_limit': {'config': {'episode_max_steps': 4}}}}}
    request['topology']['world_count'] = 1
    request['root_seed'] = exp['root_seed'] + 1
    request['preview']['rate_hz'] = 5
    response = client.post('/experiments', json=request)
    assert response.status_code == 201, response.text
    target_id = response.json()['experiment_id']
    response = client.post('/runs', json=dict(experiment_id=target_id, mode='TRAIN', max_decision_steps=12,
        checkpoint_artifact_id=artifact_id, checkpoint_interval_updates=2))
    assert response.status_code == 201, response.text
    assert client.app.state.supervisor.get_plan(response.json()['run_id']).checkpoint_interval_updates == 2
    for kwargs in ({}, {'checkpoint_artifact_id': artifact_id, 'checkpoint_interval_updates': 1}):
        assert client.post('/runs', json=dict(experiment_id=identity, mode='EVALUATE',
            max_decision_steps=3, **kwargs)).status_code == 422
    rule_id, _ = register_experiment(client)
    assert client.post('/runs', json=dict(experiment_id=rule_id, mode='EVALUATE', max_decision_steps=3)).status_code == 201
    assert client.post('/runs', json=dict(experiment_id=rule_id, mode='EVALUATE', max_decision_steps=3,
        checkpoint_artifact_id=artifact_id)).status_code == 422


@pytest.mark.parametrize('fault', ['tmp', 'outside', 'absolute', 'run', 'id', 'malformed_id', 'kind',
                                 'payload', 'size', 'hash', 'missing_file', 'directory', 'symlink', 'io_error'])
def test_invalid_checkpoint_fails_run_but_supervisor_continues(tmp_path, fault, monkeypatch):
    from muse_server.supervisor import Supervisor, TERMINAL_STATES
    catalog, plan = make_plan(tmp_path, stop=.1)
    if fault == 'io_error':
        original_open = Path.open
        def denied_checkpoint(path, *args, **kwargs):
            if path.suffix == '.pt':
                raise PermissionError('checkpoint read denied for test')
            return original_open(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'open', denied_checkpoint)
    with Supervisor(catalog, worker_command=checkpoint_command(fault)) as supervisor:
        supervisor.submit(plan)
        record = wait_for(lambda: (r if (r := supervisor.get_run(plan.run_id))['state'] in TERMINAL_STATES else None))
        assert record['state'] == 'FAILED'
        assert 'Invalid Worker protocol' in record['failure']
        if fault == 'io_error':
            assert 'checkpoint read denied for test' in record['failure']
        assert not any(a['kind'] == 'checkpoint' for a in record['artifacts'])
        assert supervisor.error is None
        supervisor.worker_command = command('success')
        next_plan = catalog.create_run_plan(plan.experiment_id, mode='EVALUATE', device='cuda:0', max_decision_steps=1)
        supervisor.submit(next_plan)
        wait_for(lambda: supervisor.get_run(next_plan.run_id)['state'] == 'COMPLETED')


def test_orphan_checkpoint_is_not_scanned_and_failure_keeps_registered_files(tmp_path):
    from muse_server.supervisor import Supervisor, TERMINAL_STATES
    from muse_sdk.config import ConfigError
    catalog, plan = make_plan(tmp_path)
    with Supervisor(catalog, worker_command=checkpoint_command('orphan')) as supervisor:
        supervisor.submit(plan)
        wait_for(lambda: supervisor.get_run(plan.run_id)['state'] == 'COMPLETED')
        assert list(Path(plan.artifact_directory, 'checkpoints').glob('*.pt'))
        assert not any(a['kind'] == 'checkpoint' for a in supervisor.artifacts(plan.run_id))
        supervisor.worker_command = checkpoint_command('failure')
        second = catalog.create_run_plan(plan.experiment_id, mode='EVALUATE', device='cuda:0', max_decision_steps=1)
        supervisor.submit(second)
        record = wait_for(lambda: (r if (r := supervisor.get_run(second.run_id))['state'] in TERMINAL_STATES else None))
        assert record['state'] == 'FAILED'
        items = [a for a in record['artifacts'] if a['kind'] == 'checkpoint']
        assert len(items) == 2
        assert catalog.resolve_checkpoint(items[0]['id'], second.experiment).size == len(b'first checkpoint')
        orphan_path = next(Path(plan.artifact_directory, 'checkpoints').glob('*.pt'))
        with pytest.raises(ConfigError):
            catalog.resolve_checkpoint(f'{plan.run_id}:checkpoint:{orphan_path.stem}', plan.experiment)


def test_catalog_revalidates_persisted_checkpoint_identity_and_path(published):
    client, identity, _, run_id, checkpoints = published
    catalog = client.app.state.catalog
    record = client.get(f'/runs/{run_id}').json()
    for changes in ({'run': 'another-run'}, {'path': '../outside.pt'}, {'kind': 'log'}, {'sha256': '0' * 64}):
        damaged = copy.deepcopy(record)
        damaged['artifacts'][0].update(changes)
        with sqlite3.connect(catalog.database) as db:
            db.execute('UPDATE runs SET record=? WHERE id=?', (json.dumps(damaged), run_id))
        response = client.post('/runs', json=dict(experiment_id=identity, mode='EVALUATE',
            max_decision_steps=3, checkpoint_artifact_id=checkpoints[0]['id']))
        assert response.status_code == 422, response.text
    with sqlite3.connect(catalog.database) as db:
        db.execute('UPDATE runs SET record=? WHERE id=?', (json.dumps(record), run_id))
    plan_path = Path(record['plan_path'])
    original = plan_path.read_text()
    plan = json.loads(original)
    plan['run_id'] = 'another-run'
    plan_path.write_text(json.dumps(plan))
    try:
        response = client.post('/runs', json=dict(experiment_id=identity, mode='EVALUATE',
            max_decision_steps=3, checkpoint_artifact_id=checkpoints[0]['id']))
        assert response.status_code == 422, response.text
    finally:
        plan_path.write_text(original)


def test_catalog_rejects_unknown_checkpoint_before_supervisor_has_created_runs(tmp_path):
    from muse_sdk.config import ConfigError
    catalog, plan = make_plan(tmp_path)
    for artifact_id in ('checkpoint', 'unknown:checkpoint:' + 'a' * 32):
        with pytest.raises(ConfigError):
            catalog.resolve_checkpoint(artifact_id, plan.experiment)


@pytest.mark.parametrize('cancel', [False, True])
def test_worker_final_save_failure_overrides_success_or_cancel(tmp_path, cancel):
    import sys
    from muse_server.supervisor import Supervisor, TERMINAL_STATES
    catalog, plan = make_plan(tmp_path)
    # 只替换GPU装配；保留生产driver、Worker main及实际文件发布/异常协议。
    child = r'''
import sys
from pathlib import Path
from types import SimpleNamespace
from muse_sdk.runtime_types import ActResult, PolicyInput, StepOutput
from muse_runtime import worker
from muse_runtime.checkpoint import publish_checkpoint
from muse_runtime.driver import run
def execute(plan, protocol):
    env = SimpleNamespace(policy_input=PolicyInput(None, {}), rng=None)
    env.step = lambda actions: StepOutput(None, {}, None, {})
    algorithm = SimpleNamespace(set_mode=lambda mode: None,
        act=lambda *args, **kwargs: ActResult({}, None),
        process_env_step=lambda *args: None, update=lambda: {}, learning_state_dict=lambda: {'iteration': 0})
    (Path(plan.artifact_directory) / 'checkpoints').write_text('blocks directory creation')
    protocol.emit('running')
    return run(env, algorithm, protocol.read_control if CANCEL else lambda: None,
        lambda *args: None, mode='TRAIN', collection_steps=4, max_decision_steps=8,
        checkpoint_sink=lambda **counts: publish_checkpoint(plan, algorithm, **counts))
worker.execute = execute
raise SystemExit(worker.main())
'''.replace('CANCEL', repr(cancel))
    if cancel:
        child = child.replace("protocol.emit('running')", "protocol.emit('running')\n    import time\n    while protocol.read_control() != 'cancel': time.sleep(.01)")
        child = child.replace('protocol.read_control if True else lambda: None', "lambda: 'cancel'")
    def launch(plan):
        return [sys.executable, '-c', child, '--plan', str(Path(plan.artifact_directory) / 'resolved_run_plan.json')]
    with Supervisor(catalog, worker_command=launch) as supervisor:
        supervisor.submit(plan)
        if cancel:
            wait_for(lambda: supervisor.get_run(plan.run_id)['state'] == 'RUNNING')
            supervisor.cancel(plan.run_id)
        result = wait_for(lambda: (r if (r := supervisor.get_run(plan.run_id))['state'] in TERMINAL_STATES else None))
        assert result['state'] == 'FAILED' and result['exit_code'] == 1
        assert 'File exists' in result['failure']
        assert not any(item['kind'] == 'checkpoint' for item in result['artifacts'])


import os


@pytest.mark.skipif(os.environ.get('MUSE_TEST_GPU') != '1',
                    reason='Explicit final serial GPU acceptance')
@pytest.mark.parametrize('local_algorithm', [False, True], ids=['builtin', 'package'])
def test_real_gpu_checkpoint_roundtrip(tmp_path, monkeypatch, local_algorithm):
    import hashlib
    import io
    import socket
    import subprocess
    import sys

    import httpx2
    import torch
    from test_supervisor import ROOT

    monkeypatch.setenv('PYTHONPATH', str(ROOT / 'src'))
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    with (tmp_path / 'server.log').open('wb') as log:
        server = subprocess.Popen([sys.executable, '-m', 'muse_server.app', '--port', str(port),
            '--data-dir', str(tmp_path / 'gpu-data')], stdout=log, stderr=log)
        try:
            with httpx2.Client(base_url=f'http://127.0.0.1:{port}', timeout=30.) as client:
                def healthy():
                    try:
                        return client.get('/health').is_success
                    except httpx2.HTTPError:
                        return False
                wait_for(healthy, timeout=30)
                identity, exp = register_experiment(client, train=True, local_algorithm=local_algorithm)

                def execute(mode, budget, checkpoint=None):
                    payload = dict(experiment_id=identity, mode=mode, max_decision_steps=budget)
                    if checkpoint is not None:
                        payload['checkpoint_artifact_id'] = checkpoint
                    if mode == 'TRAIN':
                        payload['checkpoint_interval_updates'] = 1
                    response = client.post('/runs', json=payload)
                    assert response.status_code == 201, response.text
                    run_id = response.json()['run_id']
                    result = wait_for(lambda: (r if (r := client.get(f'/runs/{run_id}').json())['state']
                        in ('COMPLETED', 'FAILED', 'LOST', 'CANCELLED') else None), timeout=180)
                    directory = Path(result['artifact_directory'])
                    assert result['state'] == 'COMPLETED', (result, (directory / 'logs/worker.log').read_text())
                    assert result['exit_code'] == 0
                    assert result['metrics']['steps'] == budget
                    assert result['metrics']['updates'] == (2 if mode == 'TRAIN' else 0)
                    plan = json.loads((directory / 'resolved_run_plan.json').read_text())
                    assert plan['experiment_id'] == identity and plan['experiment'] == exp
                    assert (plan['checkpoint']['artifact_id'] if plan['checkpoint'] else None) == checkpoint
                    items = client.get(f'/runs/{run_id}/artifacts').json()
                    saved = []
                    for item in items:
                        if item['kind'] != 'checkpoint':
                            continue
                        downloaded = client.get(f"/runs/{run_id}/artifacts/{item['id']}")
                        assert downloaded.status_code == 200
                        assert len(downloaded.content) == item['size'] > 0
                        assert hashlib.sha256(downloaded.content).hexdigest() == item['sha256']
                        envelope = torch.load(io.BytesIO(downloaded.content), weights_only=True, map_location='cpu')
                        assert envelope['run_id'] == run_id and envelope['experiment_id'] == identity
                        saved.append((item['id'], envelope))
                    return saved

                first = execute('TRAIN', 8)
                assert [(e['steps'], e['updates']) for _, e in first] == [(4, 1), (8, 2), (8, 2)]
                assert len({identity for identity, _ in first}) == 3
                assert [e['learning_state']['updates'] for _, e in first] == [1, 2, 2]
                second = execute('TRAIN', 8, first[-1][0])
                assert [(e['steps'], e['updates']) for _, e in second] == [(4, 1), (8, 2), (8, 2)]
                assert len({identity for identity, _ in second}) == 3
                assert [e['learning_state']['updates'] for _, e in second] == [3, 4, 4]
                for role, state in first[-1][1]['learning_state']['roles'].items():
                    after = second[-1][1]['learning_state']['roles'][role]['actor']
                    assert any(not torch.equal(value, after[name]) for name, value in state['actor'].items())
                assert execute('EVALUATE', 3, second[-1][0]) == []
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=15)
            assert server.poll() is not None


@pytest.mark.parametrize('mode,reuse', [('TRAIN', False), ('EVALUATE', False), ('TRAIN', True), ('EVALUATE', True)])
def test_checkpoint_submit_cli_smoke(mode, reuse):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import subprocess
    import sys
    import threading

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            payload = body if self.headers['Content-Type'] == 'application/zip' else json.loads(body)
            requests.append((self.path, payload))
            responses = {'/catalog/environments': {'resource_id': 'env', 'revision': 1},
                         '/catalog/training-packages': {'resource_id': 'pkg', 'revision': 1},
                         '/experiments': {'experiment_id': 'exp'}, '/runs': {'run_id': 'run'}}
            content = json.dumps(responses[self.path]).encode()
            self.send_response(201)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    with HTTPServer(('127.0.0.1', 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            args = [sys.executable, str(EXAMPLE / 'submit.py'), '--server',
                    f'http://127.0.0.1:{server.server_port}', '--mode', mode,
                    '--max-decision-steps', '8' if mode == 'TRAIN' else '3', '--no-display']
            expected = dict(experiment_id='exp', mode=mode, max_decision_steps=8 if mode == 'TRAIN' else 3)
            if reuse:
                args += ['--experiment-id', 'exp', '--checkpoint-artifact-id', 'source:checkpoint:id']
                expected['checkpoint_artifact_id'] = 'source:checkpoint:id'
            if mode == 'TRAIN':
                args += ['--checkpoint-interval-updates', '2']
                expected['checkpoint_interval_updates'] = 2
            result = subprocess.run(args, capture_output=True, text=True, timeout=15)
            assert result.returncode == 0, result.stderr
            assert json.loads(result.stdout)['experiment_id'] == 'exp'
            assert requests[-1] == ('/runs', expected)
            assert [route for route, _ in requests] == (['/runs'] if reuse else
                ['/catalog/environments', '/catalog/training-packages', '/experiments', '/runs'])
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.parametrize('args,message', [
    (['--checkpoint-artifact-id', 'id'], 'requires --experiment-id'),
    (['--checkpoint-interval-updates', '0'], 'must be positive'),
    (['--mode', 'EVALUATE', '--checkpoint-interval-updates', '1'], 'requires --mode TRAIN'),
])
def test_checkpoint_submit_cli_rejects_invalid_arguments(args, message):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(EXAMPLE / 'submit.py'), *args],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and message in result.stderr
