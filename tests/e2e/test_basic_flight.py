import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import zipfile

import httpx2
import pytest


ROOT = Path(__file__).resolve().parents[2]
TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'LOST'}
pytestmark = pytest.mark.skipif(os.environ.get('MUSE_TEST_GPU') != '1',
                                reason='Explicit serial GPU E2E acceptance')


def wait_for(check, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(.02)
    raise AssertionError(f'Timed out after {timeout}s: {check}')


def response_json(response, status=200):
    assert response.status_code == status, response.text
    return response.json()


def archive(directory):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as bundle:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                bundle.write(path, path.relative_to(directory))
    return buffer.getvalue()


def register(client):
    example = ROOT / 'examples/basic_flight'
    env = response_json(client.post('/catalog/environments', content=archive(example / 'environment'),
                                   headers={'content-type': 'application/zip'}), 201)
    package = response_json(client.post('/catalog/training-packages',
        content=archive(example / 'training_package_ippo'), headers={'content-type': 'application/zip'}), 201)
    request = json.loads((example / 'request_train.json').read_text())
    request.update(environment_id=env['resource_id'], environment_revision=env['revision'],
                   training_package_id=package['resource_id'], training_package_revision=package['revision'])
    models_page = response_json(client.get('/catalog/entity-models'))
    assert models_page['next_offset'] is None
    models = {model['resource_id']: model for model in models_page['items']}
    for group in request['scenario']['entity_groups']:
        model = models[group['model_id']]
        group.update(model_id=model['resource_id'], model_revision=model['revision'])
    request['algorithm_config'].update(collection_steps=4, num_learning_epochs=1, num_mini_batches=1)
    request['term_config'] = {'world': {'termination': {'episode_limit': {'config': {'episode_max_steps': 4}}}}}
    validated = response_json(client.post('/experiments/validate', json=request))
    assert validated['valid'], validated
    identity = response_json(client.post('/experiments', json=request), 201)['experiment_id']
    assert response_json(client.get(f'/experiments/{identity}')) == validated['resolved_experiment']
    return identity, validated['resolved_experiment']


def run_state(client, run_id):
    return response_json(client.get(f'/runs/{run_id}'))


def terminal(client, run_id):
    return wait_for(lambda: (r if (r := run_state(client, run_id))['state'] in TERMINAL else None))


def start_run(client, identity, budget):
    created = response_json(client.post('/runs', json={
        'experiment_id': identity, 'mode': 'TRAIN', 'max_decision_steps': budget}), 201)
    run_id = created['run_id']
    def live():
        state = run_state(client, run_id)
        assert state['state'] not in TERMINAL, state
        return state if state['pid'] is not None else None
    return wait_for(live)


def patch(client, run_id, **settings):
    return response_json(client.patch(f'/runs/{run_id}/display', json=settings))


def display(client, run_id):
    return response_json(client.get(f'/runs/{run_id}/display'))


def finite_metrics(value):
    if isinstance(value, dict):
        return bool(value) and all(finite_metrics(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(finite_metrics(item) for item in value)
    return isinstance(value, (int, float)) and math.isfinite(value)


def ready(client, run_id, seq):
    def observed():
        state = run_state(client, run_id)
        assert state['state'] not in TERMINAL, state
        result = display(client, run_id)
        return result if result['status'] == 'ready' and result['effective_seq'] == seq else None
    return wait_for(observed)


def viewer_pid(server_pid, run_id):
    matches = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
            stat = (path.parent / 'stat').read_text().rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if (b'muse_server.viewers.viser_process' in args and run_id.encode() in args
                and int(stat[1]) == server_pid):
            matches.append(int(path.parent.name))
    assert len(matches) == 1, matches
    return matches[0]


def evidence(client, result, experiment, destination):
    directory = Path(result['artifact_directory'])
    plan = json.loads(Path(result['plan_path']).read_text())
    assert plan['experiment'] == experiment
    assert plan['run_id'] == result['run_id']
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    metrics = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    updates = [event['payload'] for event in metrics if event['payload']['source'] == 'update']
    assert len(updates) == result['metrics']['updates'] > 0
    for update in updates:
        assert set(update['metrics']) == {'scout', 'guard'}
        for role in update['metrics'].values():
            assert finite_metrics(role), role
    ended = [event['payload'] for event in events if event['kind'] == 'terminal']
    assert len(ended) == 1
    for key in ('steps', 'updates'):
        assert ended[0][key] == result['metrics'][key]
    assert ended[0]['state'] == result['state']
    assert result['exit_code'] == 0 and not result['forced'], result
    model = next(event['payload'] for event in events if event['kind'] == 'preview_model')
    assert hashlib.sha256((directory / model['path']).read_bytes()).hexdigest() == model['sha256']
    published = response_json(client.get(f"/runs/{result['run_id']}/artifacts"))
    assert published
    saved = destination / result['run_id']
    shutil.copytree(directory, saved)
    return {'run_id': result['run_id'], 'experiment_id': result['experiment_id'],
            'state': result['state'], 'exit_code': result['exit_code'], 'forced': result['forced'],
            'steps': result['metrics']['steps'], 'updates': result['metrics']['updates'],
            'max_decision_steps': plan['max_decision_steps'], 'model': model, 'last_role_metrics': updates[-1]['metrics'],
            'artifact_directory': str(directory), 'copied_artifacts': str(saved),
            'display': display(client, result['run_id']), 'published_artifacts': published,
            'preview_applied': [e['payload'] for e in events if e['kind'] == 'preview_applied']}


@pytest.fixture(scope='module')
def api(tmp_path_factory):
    directory = tmp_path_factory.mktemp('basic-flight-e2e')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    report = {'scope': 'Real HTTP API and GPU Worker; browser evidence is separate', 'runs': {}}
    destination = ROOT / 'docs/validation/m1-e2e-artifacts'
    destination.mkdir(parents=True, exist_ok=True)
    with (directory / 'server.log').open('wb') as log:
        server = subprocess.Popen([sys.executable, '-m', 'muse_server.app', '--port', str(port),
            '--viewer-port', '0', '--data-dir', str(directory / 'data')], stdout=log, stderr=log,
            start_new_session=True, cwd=ROOT)
        try:
            with httpx2.Client(base_url=f'http://127.0.0.1:{port}', timeout=30., trust_env=False) as client:
                def healthy():
                    assert server.poll() is None, (directory / 'server.log').read_text()
                    try:
                        return client.get('/health').is_success
                    except httpx2.HTTPError:
                        return False
                wait_for(healthy)
                identity, experiment = register(client)
                report.update(experiment_id=identity, experiment=experiment, server_pid=server.pid, port=port)
                try:
                    yield client, identity, experiment, report, destination, server.pid
                finally:
                    for state in response_json(client.get('/runs')):
                        if state['state'] not in TERMINAL:
                            response_json(client.post(f"/runs/{state['run_id']}/cancel"))
                            terminal(client, state['run_id'])
                    wait_for(lambda: not response_json(client.get('/health'))['gpu_busy'])
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
            report['server_exit_code'] = server.returncode
            report['server_log'] = str(directory / 'server.log')
            (ROOT / 'docs/validation/m1-e2e.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def test_same_budget_headless_and_enabled_complete_two_ippo_updates(api):
    client, identity, experiment, report, destination, _ = api
    for name in ('headless', 'enabled'):
        started = start_run(client, identity, 8)
        run_id = started['run_id']
        if name == 'enabled':
            accepted = patch(client, run_id, enabled=True)
            assert accepted['requested']['enabled']
        else:
            assert not display(client, run_id)['requested']['enabled']
        result = terminal(client, run_id)
        summary = evidence(client, result, experiment, destination)
        summary['request_worker_state'] = started['state']
        report['runs'][name] = summary
        assert result['state'] == 'COMPLETED', result
        assert (summary['steps'], summary['updates']) == (8, 2)
        if name == 'enabled':
            assert any(ack['selection_seq'] == accepted['selection_seq'] and ack['display']['enabled']
                       for ack in summary['preview_applied'])
            assert summary['display']['effective']['enabled']
        print('E2E_BUDGET_RESULT', json.dumps(summary, allow_nan=False))
    assert report['runs']['headless']['model']['sha256'] == report['runs']['enabled']['model']['sha256']


def test_actual_gpu_viewer_failure_preserves_heartbeat_and_safe_cancel(api):
    client, identity, experiment, report, destination, server_pid = api
    started = start_run(client, identity, 100000)
    run_id = started['run_id']
    first = ready(client, run_id, patch(client, run_id, enabled=True)['selection_seq'])
    assert first['frame']['world_id'] == 0 and first['frame']['nbody'] == 4
    assert client.get(first['viewer_url']).is_success
    first_pid = viewer_pid(server_pid, run_id)
    switched = ready(client, run_id, patch(client, run_id, world_id=1)['selection_seq'])
    assert switched['frame']['world_id'] == 1
    disabled = ready(client, run_id, patch(client, run_id, enabled=False)['selection_seq'])
    assert not disabled['effective']['enabled'] and disabled['viewer_url'] is None
    reopened = ready(client, run_id, patch(client, run_id, enabled=True)['selection_seq'])
    assert reopened['frame']['world_id'] == 1
    assert reopened['frame']['scene_hash'] == first['frame']['scene_hash']
    second_pid = viewer_pid(server_pid, run_id)
    assert second_pid != first_pid
    before = run_state(client, run_id)
    os.kill(second_pid, signal.SIGKILL)
    wait_for(lambda: display(client, run_id)['status'] == 'error')
    failed_display = display(client, run_id)
    assert 'Viser' in failed_display['error'], failed_display
    def heartbeat_advanced():
        state = run_state(client, run_id)
        assert state['state'] == 'RUNNING' and state['pid'] == started['pid'], state
        return state if state['heartbeat'] > before['heartbeat'] else None
    after = wait_for(heartbeat_advanced)
    response_json(client.post(f'/runs/{run_id}/cancel'))
    result = terminal(client, run_id)
    summary = evidence(client, result, experiment, destination)
    summary.update(first=first, switched=switched, reopened=reopened,
                   viewer_failure=failed_display, worker_pid=started['pid'],
                   viewer_pids=[first_pid, second_pid], heartbeat_before=before['heartbeat'],
                   heartbeat_after=after['heartbeat'])
    report['runs']['lifecycle'] = summary
    assert result['state'] == 'CANCELLED' and result['cancel_requested'], result
    assert summary['steps'] < summary['max_decision_steps']
    assert 0 <= summary['steps'] - 4 * summary['updates'] <= 4
    assert summary['display']['viewer_url'] is None
    print('E2E_LIFECYCLE_RESULT', json.dumps(summary, allow_nan=False))
