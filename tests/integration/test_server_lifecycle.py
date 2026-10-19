"""真实Uvicorn与受控CPU Worker：实例锁、身份和停机队列语义。"""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import socket
import signal
import sqlite3
import subprocess
import sys

import httpx2
import pytest

from test_supervisor import ROOT, make_plan, wait_for


@pytest.fixture
def local_server(tmp_path):
    catalog, plan = make_plan(tmp_path)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    address = f'http://127.0.0.1:{port}'
    script = '''import sys, uvicorn
from muse_server.app import create_app
from test_supervisor import command
uvicorn.run(create_app(sys.argv[1], worker_command=command('cooperative'),
                      server_address=sys.argv[3]), host='127.0.0.1', port=int(sys.argv[2]))
'''
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join((str(ROOT / 'src'), str(ROOT / 'tests/integration'))),
           'PYTHONDONTWRITEBYTECODE': '1'}
    with (tmp_path / 'server.log').open('wb') as log:
        process = subprocess.Popen([sys.executable, '-c', script, str(catalog.directory), str(port), address],
                                   stdout=log, stderr=log, env=env)
        try:
            with httpx2.Client(base_url=address, timeout=5) as client:
                def healthy():
                    if process.poll() is not None:
                        raise AssertionError((tmp_path / 'server.log').read_text())
                    try:
                        return client.get('/health').is_success
                    except httpx2.ConnectError:
                        return False
                wait_for(healthy, timeout=20)
                yield process, client, catalog, plan
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=20)


def test_stop_checks_instance_and_preserves_queue(local_server):
    from muse_server.local_backend import read_status, stop_backend
    from muse_server.supervisor import process_identity
    process, client, catalog, plan = local_server
    status = read_status(catalog.directory)
    health = client.get('/health').json()
    assert health['instance_id'] == status['instance_id']
    assert status['state'] == 'running' and status['pid'] == process.pid
    active = client.post('/runs', json=dict(experiment_id=plan.experiment_id, mode='EVALUATE', max_decision_steps=8)).json()
    wait_for(lambda: client.get('/runs/' + active['run_id']).json()['state'] == 'RUNNING')
    queued = client.post('/runs', json=dict(experiment_id=plan.experiment_id, mode='EVALUATE', max_decision_steps=8)).json()
    with pytest.raises(ValueError):
        stop_backend(catalog.directory, 'wrong-instance')
    assert client.get('/health').status_code == 200
    result = stop_backend(catalog.directory, status['instance_id'])
    assert result['state'] == 'stopped'
    # 固定Uvicorn在lifespan清理后重发捕获的SIGTERM，子进程码保留原信号。
    assert process.wait(timeout=5) == -signal.SIGTERM
    with sqlite3.connect(catalog.database) as db:
        records = {row[0]: json.loads(row[1]) for row in db.execute('SELECT id,record FROM runs')}
    assert records[active['run_id']]['state'] == 'CANCELLED'
    assert records[queued['run_id']]['state'] == 'QUEUED' and records[queued['run_id']]['pid'] is None
    assert process_identity(records[active['run_id']]['pid']) is None
    assert not (catalog.directory / 'server-instance.json').exists()


def test_two_starts_share_directory_lock(local_server, tmp_path):
    process, client, catalog, _ = local_server
    original = (catalog.directory / 'server-instance.json').read_bytes()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        other_port = probe.getsockname()[1]
    second = subprocess.run([sys.executable, '-m', 'muse_server.local_backend', 'start',
        '--data-dir', str(catalog.directory), '--host', '127.0.0.1', '--port', str(other_port),
        '--access-host', '127.0.0.1', '--viewer-port', '0'], capture_output=True, text=True, timeout=20)
    assert second.returncode != 0
    assert 'owns this data directory' in second.stderr
    assert process.poll() is None and client.get('/health').status_code == 200
    assert (catalog.directory / 'server-instance.json').read_bytes() == original


def test_restart_preserves_existing_queue_terminal_and_public_history(tmp_path):
    from websockets.sync.client import connect
    from muse_server.supervisor import process_identity

    catalog, plan = make_plan(tmp_path)
    script = '''import sys, uvicorn
from muse_sdk.run_protocol import RunTimeouts
from muse_server.app import create_app
from test_supervisor import command
def worker(plan):
    args = command('hang' if plan.mode == 'INTERACTIVE' else 'success')(plan)
    args[2] = args[2].replace('behavior = sys.argv[2]',
        "p.emit('metric', {'steps': 8, 'updates': 0, 'value': 1.25})\\nbehavior = sys.argv[2]")
    return args
uvicorn.run(create_app(sys.argv[1], worker_command=worker,
    timeouts=RunTimeouts(stop=60., heartbeat=60.), server_address=sys.argv[3]),
    host='127.0.0.1', port=int(sys.argv[2]))
'''
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join((str(ROOT / 'src'), str(ROOT / 'tests/integration'))),
           'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''}
    servers, active = [], None
    with ExitStack() as stack:
        def start_server():
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            address = f'http://127.0.0.1:{port}'
            path = tmp_path / f'server-{len(servers)}.log'
            log = stack.enter_context(path.open('wb'))
            process = subprocess.Popen([sys.executable, '-c', script, str(catalog.directory), str(port), address],
                                       stdout=log, stderr=log, env=env)
            servers.append(process)
            client = stack.enter_context(httpx2.Client(base_url=address, timeout=5))
            def healthy():
                if process.poll() is not None:
                    raise AssertionError(path.read_text())
                try:
                    return client.get('/health').is_success
                except httpx2.ConnectError:
                    return False
            wait_for(healthy, timeout=20)
            return process, client, port

        try:
            server, client, _ = start_server()
            original_health = client.get('/health').json()
            request = dict(experiment_id=plan.experiment_id, mode='EVALUATE', max_decision_steps=8)
            completed = client.post('/runs', json=request).json()['run_id']
            terminal = wait_for(lambda: (r if (r := client.get(f'/runs/{completed}').json())['state'] == 'COMPLETED' else None))
            source = client.get(f'/experiments/{plan.experiment_id}/source').json()
            artifacts = client.get(f'/runs/{completed}/artifacts').json()
            assert {a['kind'] for a in artifacts} >= {'run_plan', 'log', 'events', 'metrics'}
            contents = {a['id']: client.get(f"/runs/{completed}/artifacts/{a['id']}").content for a in artifacts}
            metrics = client.get(f'/runs/{completed}/metrics').json()
            assert [event['payload'] for event in metrics['records']] == [{'steps': 8, 'updates': 0, 'value': 1.25}]
            logs = client.get(f'/runs/{completed}/logs').json()
            active = client.post('/runs', json={**request, 'mode': 'INTERACTIVE'}).json()['run_id']
            orphan = wait_for(lambda: (r if (r := client.get(f'/runs/{active}').json())['state'] == 'RUNNING' else None))
            queued = client.post('/runs', json=request).json()
            assert queued['state'] == 'QUEUED' and queued['pid'] is None
            server.kill()
            assert server.wait(timeout=5) == -signal.SIGKILL
            assert process_identity(orphan['pid']) == orphan['process_start']

            _, reopened, port = start_server()
            health = reopened.get('/health').json()
            assert health['catalog_id'] == original_health['catalog_id']
            assert health['instance_id'] != original_health['instance_id']
            assert health['gpu_busy']
            lost = reopened.get(f'/runs/{active}').json()
            assert lost['state'] == 'LOST' and lost['failure']
            assert lost['pid'] == orphan['pid']
            assert reopened.get(f'/runs/{completed}').json() == terminal
            assert reopened.get(f'/experiments/{plan.experiment_id}/source').json() == source
            assert reopened.get(f'/runs/{completed}/artifacts').json() == artifacts
            for artifact in artifacts:
                response = reopened.get(f"/runs/{completed}/artifacts/{artifact['id']}")
                assert response.status_code == 200 and response.content == contents[artifact['id']]
                assert len(response.content) == artifact['size']
                assert hashlib.sha256(response.content).hexdigest() == artifact['sha256']
            assert reopened.get(f'/runs/{completed}/metrics').json() == metrics
            assert reopened.get(f'/runs/{completed}/logs').json() == logs
            for run_id, state in ((completed, 'COMPLETED'), (active, 'LOST')):
                with connect(f'ws://127.0.0.1:{port}/streams/runs/{run_id}/events', open_timeout=5) as ws:
                    event = json.loads(ws.recv(timeout=5))
                    assert event['kind'] == 'terminal' and event['run_id'] == run_id
                    assert event['payload']['state'] == state
                display = reopened.get(f'/runs/{run_id}/display').json()
                assert not display['effective']['enabled'] and display['viewer_url'] is None
                assert not reopened.get(f'/runs/{run_id}/entities').json()['available']
            records = {r['run_id']: r for r in reopened.get('/runs').json()['items']}
            assert set(records) == {completed, active, queued['run_id']}
            assert records[queued['run_id']]['state'] == 'QUEUED'
            assert records[queued['run_id']]['pid'] is None
            assert process_identity(orphan['pid']) == orphan['process_start']
            # 故意不响应cancel/SIGTERM的CPU Worker由测试显式结束；Server不能在此之前放行队列。
            os.killpg(orphan['pgid'], signal.SIGKILL)
            wait_for(lambda: process_identity(orphan['pid']) is None)
            finished = wait_for(lambda: (r if (r := reopened.get(f"/runs/{queued['run_id']}").json())['state'] == 'COMPLETED' else None))
            assert finished['pid'] is not None and finished['exit_code'] == 0
            assert reopened.get(f'/runs/{active}').json()['state'] == 'LOST'
            assert not reopened.get('/health').json()['gpu_busy']
        finally:
            try:
                # RUNNING等待可能失败；清理身份从已创建Run的持久记录获取，不依赖等待成功。
                if active is not None:
                    with sqlite3.connect(catalog.database) as db:
                        record = json.loads(db.execute('SELECT record FROM runs WHERE id=?', (active,)).fetchone()[0])
                    if record['pid'] is not None and process_identity(record['pid']) == record['process_start']:
                        os.killpg(record['pgid'], signal.SIGKILL)
                        wait_for(lambda: process_identity(record['pid']) is None)
            finally:
                for process in reversed(servers):
                    if process.poll() is None:
                        process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
