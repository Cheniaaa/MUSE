"""真实Uvicorn与受控CPU Worker：实例锁、身份和停机队列语义。"""
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
