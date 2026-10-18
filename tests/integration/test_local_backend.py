"""本机CLI身份边界：真实Linux子进程，不启动Server、Worker或GPU。"""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


HEARTBEAT = '''
import json, os, signal, sys, time
from pathlib import Path
from muse_server.supervisor import process_identity
root, behavior, budget = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
instance = root / 'server-instance.json'
requested = False
def terminate(signum, frame):
    global requested
    (root / 'signal-received').write_text(str(signum))
    requested = True
signal.signal(signal.SIGTERM, terminate)
instance.write_text(json.dumps(dict(instance_id='heartbeat-instance', pid=os.getpid(),
    process_start=process_identity(os.getpid()), address='http://127.0.0.1:8000',
    state='running', shutdown_timeout=budget)))
print('ready', flush=True)
with (root / 'heartbeat').open('a', buffering=1) as output:
    while not (root / 'exit-request').exists():
        output.write(str(time.monotonic()) + '\\n')
        if requested and behavior != 'ignore':
            if behavior == 'clean':
                time.sleep(.1)
                instance.unlink()
                (root / 'metadata-cleared').touch()
                time.sleep(.2)
            break
        time.sleep(.01)
'''


@contextmanager
def heartbeat(root, *, behavior='clean', budget=2.):
    root.mkdir()
    process = subprocess.Popen([sys.executable, '-u', '-c', HEARTBEAT, str(root), behavior, str(budget)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline() == 'ready\n'
        yield process
    finally:
        # 独立控制文件只用于本测试子进程回收；不强杀，也不调用待测stop兜底。
        (root / 'exit-request').touch()
        process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


def cli(*arguments):
    return subprocess.run([sys.executable, '-m', 'muse_server.local_backend', *map(str, arguments)],
                          capture_output=True, text=True, timeout=10)


def test_status_without_instance_file_is_stopped(tmp_path):
    from muse_server.local_backend import read_status
    assert read_status(tmp_path / 'absent') == {'state': 'stopped'}
    result = cli('status', '--data-dir', tmp_path / 'absent')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'state': 'stopped'}


def test_status_requires_live_process_identity(tmp_path):
    from muse_server.local_backend import read_status
    root = tmp_path / '心跳 数据'
    with heartbeat(root) as process:
        status = read_status(root)
        assert status['state'] == 'running' and status['pid'] == process.pid
        assert status['instance_id'] == 'heartbeat-instance'
        assert status['address'] == 'http://127.0.0.1:8000'
        result = cli('status', '--data-dir', root)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == status


@pytest.mark.parametrize('mismatch', ['instance_id', 'process_start'])
def test_stop_rejects_wrong_identity_without_signaling(tmp_path, mismatch):
    from muse_sdk.config import ConfigError
    from muse_server.local_backend import read_status, stop_backend
    root = tmp_path / mismatch
    with heartbeat(root) as process:
        expected = 'heartbeat-instance'
        if mismatch == 'instance_id':
            expected = 'another-instance'
        else:
            path = root / 'server-instance.json'
            metadata = json.loads(path.read_text())
            metadata['process_start'] += '-wrong'
            path.write_text(json.dumps(metadata))
            assert read_status(root)['state'] == 'stale'
        with pytest.raises(ConfigError) as error:
            stop_backend(root, expected)
        assert error.value.errors[0]['field'] == mismatch
        result = cli('stop', '--data-dir', root, '--instance-id', expected)
        assert result.returncode != 0
        assert json.loads(result.stderr)['errors'][0]['field'] == mismatch
        before = (root / 'heartbeat').read_text()
        time.sleep(.05)
        assert (root / 'heartbeat').read_text() != before
        assert process.poll() is None and not (root / 'signal-received').exists()


def test_correct_stop_waits_for_cleanup_and_real_process_exit(tmp_path):
    from muse_server.local_backend import stop_backend
    root = tmp_path / 'clean'
    with heartbeat(root) as process:
        started = time.monotonic()
        result = stop_backend(root, 'heartbeat-instance')
        assert result['state'] == 'stopped' and result['instance_id'] == 'heartbeat-instance'
        assert time.monotonic() - started >= .25
        assert (root / 'metadata-cleared').exists()
        assert not (root / 'server-instance.json').exists()
        assert process.wait(timeout=0) == 0
        assert (root / 'signal-received').read_text() == '15'


def test_cli_stop_waits_and_returns_json(tmp_path):
    root = tmp_path / 'cli clean'
    with heartbeat(root) as process:
        result = cli('stop', '--data-dir', root, '--instance-id', 'heartbeat-instance')
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)['state'] == 'stopped'
        assert process.wait(timeout=0) == 0
        assert not (root / 'server-instance.json').exists()


@pytest.mark.parametrize('use_cli', [False, True])
def test_shutdown_timeout_is_unconfirmed_and_never_forces_exit(tmp_path, use_cli):
    from muse_server.local_backend import stop_backend
    root = tmp_path / 'ignore'
    with heartbeat(root, behavior='ignore', budget=.15) as process:
        started = time.monotonic()
        if use_cli:
            result = cli('stop', '--data-dir', root, '--instance-id', 'heartbeat-instance')
            assert result.returncode != 0
            status = json.loads(result.stdout)
        else:
            status = stop_backend(root, 'heartbeat-instance')
        assert status['state'] == 'unconfirmed'
        assert .12 <= time.monotonic() - started < 3
        assert process.poll() is None
        assert (root / 'server-instance.json').exists()
        assert (root / 'signal-received').read_text() == '15'
        before = (root / 'heartbeat').read_text()
        time.sleep(.05)
        assert (root / 'heartbeat').read_text() != before


def test_exited_process_without_metadata_cleanup_is_not_success(tmp_path):
    from muse_server.local_backend import read_status, stop_backend
    root = tmp_path / 'no cleanup'
    with heartbeat(root, behavior='leave_metadata') as process:
        result = stop_backend(root, 'heartbeat-instance')
        assert result['state'] == 'unconfirmed'
        assert process.wait(timeout=0) == 0
        assert read_status(root)['state'] == 'stale'
        assert (root / 'server-instance.json').exists()


def test_stop_rejects_dead_process_identity(tmp_path):
    from muse_sdk.config import ConfigError
    from muse_server.local_backend import read_status, stop_backend
    root = tmp_path / 'exited'
    with heartbeat(root, behavior='leave_metadata') as process:
        (root / 'exit-request').touch()
        process.wait(timeout=5)
        assert read_status(root)['state'] == 'stale'
        with pytest.raises(ConfigError) as error:
            stop_backend(root, 'heartbeat-instance')
        assert error.value.errors[0]['field'] == 'process_start'


def test_stop_without_metadata_is_not_reported_as_success(tmp_path):
    from muse_sdk.config import ConfigError
    from muse_server.local_backend import stop_backend
    with pytest.raises(ConfigError) as error:
        stop_backend(tmp_path, 'missing')
    assert error.value.errors[0]['field'] == 'instance_id'


def test_start_forwards_argv_to_single_app_entrypoint(monkeypatch, tmp_path):
    from muse_server import app, local_backend
    forwarded = []
    monkeypatch.setattr(app, 'main', lambda argv: forwarded.append(argv))
    arguments = ['--data-dir', str(tmp_path / '中文 ; $() 空格'), '--host', '127.0.0.1',
                 '--port', '18000', '--access-host', 'localhost', '--viewer-port', '18080']
    assert local_backend.main(['start', *arguments]) == 0
    assert forwarded == [arguments]
