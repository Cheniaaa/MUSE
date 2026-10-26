import json
import os
from pathlib import Path
import subprocess
import signal
import sqlite3
import sys
import threading
import time

import pytest
from muse_sdk.run_protocol import RunRequest

ROOT = Path(__file__).resolve().parents[2]


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.02)
    raise AssertionError('Timed out waiting for process state')


def make_plan(tmp_path, run_id='run1', **timeouts):
    # 真正的冻结 Experiment 来自已有生产请求/resolver。
    from muse_server.catalog import Catalog
    from muse_server.resolver import resolve_experiment
    from test_experiment_pipeline import archive, EXAMPLE
    from muse_sdk.run_protocol import RunTimeouts
    metadata = tmp_path / 'metadata.json'
    subprocess.run([sys.executable, '-m', 'muse_builtins.metadata', '--output', str(metadata),
                    '--controller', str(EXAMPLE / 'controller.json')], check=True)
    catalog = Catalog(tmp_path / 'catalog', metadata)
    env = catalog.register_archive('environment', archive(EXAMPLE / 'environment', tmp_path / 'env.zip'))
    pkg = catalog.register_archive('training_package', archive(EXAMPLE / 'training_package', tmp_path / 'pkg.zip'))
    request = json.loads((EXAMPLE / 'request.json').read_text())
    request.update(environment_id=env.resource_id, environment_revision=env.revision,
                   training_package_id=pkg.resource_id, training_package_revision=pkg.revision)
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment, source=request)
    plan = catalog.create_run_plan(RunRequest(experiment_id=identity, mode='EVALUATE', max_decision_steps=8),
                                   device='cuda:0', timeouts=RunTimeouts(**timeouts))
    return catalog, plan


CHILD = r'''
import os, sys, time, json, signal
from pathlib import Path
from muse_runtime.protocol import WorkerProtocol
from muse_sdk.experiment import RunPlan
plan = RunPlan.model_validate_json(Path(sys.argv[1]).read_text())
p = WorkerProtocol(plan.run_id)
p.emit('preparing')
os.write(1, b'not JSON native stdout\n')
os.write(2, b'native stderr\n')
behavior = sys.argv[2]
if behavior in ('hang', 'failure_hang', 'cooperative'):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    p.emit('running')
    if behavior == 'failure_hang': p.emit('failure', {'message': 'original failure'})
    if behavior == 'cooperative':
        while p.read_control() != 'cancel': time.sleep(.01)
        p.emit('terminal', {'state': 'CANCELLED', 'steps': 1, 'updates': 0})
        sys.exit(0)
    while True: time.sleep(.1)
if behavior == 'compile':
    p.emit('compiling')
    while True:
        p.emit('heartbeat')
        time.sleep(.03)
if behavior == 'failure': p.emit('failure', {'message': 'first failure'})
if behavior != 'missing': p.emit('terminal', {'state': 'COMPLETED', 'steps': 8, 'updates': 0})
if behavior == 'eof_nonzero':
    p.close()
    time.sleep(.3)
    sys.exit(7)
sys.exit(0)
'''


def command(behavior):
    return lambda plan: [sys.executable, '-c', CHILD,
                         str(Path(plan.artifact_directory) / 'resolved_run_plan.json'), behavior]


@pytest.fixture(autouse=True)
def source_environment(monkeypatch):
    monkeypatch.setenv('PYTHONPATH', str(ROOT / 'src') + ':' + str(ROOT))
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')


@pytest.mark.parametrize('behavior,state,exit_code', [
    ('success', 'COMPLETED', 0), ('missing', 'FAILED', 0),
    ('failure', 'FAILED', 0), ('eof_nonzero', 'FAILED', 7),
])
def test_terminal_and_exit_priority_with_native_logs(tmp_path, behavior, state, exit_code):
    from muse_server.supervisor import Supervisor, TERMINAL_STATES
    catalog, plan = make_plan(tmp_path)
    with Supervisor(catalog, worker_command=command(behavior)) as supervisor:
        supervisor.submit(plan)
        record = wait_for(lambda: (r if (r := supervisor.get_run(plan.run_id))['state'] in TERMINAL_STATES else None))
        assert record['state'] == state
        assert record['exit_code'] == exit_code
        assert record['recording']['status'] == 'disabled'
        assert record['recording']['config'] is None
        log = Path(plan.artifact_directory, 'logs/worker.log').read_text()
        assert 'not JSON native stdout' in log and 'native stderr' in log
        events = supervisor.events(plan.run_id)
        assert events[-1]['kind'] == 'terminal' and events[-1]['payload']['state'] == state
        assert all(e['run_id'] == plan.run_id for e in events)
        assert {a['kind'] for a in supervisor.artifacts(plan.run_id)} >= {'log', 'run_plan', 'events'}


def test_eof_does_not_release_gpu_before_exit(tmp_path):
    from muse_server.supervisor import Supervisor
    catalog, plan = make_plan(tmp_path)
    with Supervisor(catalog, worker_command=command('eof_nonzero')) as supervisor:
        supervisor.submit(plan)
        wait_for(lambda: supervisor.get_run(plan.run_id)['worker_terminal'])
        assert supervisor.get_run(plan.run_id)['state'] not in ('COMPLETED', 'FAILED')
        assert supervisor.gpu_busy
        wait_for(lambda: supervisor.get_run(plan.run_id)['state'] == 'FAILED')
        assert not supervisor.gpu_busy


def test_cancel_timeout_waits_for_death(tmp_path):
    from muse_server.supervisor import Supervisor, process_identity
    catalog, plan = make_plan(tmp_path, cancel=.1, stop=.1)
    with Supervisor(catalog, worker_command=command('hang')) as supervisor:
        supervisor.submit(plan)
        record = wait_for(lambda: (r if (r := supervisor.get_run(plan.run_id))['state'] == 'RUNNING' else None))
        supervisor.cancel(plan.run_id)
        assert supervisor.gpu_busy
        done = wait_for(lambda: (r if (r := supervisor.get_run(plan.run_id))['state'] == 'CANCELLED' else None))
        assert done['exit_code'] != 0
        assert process_identity(record['pid']) is None
        assert not supervisor.gpu_busy


@pytest.mark.parametrize('behavior,timeout', [('hang', 'heartbeat'), ('compile', 'compile')])
def test_actual_timeouts_are_failures(tmp_path, behavior, timeout):
    from muse_server.supervisor import Supervisor
    catalog, plan = make_plan(tmp_path, heartbeat_interval=.05, **{timeout: .2, 'stop': .1})
    with Supervisor(catalog, worker_command=command(behavior)) as supervisor:
        supervisor.submit(plan)
        record = wait_for(lambda: (r if (r := supervisor.get_run(plan.run_id))['state'] == 'FAILED' else None))
        assert timeout in record['failure']


def test_os_lock_rejects_second_supervisor(tmp_path):
    from muse_server.supervisor import Supervisor
    catalog, _ = make_plan(tmp_path)
    with Supervisor(catalog):
        with pytest.raises(RuntimeError, match='supervisor'):
            Supervisor(catalog).start()


def test_wire_cpu_roundtrip_and_server_import_blockers():
    source = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('muse_runtime','muse_builtins','torch','mjlab','mujoco','warp') or fullname.startswith('_muse_package_'):
            raise RuntimeError('Forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
from muse_sdk.run_protocol import WorkerEvent, WorkerControl
from muse_server.supervisor import Supervisor
from muse_server.app import create_app
e = WorkerEvent(event_seq=1, run_id='r', kind='metric', sim_time=0.4, wall_time=1., payload={'steps': 10})
assert WorkerEvent.model_validate_json(e.model_dump_json()) == e
c = WorkerControl(run_id='r', kind='cancel')
assert WorkerControl.model_validate_json(c.model_dump_json()) == c
'''
    subprocess.run([sys.executable, '-c', source], check=True)


def test_queued_cancel_never_spawns_and_queue_waits_for_exit(tmp_path):
    from muse_server.supervisor import Supervisor
    catalog, plan = make_plan(tmp_path, cancel=.1, stop=.1)
    second = catalog.create_run_plan(RunRequest(experiment_id=plan.experiment_id, mode='EVALUATE', max_decision_steps=1), device='cuda:0')
    with Supervisor(catalog, worker_command=command('hang')) as supervisor:
        supervisor.submit(plan)
        wait_for(lambda: supervisor.get_run(plan.run_id)['state'] == 'RUNNING')
        supervisor.submit(second)
        time.sleep(.1)
        assert supervisor.get_run(second.run_id)['pid'] is None
        supervisor.cancel(second.run_id)
        assert supervisor.get_run(second.run_id)['state'] == 'CANCELLED'
        supervisor.cancel(plan.run_id)
        wait_for(lambda: supervisor.get_run(plan.run_id)['state'] == 'CANCELLED')
        assert supervisor.get_run(second.run_id)['pid'] is None


@pytest.mark.parametrize('close_on_restart', [False, True])
def test_restart_marks_lost_blocks_queue_until_old_process_dead(tmp_path, close_on_restart):
    from muse_server.supervisor import Supervisor, process_identity
    catalog, plan = make_plan(tmp_path, stop=.4)
    parent_source = '''
import sys, time
from pathlib import Path
from muse_server.catalog import Catalog
from muse_server.supervisor import Supervisor
from muse_sdk.experiment import RunPlan
catalog = Catalog(Path(sys.argv[1]), Path(sys.argv[1]) / 'builtin-metadata.json')
supervisor = Supervisor(catalog, worker_command=lambda p: [sys.executable, '-c',
    'import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); os.write(2,b"ready"); time.sleep(100)'])
supervisor.start()
supervisor.submit(RunPlan.model_validate_json(Path(sys.argv[2]).read_text()))
while True: time.sleep(.1)
'''
    parent = subprocess.Popen([sys.executable, '-c', parent_source, str(catalog.directory),
                               str(Path(plan.artifact_directory) / 'resolved_run_plan.json')])
    old_pid = None
    try:
        def persisted():
            try:
                with sqlite3.connect(catalog.database) as db:
                    row = db.execute('SELECT record FROM runs WHERE id=?', (plan.run_id,)).fetchone()
                    record = json.loads(row[0]) if row else None
                return record if record and record['pid'] else None
            except sqlite3.OperationalError:
                return None
        record = wait_for(persisted)
        old_pid = record['pid']
        wait_for(lambda: (p := Path(plan.artifact_directory, 'logs/worker.log')).exists() and p.read_text() == 'ready')
        parent.kill()
        parent.wait()
        second = catalog.create_run_plan(RunRequest(experiment_id=plan.experiment_id, mode='EVALUATE', max_decision_steps=1), device='cuda:0')
        with Supervisor(catalog, worker_command=command('success')) as supervisor:
            assert supervisor.get_run(plan.run_id)['state'] == 'LOST'
            supervisor.submit(second)
            assert supervisor.gpu_busy
            assert supervisor.get_run(second.run_id)['pid'] is None
            assert process_identity(old_pid) == record['process_start']
            if close_on_restart:
                supervisor.close()
                assert not supervisor.gpu_busy
                assert supervisor.get_run(second.run_id)['state'] == 'QUEUED'
                assert supervisor._lock_file.closed
            else:
                wait_for(lambda: supervisor.get_run(second.run_id)['state'] == 'COMPLETED')
            assert process_identity(old_pid) is None
            assert supervisor.events(plan.run_id)[-1]['payload']['state'] == 'LOST'
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if old_pid and process_identity(old_pid):
            os.killpg(old_pid, signal.SIGKILL)


def test_reused_pid_is_not_signalled_or_gpu_blocked(tmp_path):
    from muse_server.supervisor import Supervisor, process_identity
    catalog, plan = make_plan(tmp_path)
    with Supervisor(catalog, worker_command=command('success')) as supervisor:
        supervisor.submit(plan)
        wait_for(lambda: supervisor.get_run(plan.run_id)['state'] == 'COMPLETED')
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    try:
        with sqlite3.connect(catalog.database) as db:
            record = json.loads(db.execute('SELECT record FROM runs WHERE id=?', (plan.run_id,)).fetchone()[0])
            record.update(state='RUNNING', pid=other.pid, pgid=other.pid,
                          process_start=process_identity(other.pid) + '-old-start')
            db.execute('UPDATE runs SET record=? WHERE id=?', (json.dumps(record), plan.run_id))
        with Supervisor(catalog) as supervisor:
            wait_for(lambda: not supervisor.gpu_busy)
            assert supervisor.get_run(plan.run_id)['state'] == 'LOST'
            assert other.poll() is None
    finally:
        other.terminate()
        other.wait()


@pytest.mark.parametrize('identity_fault', ['unreadable', 'missing_start'])
def test_unconfirmed_worker_identity_blocks_queue_without_signalling(tmp_path, monkeypatch, identity_fault):
    from muse_server.supervisor import Supervisor, process_identity

    catalog, plan = make_plan(tmp_path, stop=.1)
    with Supervisor(catalog, worker_command=command('success')) as previous:
        previous.submit(plan)
        wait_for(lambda: previous.get_run(plan.run_id)['state'] == 'COMPLETED')
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    supervisor = Supervisor(catalog, worker_command=command('success'))
    observed = threading.Event()
    reads, signals = [], []
    original_read, original_killpg = Path.read_text, os.killpg
    try:
        identity = process_identity(other.pid)
        assert identity is not None
        with sqlite3.connect(catalog.database) as db:
            record = json.loads(db.execute('SELECT record FROM runs WHERE id=?', (plan.run_id,)).fetchone()[0])
            record.update(state='RUNNING', pid=other.pid, pgid=other.pid,
                          process_start=None if identity_fault == 'missing_start' else identity)
            db.execute('UPDATE runs SET record=? WHERE id=?', (json.dumps(record), plan.run_id))
        queued = catalog.create_run_plan(RunRequest(experiment_id=plan.experiment_id, mode='EVALUATE', max_decision_steps=1), device='cuda:0')

        def identity_read(path, *args, **kwargs):
            if path == Path(f'/proc/{other.pid}/stat'):
                reads.append(path)
                if len(reads) >= 2:
                    observed.set()
                if identity_fault == 'unreadable':
                    raise PermissionError('identity read denied for test')
            return original_read(path, *args, **kwargs)

        def signal_group(pgid, sig):
            signals.append((pgid, sig))
            return original_killpg(pgid, sig)

        with monkeypatch.context() as patch:
            patch.setattr(Path, 'read_text', identity_read)
            patch.setattr(os, 'killpg', signal_group)
            # 排队请求在第一次身份检查之前已被接收；监督线程本身保持生产实现。
            with supervisor._lock:
                supervisor.start()
                supervisor.submit(queued)
            assert observed.wait(5)
            if identity_fault == 'unreadable':
                wait_for(lambda: supervisor.error)
                assert 'identity read denied for test' in supervisor.error
                with pytest.raises(RuntimeError, match='shutdown unconfirmed'):
                    supervisor.close()
            else:
                assert supervisor.error is None
            assert supervisor.get_run(plan.run_id)['state'] == 'LOST'
            pending = supervisor.get_run(queued.run_id)
            assert pending['state'] == 'QUEUED' and pending['pid'] is None
            assert supervisor.gpu_busy and other.poll() is None
            assert signals == []
            with pytest.raises(RuntimeError, match='Another supervisor'):
                Supervisor(catalog).start()
    finally:
        # 结束自有测试进程后才释放测试故意保留的fatal/目录锁；不增加生产恢复入口。
        if other.poll() is None:
            other.terminate()
        other.wait(timeout=5)
        supervisor._fatal = None
        supervisor.close()


def test_startup_timeout_and_spawn_error_have_terminal(tmp_path):
    from muse_server.supervisor import Supervisor
    catalog, plan = make_plan(tmp_path, startup=.1, stop=.1)
    with Supervisor(catalog, worker_command=lambda p: [sys.executable, '-c', 'import time; time.sleep(30)']) as supervisor:
        supervisor.submit(plan)
        record = wait_for(lambda: (r if (r := supervisor.get_run(plan.run_id))['state'] == 'FAILED' else None))
        assert 'startup timeout' in record['failure']
    second = catalog.create_run_plan(RunRequest(experiment_id=plan.experiment_id, mode='EVALUATE', max_decision_steps=1), device='cuda:0')
    with Supervisor(catalog, worker_command=lambda p: ['/nonexistent/muse-worker']) as supervisor:
        supervisor.submit(second)
        record = wait_for(lambda: (r if (r := supervisor.get_run(second.run_id))['state'] == 'FAILED' else None))
        assert 'spawn' in record['failure']
        assert not supervisor.gpu_busy and supervisor.error is None


def test_slow_consumer_keeps_latest_metrics_and_reliable_failure_terminal(tmp_path):
    from muse_server.supervisor import Supervisor
    catalog, plan = make_plan(tmp_path)
    source = CHILD.replace("behavior = sys.argv[2]", "behavior = sys.argv[2]\nfor i in range(1000): p.emit('metric', {'steps': i})")
    with Supervisor(catalog, worker_command=lambda p: [sys.executable, '-c', source,
            str(Path(p.artifact_directory) / 'resolved_run_plan.json'), 'failure']) as supervisor:
        supervisor.submit(plan)
        wait_for(lambda: supervisor.get_run(plan.run_id)['state'] == 'FAILED')
        events = supervisor.events(plan.run_id)
        assert len(events) <= 7
        assert next(e for e in events if e['kind'] == 'metric')['payload']['steps'] == 999
        assert any(e['kind'] == 'failure' for e in events)
        assert events[-1]['kind'] == 'terminal'


def test_worker_import_failure_keeps_native_output_out_of_protocol(tmp_path):
    from muse_server.supervisor import Supervisor
    catalog, plan = make_plan(tmp_path)
    Path(plan.resource_paths['training_package'], '__init__.py').write_text(
        "import os\nos.write(1,b'import native stdout\\n')\nos.write(2,b'import native stderr\\n')\nraise RuntimeError('user import failure')\n")
    with Supervisor(catalog) as supervisor:
        supervisor.submit(plan)
        record = wait_for(lambda: (r if (r := supervisor.get_run(plan.run_id))['state'] == 'FAILED' else None))
        assert record['exit_code'] == 1 and record['failure'] == 'user import failure'
        assert record['worker_terminal']['state'] == 'FAILED'
        log = Path(plan.artifact_directory, 'logs/worker.log').read_text()
        assert 'import native stdout' in log and 'import native stderr' in log


def test_cancel_during_final_update_is_seen_at_update_boundary():
    from muse_runtime.driver import run
    from muse_sdk.runtime_types import PolicyInput
    from types import SimpleNamespace
    cancelled = False
    class Algorithm:
        def set_mode(self, mode): pass
        def act(self, *args, **kwargs): return SimpleNamespace(role_actions={}, act_aux=None)
        def process_env_step(self, *args): pass
        def update(self):
            nonlocal cancelled
            cancelled = True
            return {}
    env = SimpleNamespace(policy_input=PolicyInput({}, {}), rng=None,
        step=lambda actions: SimpleNamespace(transition=None, policy_observations={}, policy_role_masks={}, diagnostics={}))
    result = run(env, Algorithm(), lambda: 'cancel' if cancelled else None, lambda *args: None,
                 mode='TRAIN', collection_steps=1, max_decision_steps=1)
    assert result == {'state': 'CANCELLED', 'steps': 1, 'updates': 1}


def test_sdk_imports_without_server_or_gpu_and_rejects_nonfinite_wire():
    source = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('muse_runtime','muse_builtins','muse_server','torch','fastapi','sqlite3'):
            raise RuntimeError('Forbidden: ' + fullname)
sys.meta_path.insert(0, Block())
from muse_sdk.run_protocol import WorkerEvent
try:
    WorkerEvent.model_validate_json('{"event_seq":1,"run_id":"r","kind":"metric","sim_time":0,"wall_time":1,"payload":{"loss":NaN}}')
except ValueError:
    pass
else:
    raise AssertionError('Nonfinite metric must not silently become null')
'''
    subprocess.run([sys.executable, '-c', source], check=True)


@pytest.mark.parametrize('behavior,state', [('cooperative', 'CANCELLED'), ('failure_hang', 'FAILED')])
def test_cancel_control_and_failure_priority(tmp_path, behavior, state):
    from muse_server.supervisor import Supervisor
    catalog, plan = make_plan(tmp_path, cancel=.1, stop=.1)
    with Supervisor(catalog, worker_command=command(behavior)) as supervisor:
        supervisor.submit(plan)
        wait_for(lambda: supervisor.get_run(plan.run_id)['state'] == 'RUNNING')
        if behavior == 'failure_hang':
            wait_for(lambda: supervisor.get_run(plan.run_id)['failure'])
        supervisor.cancel(plan.run_id)
        record = wait_for(lambda: (r if (r := supervisor.get_run(plan.run_id))['state'] == state else None))
        if behavior == 'cooperative':
            assert record['exit_code'] == 0 and not record['forced']
        else:
            assert record['failure'] == 'original failure'


def test_large_native_output_is_continuously_drained(tmp_path):
    from muse_server.supervisor import Supervisor
    catalog, plan = make_plan(tmp_path)
    source = CHILD.replace("behavior = sys.argv[2]", "os.write(1, b'x' * (2 * 1024 * 1024))\nbehavior = sys.argv[2]")
    with Supervisor(catalog, worker_command=lambda p: [sys.executable, '-c', source,
                         str(Path(p.artifact_directory) / 'resolved_run_plan.json'), 'success']) as supervisor:
        supervisor.submit(plan)
        wait_for(lambda: supervisor.get_run(plan.run_id)['state'] == 'COMPLETED')
        assert Path(plan.artifact_directory, 'logs/worker.log').stat().st_size > 2 * 1024 * 1024


def test_empty_failure_still_wins_over_success(tmp_path):
    from muse_server.supervisor import Supervisor
    catalog, plan = make_plan(tmp_path)
    source = CHILD.replace("'first failure'", "''")
    with Supervisor(catalog, worker_command=lambda p: [sys.executable, '-c', source,
                         str(Path(p.artifact_directory) / 'resolved_run_plan.json'), 'failure']) as supervisor:
        supervisor.submit(plan)
        record = wait_for(lambda: (r if (r := supervisor.get_run(plan.run_id))['exit_code'] is not None else None))
        assert record['state'] == 'FAILED'


def test_nonzero_exception_exit_after_cancel_timeout_is_not_cancelled(tmp_path):
    from muse_server.supervisor import Supervisor
    catalog, plan = make_plan(tmp_path, cancel=.05, stop=1.)
    source = CHILD.replace("p.emit('terminal', {'state': 'CANCELLED', 'steps': 1, 'updates': 0})",
                           "time.sleep(.2); sys.exit(7)")
    with Supervisor(catalog, worker_command=lambda p: [sys.executable, '-c', source,
                         str(Path(p.artifact_directory) / 'resolved_run_plan.json'), 'cooperative']) as supervisor:
        supervisor.submit(plan)
        wait_for(lambda: supervisor.get_run(plan.run_id)['state'] == 'RUNNING')
        supervisor.cancel(plan.run_id)
        record = wait_for(lambda: (r if (r := supervisor.get_run(plan.run_id))['exit_code'] is not None else None))
        assert record['state'] == 'FAILED' and record['exit_code'] == 7


@pytest.mark.parametrize('behavior,timeouts,already_stopping', [
    pytest.param('hang', {'cancel': .1, 'stop': 31.}, False, id='long-stop'),
    pytest.param('cooperative', {}, False, id='default-cooperative'),
    pytest.param('failure_hang', {'cancel': .1, 'stop': .2}, True, id='already-stopping'),
])
def test_close_waits_for_worker_death_and_releases_lock(tmp_path, behavior, timeouts, already_stopping):
    from muse_server.supervisor import Supervisor, process_identity
    catalog, plan = make_plan(tmp_path, **timeouts)
    supervisor = Supervisor(catalog, worker_command=command(behavior)).start()
    try:
        supervisor.submit(plan)
        record = wait_for(lambda: (r if (r := supervisor.get_run(plan.run_id))['state'] == 'RUNNING' else None))
        if already_stopping:
            wait_for(lambda: supervisor.get_run(plan.run_id)['failure'] == 'original failure')
            supervisor.cancel(plan.run_id)
            wait_for(lambda: supervisor.get_run(plan.run_id)['state'] == 'STOPPING')
        process = supervisor._active['process']
        started = time.monotonic()
        supervisor.close()
        elapsed = time.monotonic() - started
        done = supervisor.get_run(plan.run_id)
        assert done['state'] == ('FAILED' if already_stopping else 'CANCELLED')
        assert done['failure'] == ('original failure' if already_stopping else None)
        assert process.poll() is not None and process_identity(record['pid']) is None
        assert not supervisor.gpu_busy and supervisor._thread is None
        assert supervisor._lock_file.closed
        if behavior == 'hang':
            assert elapsed >= plan.timeouts.cancel + plan.timeouts.stop
            assert done['exit_code'] == -signal.SIGKILL
        with Supervisor(catalog):
            pass
    finally:
        # RED 时等待原监督流程完成，确保失败测试也不留下子进程。
        supervisor.close()


def test_close_reports_supervisor_failure_and_retains_lock(tmp_path, monkeypatch):
    from muse_server.supervisor import Supervisor
    catalog, _ = make_plan(tmp_path)
    supervisor = Supervisor(catalog)
    def fail_tick():
        raise ValueError('supervision failed')
    monkeypatch.setattr(supervisor, '_tick', fail_tick)
    supervisor.start()
    try:
        wait_for(lambda: supervisor.error)
        with pytest.raises(RuntimeError, match="Supervisor shutdown unconfirmed: ValueError.*supervision failed"):
            supervisor.close()
        assert not supervisor._thread.is_alive()
        assert not supervisor._lock_file.closed
        with pytest.raises(RuntimeError, match='Another supervisor'):
            Supervisor(catalog).start()
    finally:
        supervisor._fatal = None
        supervisor.close()
