"""CPU 子进程验证控制面；不启动物理或 GPU。"""
import json
from pathlib import Path
import sys

import pytest

from muse_sdk.run_protocol import InteractiveCommand, RunRequest, WorkerControl, WorkerEvent
from muse_server.errors import ResourceNotFound
from muse_server.supervisor import Supervisor
from test_supervisor import make_plan, wait_for


CHILD = '''
import sys, time
from muse_runtime.protocol import WorkerProtocol
p = WorkerProtocol(sys.argv[1])
p.emit('running')
count = 0
paused = False
while p.read_control() != 'cancel':
    command = p.take_command()
    if command:
        time.sleep(.1)
        error = 'reset_world requires pause' if command.command == 'reset_world' and not paused else None
        if not error:
            count += 1
            paused = command.command != 'resume'
        p.emit('command_result', dict(**command.model_dump(), status='rejected' if error else 'applied',
               paused=paused, steps=count, updates=0, error=error))
    p.emit('heartbeat')
    time.sleep(.01)
p.emit('terminal', {'state': 'CANCELLED', 'steps': count, 'updates': 0})
'''


@pytest.fixture(scope='module')
def catalog(tmp_path_factory):
    return make_plan(tmp_path_factory.mktemp('interactive-catalog'))[0]


@pytest.fixture
def live(catalog):
    # 独立验证 Supervisor；准入未实现前仍可构造 SDK 已支持的冻结计划。
    with catalog._connect() as db:
        identity = db.execute('SELECT id FROM experiments LIMIT 1').fetchone()[0]
    plan = catalog.create_run_plan(RunRequest(experiment_id=identity, mode='EVALUATE', max_decision_steps=8), device='cpu')
    plan = plan.model_copy(update={'mode': 'INTERACTIVE'})
    Path(plan.artifact_directory, 'resolved_run_plan.json').write_text(plan.model_dump_json())
    with Supervisor(catalog, worker_command=lambda p: [sys.executable, '-c', CHILD, p.run_id]) as supervisor:
        record = supervisor.submit(plan)
        wait_for(lambda: supervisor.get_run(plan.run_id)['state'] == 'RUNNING')
        yield supervisor, plan, record


def cmd(seq=1, command='pause', world_id=None):
    return InteractiveCommand(command_seq=seq, command=command, world_id=world_id)


def test_pending_dedup_conflicts_ack_and_next_sequence(live):
    supervisor, plan, record = live
    assert record['interaction'] == {'command': None, 'result': None}
    frozen = Path(plan.artifact_directory, 'resolved_run_plan.json').read_bytes()
    with supervisor._lock:
        pending = supervisor.submit_command(plan.run_id, cmd())
        assert pending == {'command': cmd().model_dump(), 'result': None}
        assert supervisor.submit_command(plan.run_id, cmd()) == pending
        for invalid in (cmd(1, 'resume'), cmd(2), cmd(3)):
            with pytest.raises(ValueError):
                supervisor.submit_command(plan.run_id, invalid)
        assert supervisor.get_run(plan.run_id)['interaction'] == pending
    result = wait_for(lambda: supervisor.get_run(plan.run_id)['interaction']['result'])
    assert result == dict(command_seq=1, command='pause', world_id=None,
                         status='applied', paused=True, steps=1, updates=0, error=None)
    assert supervisor.submit_command(plan.run_id, cmd())['result'] == result
    supervisor.submit_command(plan.run_id, cmd(2, 'reset_world', 1))
    wait_for(lambda: supervisor.get_run(plan.run_id)['interaction']['result'])
    with pytest.raises(ValueError):
        supervisor.submit_command(plan.run_id, cmd())
    assert Path(plan.artifact_directory, 'resolved_run_plan.json').read_bytes() == frozen


@pytest.mark.parametrize('phase', ['PREPARING', 'COMPILING', 'RUNNING'])
def test_live_startup_phases_accept_pause(live, phase):
    supervisor, plan, _ = live
    with supervisor._lock:
        supervisor._active['record']['state'] = phase
        supervisor._save(supervisor._active['record'])
        assert supervisor.submit_command(plan.run_id, cmd())['result'] is None


@pytest.mark.parametrize('change', [
    {'state': 'QUEUED'}, {'state': 'STOP_REQUESTED'}, {'state': 'STOPPING'},
    {'state': 'COMPLETED'}, {'state': 'CANCELLED'}, {'state': 'FAILED'}, {'state': 'LOST'},
    {'cancel_requested': True}, {'worker_terminal': {'state': 'COMPLETED'}}, {'failure': 'failed'},
])
def test_nonactive_or_terminating_record_rejects(live, change):
    supervisor, plan, _ = live
    with supervisor._lock:
        record = supervisor._active['record']
        previous = record.copy()
        try:
            record.update(change)
            supervisor._save(record)
            with pytest.raises(ValueError):
                supervisor.submit_command(plan.run_id, cmd())
            assert not supervisor._active['control']
        finally:
            record.clear()
            record.update(previous)
            supervisor._save(record)


@pytest.mark.parametrize('mode', ['TRAIN', 'EVALUATE'])
def test_other_modes_cannot_use_api_or_viewer_commands(live, mode):
    supervisor, plan, _ = live
    with supervisor._lock:
        supervisor._active['plan'] = plan.model_copy(update={'mode': mode})
        with pytest.raises(ValueError, match='INTERACTIVE'):
            supervisor.submit_command(plan.run_id, cmd())
        with pytest.raises(ValueError, match='INTERACTIVE'):
            supervisor.viewer_command(plan.run_id, 'pause')
        with pytest.raises(ValueError, match='INTERACTIVE'):
            supervisor.send_control(plan.run_id, WorkerControl(run_id=plan.run_id, kind='command', command=cmd()))


def test_missing_worker_world_bound_and_ahead_sequence(live):
    supervisor, plan, _ = live
    with supervisor._lock:
        for request in (cmd(1, 'reset_world', 2), cmd(2)):
            with pytest.raises(ValueError):
                supervisor.submit_command(plan.run_id, request)
        active = supervisor._active
        supervisor._active = None
        try:
            with pytest.raises(ValueError):
                supervisor.submit_command(plan.run_id, cmd())
        finally:
            supervisor._active = active
        with pytest.raises(ResourceNotFound):
            supervisor.submit_command('missing', cmd())


@pytest.mark.parametrize('payload_change', [
    {'command_seq': 2}, {'command': 'resume'}, {'command': 'reset_world', 'world_id': 1},
])
def test_ack_identity_must_match_pending(live, payload_change):
    supervisor, plan, _ = live
    with supervisor._lock:
        supervisor.submit_command(plan.run_id, cmd())
        payload = dict(**cmd().model_dump(), status='applied', paused=True, steps=0, updates=0)
        payload.update(payload_change)
        event = WorkerEvent(event_seq=supervisor._active['record']['event_seq'] + 1,
                           run_id=plan.run_id, kind='command_result', sim_time=0, wall_time=1, payload=payload)
        with pytest.raises(ValueError, match='command'):
            supervisor._event(event.model_dump_json().encode())
        assert supervisor.get_run(plan.run_id)['interaction']['result'] is None


def test_viewer_uses_same_pending_sequence_and_rejected_ack(live):
    supervisor, plan, _ = live
    with supervisor._lock:
        session = supervisor._display(plan.run_id)
        slot = session.slots['viser']
        session._viewer_event(slot, {'kind': 'command', 'command': 'reset_world', 'world_id': 1})
        pending = supervisor.get_run(plan.run_id)['interaction']
        assert pending == {'command': cmd(1, 'reset_world', 1).model_dump(), 'result': None}
        assert session.snapshot()['interaction'] == pending
        session._viewer_event(slot, {'kind': 'command', 'command': 'pause'})
        error = json.loads(slot.outgoing.splitlines()[-1])
        assert error['kind'] == 'command_error' and error['message']
        assert supervisor.get_run(plan.run_id)['interaction'] == pending
    result = wait_for(lambda: supervisor.get_run(plan.run_id)['interaction']['result'])
    payload = dict(**pending['command'], status='rejected', paused=False, steps=0, updates=0,
                   error='reset_world requires pause')
    assert result == payload
    with supervisor._lock:
        assert session.snapshot()['interaction']['result'] == payload
        session._viewer_event(slot, {'kind': 'command', 'command': 'pause'})
        assert supervisor.get_run(plan.run_id)['interaction']['command'] == cmd(2).model_dump()
    wait_for(lambda: supervisor.get_run(plan.run_id)['interaction']['result'])
    assert supervisor.get_run(plan.run_id)['failure'] is None


def test_unsolicited_and_duplicate_ack_cannot_replace_result(live):
    supervisor, plan, _ = live
    payload = dict(**cmd().model_dump(), status='applied', paused=True, steps=0, updates=0, error=None)
    with supervisor._lock:
        event = WorkerEvent(event_seq=supervisor._active['record']['event_seq'] + 1,
                           run_id=plan.run_id, kind='command_result', sim_time=0, wall_time=1, payload=payload)
        with pytest.raises(ValueError, match='command'):
            supervisor._event(event.model_dump_json().encode())
    supervisor.submit_command(plan.run_id, cmd())
    result = wait_for(lambda: supervisor.get_run(plan.run_id)['interaction']['result'])
    with supervisor._lock:
        event = event.model_copy(update={'event_seq': supervisor._active['record']['event_seq'] + 1})
        with pytest.raises(ValueError, match='command'):
            supervisor._event(event.model_dump_json().encode())
        assert supervisor.get_run(plan.run_id)['interaction']['result'] == result


def test_exited_worker_cannot_accept_commands(live):
    supervisor, plan, _ = live
    with supervisor._lock:
        process = supervisor._active['process']
        process.kill()
        process.wait(timeout=5)
        with pytest.raises(ValueError, match='live Worker'):
            supervisor.submit_command(plan.run_id, cmd())
        assert supervisor.get_run(plan.run_id)['interaction'] == {'command': None, 'result': None}


def test_concurrent_viewer_requests_allocate_one_pending_sequence(live):
    from concurrent.futures import ThreadPoolExecutor
    supervisor, plan, _ = live

    def request():
        try:
            return supervisor.viewer_command(plan.run_id, 'pause')
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: request(), range(2)))
    accepted = [r for r in results if r is not None]
    assert accepted == [{'command': cmd().model_dump(), 'result': None}]
    result = wait_for(lambda: supervisor.get_run(plan.run_id)['interaction']['result'])
    assert result['command_seq'] == 1 and result['steps'] == 1
