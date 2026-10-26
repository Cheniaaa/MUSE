"""独立CPU历史进程、唯一控制连接及持久轨迹引用。"""
from pathlib import Path
import asyncio
import sys
import subprocess
import threading
import time

from fastapi.testclient import TestClient
from fastapi import WebSocketDisconnect
import pytest

from test_supervisor import wait_for


@pytest.fixture
def source(pipeline, recording_source, recorded_trajectory):
    from muse_server.supervisor import Supervisor
    catalog, _ = pipeline
    plan, _ = recording_source
    plan_path = Path(plan.artifact_directory) / 'resolved_run_plan.json'
    plan_path.write_text(plan.model_dump_json())
    record = dict(run_id=plan.run_id, state='COMPLETED', pid=None,
        artifact_directory=plan.artifact_directory, plan_path=str(plan_path),
        artifacts=[recorded_trajectory.model_dump(mode='json')])
    # 仅登记CPU fixture，不调用submit，也不创建GPU Run。
    with Supervisor(catalog) as supervisor:
        supervisor._save(record)
    return catalog, plan, dict(run_id=plan.run_id, trajectory_artifact_id=recorded_trajectory.id)


def ready(manager, identity):
    def check():
        state = manager.get(identity)
        assert state['status'] != 'error', state['error']
        return state if state['status'] == 'ready' else None
    return wait_for(check, timeout=25)


def applied(manager, identity, sequence):
    return wait_for(lambda: (s if (s := manager.get(identity))['last_command_seq'] == sequence else None))


def test_two_real_cpu_sessions_are_independent_and_reopen_after_restart(source):
    from muse_sdk.replay_protocol import ReplayRequest, ReplayState
    from muse_server.replays import ReplayManager
    from muse_server.supervisor import Supervisor
    catalog, _, request = source
    supervisor = Supervisor(catalog)
    try:
        for restart in range(2):
            manager = ReplayManager(catalog, supervisor)
            try:
                one = manager.create(ReplayRequest(**request))
                assert one['status'] == 'preparing' and one['cursor'] is None
                manager.attach(one['session_id'])
                initial = ready(manager, one['session_id'])
                ReplayState.model_validate(initial)
                assert initial['source'] == request
                assert all('body_id' not in obj for obj in initial['objects'])
                if restart == 0:
                    two = manager.create(ReplayRequest(**request))
                    manager.attach(two['session_id'])
                    ready(manager, two['session_id'])
                    manager.control(one['session_id'], {'kind': 'playback', 'command': {
                        'command_seq': 1, 'kind': 'seek', 'sample_index': 8}})
                    before = applied(manager, one['session_id'], 1)
                    manager.control(one['session_id'], {'kind': 'playback', 'command': {
                        'command_seq': 2, 'kind': 'seek', 'sample_index': 2}})
                    after = applied(manager, one['session_id'], 2)
                    assert after['revision'] > before['revision']
                    assert after['cursor']['sample_index'] == after['diagnostics']['sample_index'] == 2
                    assert manager.get(two['session_id'])['cursor']['sample_index'] == 0
                    assert not supervisor.gpu_busy
                    session = manager._sessions[one['session_id']]
                    temporary = Path(session.temporary.name)
                    manager.close(one['session_id'])
                    assert session.process.poll() is not None and not temporary.exists()
                    assert not session.reader.is_alive()
            finally:
                manager.shutdown()
    finally:
        supervisor._selector.close()


def test_api_routes_unique_ws_owner_errors_disconnect_and_restart(source):
    catalog, _, request = source
    from muse_server.app import create_app
    for _ in range(2):
        with TestClient(create_app(catalog.directory)) as client:
            schema = client.get('/catalog/schema').json()
            assert schema['replay_request']['properties']['trajectory_artifact_id']
            response = client.post('/replays', json=request)
            assert response.status_code == 201
            initial = response.json()
            identity = initial['session_id']
            manager = client.app.state.replays
            with client.websocket_connect(f'/streams/replays/{identity}') as ws:
                state = ws.receive_json()
                assert state['kind'] == 'state'
                with pytest.raises(WebSocketDisconnect) as rejected:
                    with client.websocket_connect(f'/streams/replays/{identity}'):
                        pass
                assert rejected.value.code == 1008
                ws.send_json({'kind': 'playback', 'command': {'command_seq': 7, 'kind': 'seek', 'sample_index': -1}})
                while (event := ws.receive_json())['kind'] != 'command_result':
                    pass
                assert event['command_seq'] == 7 and event['status'] == 'error'
                ws.send_json([])
                while (event := ws.receive_json())['kind'] != 'view':
                    pass
                assert event['event']['kind'] == 'error'
            wait_for(lambda: identity not in manager._sessions)
            assert client.get(f'/replays/{identity}').status_code == 404
            assert client.delete('/replays/missing').status_code == 404
            assert not client.get('/health').json()['gpu_busy']


def test_rejects_unregistered_wrong_run_and_wrong_kind(source):
    from muse_server.app import create_app
    catalog, _, request = source
    with TestClient(create_app(catalog.directory)) as client:
        for payload in [dict(request, trajectory_artifact_id='unregistered'),
                        dict(request, run_id='other')]:
            assert client.post('/replays', json=payload).status_code == 404
        supervisor = client.app.state.supervisor
        record = supervisor.get_run(request['run_id'])
        record['artifacts'][0]['run'] = 'other'
        supervisor._save(record)
        assert client.post('/replays', json=request).status_code == 422
        record['artifacts'][0]['run'] = request['run_id']
        record['artifacts'][0]['kind'] = 'checkpoint'
        supervisor._save(record)
        assert client.post('/replays', json=request).status_code == 422


def test_preparation_does_not_wait_for_child_and_unattached_session_expires(source):
    from muse_sdk.replay_protocol import ReplayRequest
    from muse_server.replays import ReplayManager
    from muse_server.supervisor import Supervisor
    catalog, _, request = source
    supervisor = Supervisor(catalog)
    manager = ReplayManager(catalog, supervisor, attach_timeout=.15,
        process_command=lambda identity: [sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        started = time.monotonic()
        state = manager.create(ReplayRequest(**request))
        assert time.monotonic() - started < 1. and state['status'] == 'preparing'
        session = manager._sessions[state['session_id']]
        temporary = Path(session.temporary.name)
        wait_for(lambda: state['session_id'] not in manager._sessions)
        assert session.process.poll() is not None and not temporary.exists()
    finally:
        manager.shutdown()
        supervisor._selector.close()


def test_child_error_preserves_root_cause_and_cleans_resources(source):
    from muse_sdk.replay_protocol import ReplayRequest
    from muse_server.replays import ReplayManager
    from muse_server.supervisor import Supervisor
    catalog, plan, request = source
    supervisor = Supervisor(catalog)
    manager = ReplayManager(catalog, supervisor)
    archive = Path(plan.artifact_directory) / supervisor.artifacts(plan.run_id)[0]['path']
    archive.write_bytes(b'corrupt after publication')
    try:
        state = manager.create(ReplayRequest(**request))
        session = manager._sessions[state['session_id']]
        manager.attach(state['session_id'])
        error = wait_for(lambda: (s if (s := manager.get(state['session_id']))['status'] == 'error' else None))
        assert 'size' in error['error'].lower() or 'hash' in error['error'].lower()
        wait_for(lambda: not Path(session.temporary.name).exists())
        assert session.process.poll() is not None
        assert manager.get(state['session_id'])['error'] == error['error']
        assert supervisor.get_run(plan.run_id)['state'] == 'COMPLETED'
    finally:
        manager.shutdown()
        supervisor._selector.close()


def test_server_shutdown_reaps_replay_and_removes_temporary_directory(source):
    from muse_server.app import create_app
    catalog, _, request = source
    with TestClient(create_app(catalog.directory)) as client:
        state = client.post('/replays', json=request).json()
        session = client.app.state.replays._sessions[state['session_id']]
        temporary = Path(session.temporary.name)
    assert session.process.poll() is not None
    assert not temporary.exists() and not session.reader.is_alive()


def test_error_cleanup_and_explicit_close_cannot_join_each_other(source, monkeypatch):
    from muse_sdk.replay_protocol import ReplayRequest
    from muse_server.replays import ReplayManager
    from muse_server.supervisor import Supervisor
    catalog, plan, request = source
    plan.timeouts.stop = .2
    Path(plan.artifact_directory, 'resolved_run_plan.json').write_text(plan.model_dump_json())
    supervisor = Supervisor(catalog)
    script = "import sys,json,time; input(); print(json.dumps({'kind':'error','message':'original error'}),flush=True); time.sleep(60)"
    manager = ReplayManager(catalog, supervisor,
        process_command=lambda identity: [sys.executable, '-c', script])
    about_to_dispose, release_reader = threading.Event(), threading.Event()
    original = manager._dispose

    def gated(session, **options):
        if threading.current_thread() is session.reader:
            about_to_dispose.set()
            assert release_reader.wait(5)
        return original(session, **options)

    monkeypatch.setattr(manager, '_dispose', gated)
    state = manager.create(ReplayRequest(**request))
    session = manager._sessions[state['session_id']]
    errors = []

    def close():
        try:
            manager.close(state['session_id'])
        except Exception as exc:
            errors.append(exc)

    closer = threading.Thread(target=close)
    try:
        assert about_to_dispose.wait(5)
        closer.start()
        wait_for(lambda: session.stopping)
        release_reader.set()
        closer.join(5)
        assert not closer.is_alive() and not errors
        assert not session.reader.is_alive() and not Path(session.temporary.name).exists()
    finally:
        release_reader.set()
        if closer.ident is not None:
            closer.join(5)
        manager.shutdown()
        supervisor._selector.close()


def test_control_receipt_overflow_is_bounded_and_fails_explicitly(source):
    from muse_sdk.replay_protocol import ReplayRequest
    from muse_server.replays import ReplayManager
    from muse_server.supervisor import Supervisor
    catalog, _, request = source
    supervisor = Supervisor(catalog)
    script = """import json,time
input()
input()
for sequence in range(300):
    print(json.dumps(dict(kind='command_result',command_seq=sequence,status='applied',error=None,revision=1)),flush=True)
time.sleep(60)
"""
    manager = ReplayManager(catalog, supervisor,
        process_command=lambda identity: [sys.executable, '-c', script])
    try:
        state = manager.create(ReplayRequest(**request))
        session = manager._sessions[state['session_id']]
        manager.attach(state['session_id'])
        manager._send(session, {'kind': 'release'})
        error = wait_for(lambda: (s if (s := manager.get(state['session_id']))['status'] == 'error' else None))
        assert 'overflow' in error['error'] and len(manager.events(state['session_id'])) <= 256
        assert session.process.poll() is not None and not Path(session.temporary.name).exists()
    finally:
        manager.shutdown()
        supervisor._selector.close()


def test_close_waits_for_term_ignoring_descendant_process_group(source):
    from muse_sdk.replay_protocol import ReplayRequest
    from muse_server.replays import ReplayManager
    from muse_server.supervisor import Supervisor, process_identity
    catalog, plan, request = source
    plan.timeouts.stop = .15
    Path(plan.artifact_directory, 'resolved_run_plan.json').write_text(plan.model_dump_json())
    supervisor = Supervisor(catalog)
    script = """import json,subprocess,sys,time
from pathlib import Path
message=json.loads(input())
subprocess.Popen([sys.executable,'-c',
    'import os,signal,sys,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)',
    str(Path(message['directory'])/'descendant.pid')])
time.sleep(60)
"""
    manager = ReplayManager(catalog, supervisor,
        process_command=lambda identity: [sys.executable, '-c', script])
    try:
        state = manager.create(ReplayRequest(**request))
        session = manager._sessions[state['session_id']]
        manager.attach(state['session_id'])
        path = Path(session.temporary.name) / 'descendant.pid'
        pid = int(wait_for(lambda: path.read_text() if path.exists() else None))
        assert process_identity(pid) is not None
        manager.close(state['session_id'])
        assert session.process.poll() is not None and process_identity(pid) is None
        assert not Path(session.temporary.name).exists()
    finally:
        manager.shutdown()
        supervisor._selector.close()


def test_server_import_does_not_load_runtime_or_graphics():
    script = """import sys
import muse_server.app
for root in ('muse_runtime','torch','mujoco','viser','mjviser'):
    assert not any(name == root or name.startswith(root+'.') for name in sys.modules), root
"""
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)


def test_projected_scene_and_diagnostics_follow_backward_seek(source):
    from muse_sdk.replay_protocol import ReplayRequest
    from muse_server.replays import ReplayManager
    from muse_server.supervisor import Supervisor
    catalog, _, request = source
    supervisor = Supervisor(catalog)
    fixture = Path(__file__).parents[1] / 'fixtures' / 'replay_projection_child.py'
    manager = ReplayManager(catalog, supervisor,
        process_command=lambda identity: [sys.executable, str(fixture), '--view-id', identity])
    try:
        identity = manager.create(ReplayRequest(**request))['session_id']
        manager.attach(identity)
        ready(manager, identity)
        for sequence, sample, distance in [(1, 8, 15.9), (2, 2, 7.9)]:
            manager.events(identity)
            manager.control(identity, dict(kind='playback', command=dict(
                command_seq=sequence, kind='seek', sample_index=sample)))
            state = applied(manager, identity, sequence)
            projection = next(event['event']['hit'] for event in manager.events(identity)
                              if event['kind'] == 'view' and event['event']['kind'] == 'projection')
            assert projection['object_id'] == 'scout_group:0'
            assert projection['distance'] == pytest.approx(distance)
            assert projection['frame']['sample_index'] == state['diagnostics']['sample_index'] == sample
    finally:
        manager.shutdown()
        supervisor._selector.close()


def test_child_exiting_before_prepare_write_is_reaped_and_reported(source, monkeypatch):
    from muse_sdk.replay_protocol import ReplayRequest
    from muse_server.replays import ReplayManager
    from muse_server.supervisor import Supervisor
    catalog, _, request = source
    supervisor = Supervisor(catalog)
    popen = subprocess.Popen

    def already_exited(*args, **kwargs):
        process = popen(*args, **kwargs)
        process.wait(timeout=5)
        return process

    manager = ReplayManager(catalog, supervisor,
        process_command=lambda identity: [sys.executable, '-c', 'raise SystemExit(17)'])
    monkeypatch.setattr('muse_server.replays.subprocess.Popen', already_exited)
    try:
        state = manager.create(ReplayRequest(**request))
        session = manager._sessions[state['session_id']]
        error = wait_for(lambda: (s if (s := manager.get(state['session_id']))['status'] == 'error' else None))
        assert error['error'] and session.process.returncode == 17
        assert not Path(session.temporary.name).exists()
    finally:
        manager.shutdown()
        supervisor._selector.close()


def test_close_joins_reader_even_after_error_cleanup_disposed_resources(source, monkeypatch):
    from muse_sdk.replay_protocol import ReplayRequest
    from muse_server.replays import ReplayManager
    from muse_server.supervisor import Supervisor
    catalog, _, request = source
    supervisor = Supervisor(catalog)
    script = "import json; input(); print(json.dumps({'kind':'error','message':'root cause'}),flush=True)"
    manager = ReplayManager(catalog, supervisor,
        process_command=lambda identity: [sys.executable, '-c', script])
    disposed, release_reader = threading.Event(), threading.Event()
    original = manager._dispose

    def gated(session, **options):
        result = original(session, **options)
        if threading.current_thread() is session.reader:
            disposed.set()
            assert release_reader.wait(5)
        return result

    monkeypatch.setattr(manager, '_dispose', gated)
    state = manager.create(ReplayRequest(**request))
    session = manager._sessions[state['session_id']]
    join = session.reader.join

    def release_and_join(timeout=None):
        release_reader.set()
        join(timeout)

    monkeypatch.setattr(session.reader, 'join', release_and_join)
    try:
        assert disposed.wait(5)
        manager.close(state['session_id'])
        assert not session.reader.is_alive()
    finally:
        release_reader.set()
        join(5)
        manager.shutdown()
        supervisor._selector.close()


def test_explicit_delete_closes_the_attached_websocket(source):
    from muse_server.app import create_app
    catalog, _, request = source
    with TestClient(create_app(catalog.directory)) as client:
        identity = client.post('/replays', json=request).json()['session_id']
        manager = client.app.state.replays
        endpoint = next(route.endpoint for route in client.app.routes
                        if getattr(route, 'path', None) == '/streams/replays/{session_id}')

        class Socket:
            codes = []

            async def accept(self):
                pass

            async def send_json(self, message):
                pass

            async def receive_json(self):
                # 模拟控制WS仍在线时另一个HTTP任务完成DELETE。
                await asyncio.to_thread(manager.close, identity)
                return {'kind': 'playback', 'command': {'command_seq': 1, 'kind': 'pause'}}

            async def close(self, code):
                self.codes.append(code)

        socket = Socket()
        client.portal.call(endpoint, socket, identity)
        assert socket.codes == [1000]


def test_child_error_racing_with_input_still_reaches_websocket_owner(source):
    from muse_server.app import create_app
    catalog, _, request = source
    with TestClient(create_app(catalog.directory)) as client:
        identity = client.post('/replays', json=request).json()['session_id']
        manager = client.app.state.replays
        session = manager._sessions[identity]
        endpoint = next(route.endpoint for route in client.app.routes
                        if getattr(route, 'path', None) == '/streams/replays/{session_id}')

        class Socket:
            def __init__(self):
                self.codes, self.messages = [], []

            async def accept(self):
                pass

            async def send_json(self, message):
                self.messages.append(message)

            async def receive_json(self):
                await asyncio.to_thread(manager._dispose, session, remove=False, failure='child root cause')
                return {'kind': 'playback', 'command': {'command_seq': 1, 'kind': 'pause'}}

            async def close(self, code):
                self.codes.append(code)

        socket = Socket()
        client.portal.call(endpoint, socket, identity)
        assert socket.codes == [1011]
        assert any(message.get('status') == 'error' and message['error'] == 'child root cause'
                   for message in socket.messages)
