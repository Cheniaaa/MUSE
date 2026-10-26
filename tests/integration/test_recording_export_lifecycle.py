"""CPU导出所有权/发布竞争；fake编码不是软件渲染验收。"""
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

from fastapi.testclient import TestClient
import pytest

from test_supervisor import wait_for

FIXTURES = Path(__file__).parents[1] / 'fixtures'


@pytest.fixture
def source(pipeline, recording_source, recorded_trajectory):
    from muse_server.supervisor import Supervisor
    catalog, _ = pipeline
    plan, _ = recording_source
    plan.timeouts.stop = .3
    path = Path(plan.artifact_directory) / 'resolved_run_plan.json'
    path.write_text(plan.model_dump_json())
    with Supervisor(catalog) as supervisor:
        supervisor._save(dict(run_id=plan.run_id, state='COMPLETED', pid=None,
            artifact_directory=plan.artifact_directory, plan_path=str(path),
            artifacts=[recorded_trajectory.model_dump(mode='json')]))
    return catalog, plan, dict(run_id=plan.run_id, trajectory_artifact_id=recorded_trajectory.id)


def replay_command(identity, mode='paired'):
    return [sys.executable, str(FIXTURES / 'export_replay_child.py'), mode, '--view-id', identity]


@pytest.fixture
def encoder(monkeypatch):
    mode = ['success']
    monkeypatch.setattr('muse_server.replays.recording_command', lambda path: [sys.executable,
        str(FIXTURES / 'export_encoder_child.py'), '--job', str(path), '--mode', mode[0]], raising=False)
    return mode


@pytest.fixture
def rig(source, encoder):
    from muse_sdk.replay_protocol import ReplayRequest
    from muse_server.replays import ReplayManager
    from muse_server.supervisor import Supervisor
    catalog, plan, request = source
    with Supervisor(catalog) as supervisor:
        manager = ReplayManager(catalog, supervisor, process_command=replay_command)
        try:
            identity = manager.create(ReplayRequest(**request))['session_id']
            manager.attach(identity)
            wait_for(lambda: manager.get(identity)['status'] == 'ready', timeout=25)
            wait_for(lambda: manager.get(identity)['camera_ready'])
            yield manager, supervisor, identity, plan
        finally:
            manager.shutdown()


def request():
    from muse_sdk.replay_protocol import RecordingRequest
    return RecordingRequest(world_id=0, episode=0, fps=25., width=640, height=480)


def terminal(manager, identity, export_id=None):
    def check():
        state = manager.get(identity)['export']
        return state if (state and (export_id is None or state['export_id'] == export_id)
                         and state['status'] in ('completed', 'cancelled', 'failed')) else None
    return wait_for(check, timeout=15)


def test_export_freezes_source_before_seek_and_only_completes_after_publication(rig):
    manager, supervisor, identity, plan = rig
    before = manager.get(identity)['revision']
    initial = manager.export(identity, request())
    assert initial['status'] == 'preparing'
    manager.control(identity, dict(kind='playback', command=dict(command_seq=1, kind='seek', sample_index=8)))
    result = terminal(manager, identity)
    assert result['status'] == 'completed' and result['frames_written'] == result['total_frames'] == 3
    assert manager.get(identity)['revision'] > before
    artifact = next(item for item in supervisor.artifacts(plan.run_id) if item['id'] == result['artifact_id'])
    assert artifact['metadata']['world_id'] == artifact['metadata']['episode'] == 0
    assert artifact['metadata']['source_trajectory_id'] == manager.get(identity)['source']['trajectory_artifact_id']
    assert artifact['metadata']['last_sample_index'] == 3
    assert artifact['metadata']['camera']['position'] == [3., -4., 2.]
    assert (Path(plan.artifact_directory) / artifact['path']).is_file()
    manager.cancel_export(identity, initial['export_id'])
    assert (Path(plan.artifact_directory) / artifact['path']).is_file()
    assert supervisor.get_run(plan.run_id)['state'] == 'COMPLETED' and not supervisor.gpu_busy


def test_api_validates_dimensions_range_pairing_busy_and_cancel(source, encoder):
    from muse_server.app import create_app
    catalog, _, payload = source
    with TestClient(create_app(catalog.directory)) as client:
        manager = client.app.state.replays
        options = dict(world_id=0, episode=0, fps=25., width=640, height=480)
        manager.process_command = lambda identity: [sys.executable, '-c', 'import time; time.sleep(60)']
        preparing = client.post('/replays', json=payload).json()['session_id']
        assert client.post(f'/replays/{preparing}/recordings', json=options).status_code == 409
        client.delete(f'/replays/{preparing}')
        manager.process_command = lambda identity: replay_command(identity, 'unpaired')
        identity = client.post('/replays', json=payload).json()['session_id']
        wait_for(lambda: manager.get(identity)['status'] == 'ready', timeout=25)
        endpoint = f'/replays/{identity}/recordings'
        assert client.post(endpoint, json=options).status_code == 409
        assert client.post(endpoint, json=dict(options, width=641)).status_code == 422
        assert client.post(endpoint, json=dict(options, fps=0.)).status_code == 422
        client.delete(f'/replays/{identity}')
        manager.process_command = replay_command
        identity = client.post('/replays', json=payload).json()['session_id']
        endpoint = f'/replays/{identity}/recordings'
        with client.websocket_connect(f'/streams/replays/{identity}'):
            wait_for(lambda: manager.get(identity)['camera_ready'], timeout=25)
            assert client.post(endpoint, json=dict(options, world_id=99)).status_code == 422
            assert client.post(endpoint, json=dict(options, episode=99)).status_code == 422
            encoder[0] = 'hang'
            response = client.post(endpoint, json=options)
            assert response.status_code == 202
            export_id = response.json()['export_id']
            assert client.post(endpoint, json=options).status_code == 409
            assert client.delete(endpoint + '/' + export_id).status_code == 204
            assert terminal(manager, identity)['status'] == 'cancelled'
            assert client.delete(endpoint + '/missing').status_code == 404
            assert client.get('/catalog/schema').json()['recording_request']['properties']['fps']
            encoder[0] = 'success'
            response = client.post(endpoint, json=options)
            assert response.status_code == 202 and response.json()['export_id'] != export_id
            assert terminal(manager, identity, response.json()['export_id'])['status'] == 'completed'


@pytest.mark.parametrize('mode', ['error', 'missing'])
def test_encoder_failure_cleans_unregistered_files_and_does_not_change_run(rig, encoder, mode):
    manager, supervisor, identity, plan = rig
    encoder[0] = mode
    initial = manager.export(identity, request())
    result = terminal(manager, identity)
    assert result['status'] == 'failed' and result['error']
    assert not list(Path(plan.artifact_directory).glob('recording-*'))
    assert not (Path(plan.artifact_directory) / 'recordings' / (initial['export_id'] + '.mp4')).exists()
    assert not any(item['kind'] == 'recording' for item in supervisor.artifacts(plan.run_id))
    assert supervisor.get_run(plan.run_id)['state'] == 'COMPLETED'


def test_cancel_waits_for_term_ignoring_descendant_group(rig, encoder):
    from muse_server.supervisor import process_identity
    manager, supervisor, identity, plan = rig
    encoder[0] = 'hang'
    initial = manager.export(identity, request())
    job = manager._sessions[identity].export_job
    pid_file = wait_for(lambda: (p if job.staging and (p := Path(job.staging.name) / 'descendant.pid').exists() else None))
    pid = int(pid_file.read_text())
    manager.cancel_export(identity, initial['export_id'])
    assert process_identity(pid) is None and job.process.poll() is not None
    assert not job.owner.is_alive() and not Path(job.staging.name).exists()
    assert terminal(manager, identity)['status'] == 'cancelled'
    assert not any(item['kind'] == 'recording' for item in supervisor.artifacts(plan.run_id))


def test_parent_exit_cannot_publish_until_descendant_group_is_closed(rig, encoder, monkeypatch):
    from muse_server.supervisor import process_identity
    manager, supervisor, identity, _ = rig
    encoder[0] = 'orphan'
    original = supervisor.publish_recorded_artifact
    checked = threading.Event()

    def publish(artifact):
        job = manager._sessions[identity].export_job
        pid = int(Path(job.staging.name, 'descendant.pid').read_text())
        assert process_identity(pid) is None
        checked.set()
        return original(artifact)

    monkeypatch.setattr(supervisor, 'publish_recorded_artifact', publish)
    manager.export(identity, request())
    assert terminal(manager, identity)['status'] == 'completed'
    assert checked.is_set()


def test_cancel_pending_hash_waits_for_reader_before_removing_unregistered_file(rig, monkeypatch):
    import muse_server.supervisor as module
    manager, supervisor, identity, plan = rig
    entered, release = threading.Event(), threading.Event()
    original = module.verify_recorded_file

    def blocked(artifact, run_id, directory):
        entered.set()
        assert release.wait(10)
        return original(artifact, run_id, directory)

    monkeypatch.setattr(module, 'verify_recorded_file', blocked)
    initial = manager.export(identity, request())
    assert entered.wait(10)
    job = manager._sessions[identity].export_job
    path = Path(plan.artifact_directory) / 'recordings' / (initial['export_id'] + '.mp4')
    finished = threading.Event()
    thread = threading.Thread(target=lambda: (manager.cancel_export(identity, initial['export_id']), finished.set()))
    thread.start()
    try:
        wait_for(lambda: job.cancelled.is_set())
        assert path.exists() and not finished.is_set()
        with pytest.raises(RuntimeError, match='busy'):
            manager.export(identity, request())
    finally:
        release.set()
        thread.join(10)
    assert finished.is_set() and not path.exists()
    assert terminal(manager, identity)['status'] == 'cancelled'
    assert not any(item['kind'] == 'recording' for item in supervisor.artifacts(plan.run_id))


def test_registration_wins_cancellation_keeps_artifact(rig, monkeypatch):
    manager, supervisor, identity, plan = rig
    entered, release = threading.Event(), threading.Event()
    original = supervisor.publish_recorded_artifact

    def committed(artifact):
        publication = original(artifact)
        publication.result(timeout=10)
        entered.set()
        assert release.wait(10)
        return publication

    monkeypatch.setattr(supervisor, 'publish_recorded_artifact', committed)
    initial = manager.export(identity, request())
    assert entered.wait(10)
    # publish返回前持manager锁是允许的短临界区，但不能在这个门控下等取消获取同一锁。
    release.set()
    manager.cancel_export(identity, initial['export_id'])
    result = terminal(manager, identity)
    assert result['status'] == 'completed'
    artifact = next(item for item in supervisor.artifacts(plan.run_id) if item['id'] == result['artifact_id'])
    assert (Path(plan.artifact_directory) / artifact['path']).exists()


@pytest.mark.parametrize('close_kind', ['close', 'shutdown'])
def test_session_close_and_shutdown_cancel_export_before_removing_history(rig, encoder, close_kind):
    manager, _, identity, _ = rig
    encoder[0] = 'artifact_hang'
    manager.export(identity, request())
    session = manager._sessions[identity]
    job = session.export_job
    wait_for(lambda: job.process is not None and Path(job.staging.name, 'encoder.pid').exists())
    if close_kind == 'close':
        manager.close(identity)
    else:
        manager.shutdown()
    assert job.process.poll() is not None and session.process.poll() is not None
    assert not job.owner.is_alive() and not session.reader.is_alive()
    assert not Path(job.staging.name).exists() and not Path(session.temporary.name).exists()


def test_ws_disconnect_cancels_active_export_and_joins_owned_threads(source, encoder):
    from muse_server.app import create_app
    from muse_server.supervisor import process_identity
    catalog, _, payload = source
    encoder[0] = 'hang'
    with TestClient(create_app(catalog.directory)) as client:
        manager = client.app.state.replays
        manager.process_command = replay_command
        identity = client.post('/replays', json=payload).json()['session_id']
        with client.websocket_connect(f'/streams/replays/{identity}'):
            wait_for(lambda: manager.get(identity)['camera_ready'], timeout=25)
            result = client.post(f'/replays/{identity}/recordings', json=dict(
                world_id=0, episode=0, fps=25., width=640, height=480))
            assert result.status_code == 202
            session = manager._sessions[identity]
            job = session.export_job
            path = wait_for(lambda: (p if job.staging and (p := Path(job.staging.name) / 'descendant.pid').exists() else None))
            pid = int(path.read_text())
        wait_for(lambda: identity not in manager._sessions)
        assert process_identity(pid) is None
        assert not job.owner.is_alive() and not session.reader.is_alive()
        assert not Path(job.staging.name).exists() and not Path(session.temporary.name).exists()


def test_prepare_pipe_failure_does_not_leave_a_job_without_an_owner(rig, monkeypatch):
    manager, _, identity, _ = rig
    session = manager._sessions[identity]
    original = manager._send

    def failed(target, message):
        if message['kind'] == 'prepare_export':
            os.killpg(target.process.pid, signal.SIGKILL)
            target.process.wait(timeout=5)
        return original(target, message)

    monkeypatch.setattr(manager, '_send', failed)
    with pytest.raises(BrokenPipeError):
        manager.export(identity, request())
    wait_for(lambda: manager.get(identity)['status'] == 'error')
    assert not Path(session.temporary.name).exists()


def test_history_child_failure_closes_active_encoder_without_changing_run(rig, encoder):
    manager, supervisor, identity, plan = rig
    encoder[0] = 'hang'
    manager.export(identity, request())
    session = manager._sessions[identity]
    job = session.export_job
    wait_for(lambda: job.process is not None and Path(job.staging.name, 'encoder.pid').exists())
    os.killpg(session.process.pid, signal.SIGKILL)
    wait_for(lambda: manager.get(identity)['status'] == 'error')
    assert job.process.poll() is not None and not job.owner.is_alive()
    assert not Path(session.temporary.name).exists() and not Path(job.staging.name).exists()
    assert supervisor.get_run(plan.run_id)['state'] == 'COMPLETED'


def test_completed_export_does_not_signal_a_reused_process_group_identity(rig):
    manager, _, identity, _ = rig
    initial = manager.export(identity, request())
    assert terminal(manager, identity)['status'] == 'completed'
    job = manager._sessions[identity].export_job
    assert job.done.wait(5)
    job.owner.join(5)
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        # 模拟已wait的旧PID后来成为另一个group leader；不能再次按旧句柄发信号。
        job.process.pid = unrelated.pid
        manager.cancel_export(identity, initial['export_id'])
        assert unrelated.poll() is None
    finally:
        if unrelated.poll() is None:
            os.killpg(unrelated.pid, signal.SIGKILL)
        unrelated.wait(timeout=5)


def test_publication_failure_cleans_file_and_keeps_source_run_terminal(rig, monkeypatch):
    import muse_server.supervisor as module
    manager, supervisor, identity, plan = rig

    def failed(*args):
        raise ValueError('recording verification root cause')

    monkeypatch.setattr(module, 'verify_recorded_file', failed)
    initial = manager.export(identity, request())
    result = terminal(manager, identity)
    assert result['status'] == 'failed' and result['error'] == 'recording verification root cause'
    assert not (Path(plan.artifact_directory) / 'recordings' / (initial['export_id'] + '.mp4')).exists()
    assert supervisor.get_run(plan.run_id)['state'] == 'COMPLETED'


def test_camera_disconnected_before_ordered_snapshot_fails_only_this_export(source, encoder):
    from muse_sdk.replay_protocol import ReplayRequest
    from muse_server.replays import ReplayManager
    from muse_server.supervisor import Supervisor
    catalog, plan, payload = source
    with Supervisor(catalog) as supervisor:
        manager = ReplayManager(catalog, supervisor,
            process_command=lambda identity: replay_command(identity, 'snapshot_error'))
        try:
            identity = manager.create(ReplayRequest(**payload))['session_id']
            manager.attach(identity)
            wait_for(lambda: manager.get(identity)['camera_ready'], timeout=25)
            manager.export(identity, request())
            result = terminal(manager, identity)
            assert result['status'] == 'failed' and 'camera disconnected' in result['error']
            assert manager.get(identity)['status'] == 'ready'
            assert supervisor.get_run(plan.run_id)['state'] == 'COMPLETED'
        finally:
            manager.shutdown()


def test_background_camera_event_is_merged_by_history_revision_owner(source, tmp_path):
    from muse_sdk.replay_protocol import ReplayRequest
    from muse_server.replays import ReplayManager
    from muse_server.supervisor import Supervisor
    catalog, _, payload = source
    gate = tmp_path / 'initialize-camera'
    with Supervisor(catalog) as supervisor:
        manager = ReplayManager(catalog, supervisor,
            process_command=lambda identity: replay_command(identity, 'delayed:' + str(gate)))
        try:
            identity = manager.create(ReplayRequest(**payload))['session_id']
            manager.attach(identity)
            before = wait_for(lambda: (s if (s := manager.get(identity))['status'] == 'ready' else None), timeout=25)
            assert not before['camera_ready']
            gate.touch()
            after = wait_for(lambda: (s if (s := manager.get(identity))['camera_ready'] else None))
            assert after['revision'] > before['revision']
            assert after['cursor'] == before['cursor']
        finally:
            manager.shutdown()


def test_close_waits_for_publication_read_before_removing_history(rig, monkeypatch):
    import muse_server.supervisor as module
    manager, supervisor, identity, plan = rig
    entered, release = threading.Event(), threading.Event()
    original = module.verify_recorded_file

    def blocked(artifact, run_id, directory):
        entered.set()
        assert release.wait(10)
        return original(artifact, run_id, directory)

    monkeypatch.setattr(module, 'verify_recorded_file', blocked)
    initial = manager.export(identity, request())
    assert entered.wait(10)
    session = manager._sessions[identity]
    job = session.export_job
    final = Path(plan.artifact_directory) / 'recordings' / (initial['export_id'] + '.mp4')
    finished = threading.Event()
    thread = threading.Thread(target=lambda: (manager.close(identity), finished.set()))
    thread.start()
    try:
        wait_for(lambda: job.cancelled.is_set())
        assert not finished.is_set() and final.exists() and Path(session.temporary.name).exists()
    finally:
        release.set()
        thread.join(10)
    assert finished.is_set() and not final.exists()
    assert not Path(session.temporary.name).exists() and not Path(job.staging.name).exists()
    assert supervisor.get_run(plan.run_id)['state'] == 'COMPLETED'
