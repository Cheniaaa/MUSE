"""CPU 文件发布、取消和 Worker 终态顺序，不创建 MuseEnv/GPU Run。"""
from concurrent.futures import CancelledError, ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading
from uuid import uuid4

import pytest

from muse_sdk.recording_protocol import TrajectoryArtifact
from muse_server.supervisor import Supervisor, TERMINAL_STATES
from test_supervisor import wait_for


CHILD = r'''
import json, signal, sys, time
from pathlib import Path
from muse_runtime.protocol import WorkerProtocol
from muse_sdk.experiment import RunPlan
root = Path(sys.argv[1])
plan = RunPlan.model_validate_json((root / 'resolved_run_plan.json').read_bytes())
p = WorkerProtocol(plan.run_id)
behavior = sys.argv[2]
p.emit('preparing')
if behavior in ('prepare_failure', 'capture_failure', 'seal_failure', 'rename_failure'):
    message = behavior
    if behavior == 'rename_failure':
        import numpy as np
        import muse_sdk.trajectory as trajectory
        from muse_sdk.run_protocol import PreviewModel
        writer = trajectory.TrajectoryWriter(plan, PreviewModel.model_validate_json((root / 'model.json').read_bytes()))
        for world in plan.recording.world_ids:
            writer.append(world, (0, 0, 0, 0, 0, 0, 0),
                [np.zeros(f.shape, dtype=f.dtype) for f in writer.fields])
        def fail_rename(*args): raise OSError('rename failed')
        trajectory.os.replace = fail_rename
        try:
            writer.seal('completed', 0)
        except OSError as exc:
            message = str(exc)
        finally:
            writer.close()
    p.emit('failure', {'message': message})
    p.emit('terminal', {'state': 'FAILED'})
    sys.exit(1)
p.emit('running')
if behavior == 'forced':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        p.emit('heartbeat')
        time.sleep(.02)
artifact = json.loads((root / 'artifact.json').read_bytes())
state = dict(status='publishing', config=plan.recording.model_dump(), summary=artifact['summary'])
if behavior == 'spoof_published': state['status'] = 'published'
if behavior != 'orphan':
    p.emit('recording_state', state)
    if behavior == 'spoof_published':
        p.emit('terminal', {'state': 'COMPLETED'})
        sys.exit(0)
    if behavior == 'hash': artifact['sha256'] = '0' * 64
    if behavior == 'truncated':
        path = root / artifact['path']
        path.write_bytes(path.read_bytes()[:-20])
    p.emit('trajectory', artifact)
    if behavior == 'duplicate': p.emit('trajectory', artifact)
if behavior == 'cooperative':
    steps = 0
    while p.read_control() != 'cancel':
        if (root / 'fail-worker').exists():
            p.emit('failure', {'message': 'later Worker failure'})
            p.emit('terminal', {'state': 'FAILED'})
            sys.exit(1)
        steps += 1
        p.emit('heartbeat')
        p.emit('metric', {'steps': steps})
        time.sleep(.02)
    p.emit('terminal', {'state': 'CANCELLED', 'steps': steps, 'updates': 0})
else:
    p.emit('terminal', {'state': 'COMPLETED', 'steps': 14, 'updates': 0})
'''


@pytest.fixture
def publication_source(pipeline, recording_source, tmp_path):
    from muse_server.catalog import Catalog
    from muse_sdk.run_protocol import RunTimeouts
    catalog = Catalog(tmp_path / 'publication-catalog', pipeline[0].builtin_path)
    plan, model = recording_source
    plan.timeouts = RunTimeouts(heartbeat_interval=.02, heartbeat=3., cancel=.1, stop=.1)
    root = Path(plan.artifact_directory)
    (root / 'resolved_run_plan.json').write_text(plan.model_dump_json())
    (root / 'model.json').write_text(model.model_dump_json())
    return catalog, plan


def worker(behavior):
    return lambda plan: [sys.executable, '-c', CHILD, plan.artifact_directory, behavior]


def store_event(plan, artifact):
    Path(plan.artifact_directory, 'artifact.json').write_text(artifact.model_dump_json())


def finished(supervisor, plan):
    return wait_for(lambda: r if (r := supervisor.get_run(plan.run_id))['state'] in TERMINAL_STATES else None)


def video_artifact(plan, trajectory):
    from muse_sdk.recording_protocol import RecordingArtifact, RecordingMetadata, ReplayCamera
    token = uuid4().hex
    relative = f'recordings/{token}.mp4'
    path = Path(plan.artifact_directory, relative)
    path.parent.mkdir(exist_ok=True)
    content = b'CPU publication fixture; decoding belongs to export tests'
    path.write_bytes(content)
    return RecordingArtifact(id=f'{plan.run_id}:recording:{token}', run=plan.run_id,
        path=relative, size=len(content), sha256=hashlib.sha256(content).hexdigest(),
        metadata=RecordingMetadata(source_trajectory_id=trajectory.id, world_id=0, episode=0,
            first_sample_index=0, last_sample_index=3, complete_episode=True,
            camera=ReplayCamera(position=(1., 2., 3.), wxyz=(1., 0., 0., 0.), fov=1., near=.1, far=100.),
            fps=25., width=640, height=480, frame_count=8))


def block_verification(monkeypatch):
    import muse_server.supervisor as module
    entered, release = threading.Event(), threading.Event()
    verify = module.verify_recorded_file

    def blocked(*args):
        entered.set()
        assert release.wait(10), 'Test did not release file verification'
        return verify(*args)

    monkeypatch.setattr(module, 'verify_recorded_file', blocked)
    return entered, release


def test_queued_recording_request_is_frozen_and_cancel_does_not_leave_capturing(publication_source):
    catalog, plan = publication_source
    with Supervisor(catalog, worker_command=worker('forced')) as supervisor:
        with supervisor._lock:
            submitted = supervisor.submit(plan)
            assert submitted['recording']['config'] == plan.recording.model_dump(mode='json')
            assert submitted['recording']['status'] == 'capturing'
            cancelled = supervisor.cancel(plan.run_id)
        assert cancelled['state'] == 'CANCELLED' and cancelled['pid'] is None
        assert cancelled['recording']['status'] == 'failed'


def test_pending_verification_does_not_block_control_or_lose_artifact(publication_source, recorded_trajectory, monkeypatch):
    catalog, plan = publication_source
    store_event(plan, recorded_trajectory)
    entered, release = block_verification(monkeypatch)
    supervisor = Supervisor(catalog, worker_command=worker('cooperative')).start()
    try:
        supervisor.submit(plan)
        assert entered.wait(5)
        before = wait_for(lambda: supervisor.get_run(plan.run_id)['heartbeat'])
        wait_for(lambda: (supervisor.get_run(plan.run_id)['heartbeat'] or 0) > before)
        assert supervisor.cancel(plan.run_id)['cancel_requested'] is True
        wait_for(lambda: supervisor.get_run(plan.run_id)['worker_terminal'])
        assert supervisor.get_run(plan.run_id)['state'] not in TERMINAL_STATES
        release.set()
        record = finished(supervisor, plan)
        assert record['state'] == 'CANCELLED'
        assert [a['id'] for a in record['artifacts']].count(recorded_trajectory.id) == 1
        assert record['recording']['status'] == 'published'
        assert record['recording']['artifact_id'] == recorded_trajectory.id
        events = [json.loads(line) for line in Path(plan.artifact_directory, 'events.jsonl').read_text().splitlines()]
        assert [e['event_seq'] for e in events] == list(range(1, len(events) + 1))
        assert record['event_seq'] == events[-1]['event_seq'] + 1
    finally:
        release.set()
        supervisor.close()


def test_worker_exit_waits_for_received_trajectory_without_heartbeat_timeout(publication_source, recorded_trajectory, monkeypatch):
    catalog, plan = publication_source
    store_event(plan, recorded_trajectory)
    entered, release = block_verification(monkeypatch)
    supervisor = Supervisor(catalog, worker_command=worker('completed')).start()
    try:
        supervisor.submit(plan)
        assert entered.wait(5)
        wait_for(lambda: supervisor._active['process'].poll() is not None)
        wait_for(lambda: supervisor._active.get('finishing'))
        # 已退出进程不再有 heartbeat；验证可以长于 heartbeat 限期。
        threading.Event().wait(plan.timeouts.heartbeat * 2)
        record = supervisor.get_run(plan.run_id)
        assert record['state'] not in TERMINAL_STATES and record['failure'] is None
        release.set()
        assert finished(supervisor, plan)['state'] == 'COMPLETED'
    finally:
        release.set()
        supervisor.close()


@pytest.mark.parametrize('behavior,error', [
    ('hash', 'sha256'), ('truncated', 'size'),
    ('spoof_published', 'recording'), ('prepare_failure', 'prepare_failure'),
    ('capture_failure', 'capture_failure'), ('seal_failure', 'seal_failure'), ('rename_failure', 'rename failed'),
])
def test_requested_recording_failures_are_persistent(publication_source, recorded_trajectory, behavior, error):
    catalog, plan = publication_source
    store_event(plan, recorded_trajectory)
    with Supervisor(catalog, worker_command=worker(behavior)) as supervisor:
        supervisor.submit(plan)
        record = finished(supervisor, plan)
        assert record['state'] == 'FAILED'
        assert error in record['failure']
        assert record['recording']['status'] == 'failed'
        assert error in record['recording']['error']
        assert not any(a['kind'] == 'trajectory' for a in record['artifacts'])


@pytest.mark.parametrize('behavior', ['orphan', 'forced'])
def test_unreceived_final_files_are_not_discovered_on_exit_or_restart(publication_source, recorded_trajectory, behavior):
    catalog, plan = publication_source
    store_event(plan, recorded_trajectory)
    with Supervisor(catalog, worker_command=worker(behavior)) as supervisor:
        supervisor.submit(plan)
        if behavior == 'forced':
            wait_for(lambda: supervisor.get_run(plan.run_id)['heartbeat'])
            supervisor.cancel(plan.run_id)
        record = finished(supervisor, plan)
        assert not any(a['kind'] == 'trajectory' for a in record['artifacts'])
        assert record['state'] == ('CANCELLED' if behavior == 'forced' else 'FAILED')
        assert record['recording']['status'] == 'failed'
    assert Path(plan.artifact_directory, recorded_trajectory.path).is_file()
    with Supervisor(catalog) as supervisor:
        assert not any(a['kind'] == 'trajectory' for a in supervisor.artifacts(plan.run_id))


def test_mp4_append_survives_active_metrics_terminal_and_postprocessing_failure(publication_source, recorded_trajectory):
    catalog, plan = publication_source
    store_event(plan, recorded_trajectory)
    with Supervisor(catalog, worker_command=worker('cooperative')) as supervisor:
        supervisor.submit(plan)
        wait_for(lambda: any(a['kind'] == 'trajectory' for a in supervisor.artifacts(plan.run_id)))
        video = video_artifact(plan, recorded_trajectory)
        published = supervisor.publish_recorded_artifact(video).result(timeout=5)
        assert published == video.model_dump(mode='json')
        steps = supervisor.get_run(plan.run_id)['metrics']['steps']
        wait_for(lambda: supervisor.get_run(plan.run_id)['metrics']['steps'] > steps)
        assert video.id in [a['id'] for a in supervisor.artifacts(plan.run_id)]
        supervisor.cancel(plan.run_id)
        before = finished(supervisor, plan)
        assert video.id in [a['id'] for a in before['artifacts']]
        invalid = video_artifact(plan, recorded_trajectory)
        invalid.sha256 = '0' * 64
        publication = supervisor.publish_recorded_artifact(invalid)
        with pytest.raises(ValueError, match='sha256'):
            publication.result(timeout=5)
        assert supervisor.cancel_recorded_artifact(publication) is False
        with pytest.raises(ValueError, match='sha256'):
            publication.result(timeout=5)
        assert supervisor.get_run(plan.run_id) == before
        later = video_artifact(plan, recorded_trajectory)
        supervisor.publish_recorded_artifact(later).result(timeout=5)
        after = supervisor.get_run(plan.run_id)
        assert after['state'] == before['state']
        assert {video.id, later.id} <= {a['id'] for a in after['artifacts']}


def test_duplicate_trajectory_event_never_registers_twice(publication_source, recorded_trajectory):
    catalog, plan = publication_source
    store_event(plan, recorded_trajectory)
    with Supervisor(catalog, worker_command=worker('duplicate')) as supervisor:
        supervisor.submit(plan)
        record = finished(supervisor, plan)
        assert record['state'] == 'FAILED'
        assert 'trajectory' in record['failure']
        assert [a['id'] for a in record['artifacts']].count(recorded_trajectory.id) == 1


def test_worker_failure_preserves_already_published_trajectory(publication_source, recorded_trajectory):
    catalog, plan = publication_source
    store_event(plan, recorded_trajectory)
    with Supervisor(catalog, worker_command=worker('cooperative')) as supervisor:
        supervisor.submit(plan)
        wait_for(lambda: supervisor.get_run(plan.run_id)['recording']['status'] == 'published')
        # 发布完成后再发生 Worker 错误，源 Run 失败但可靠产物仍可用。
        Path(plan.artifact_directory, 'fail-worker').touch()
        record = finished(supervisor, plan)
        assert record['state'] == 'FAILED' and record['failure'] == 'later Worker failure'
        assert record['recording']['status'] == 'published'
        assert record['recording']['error'] is None
        assert recorded_trajectory.id in [a['id'] for a in record['artifacts']]


def test_cancel_waits_for_hash_reader_before_file_cleanup(publication_source, recorded_trajectory, monkeypatch):
    catalog, plan = publication_source
    store_event(plan, recorded_trajectory)
    with Supervisor(catalog, worker_command=worker('completed')) as supervisor:
        supervisor.submit(plan)
        finished(supervisor, plan)
        entered, release = block_verification(monkeypatch)
        video = video_artifact(plan, recorded_trajectory)
        publication = supervisor.publish_recorded_artifact(video)
        assert entered.wait(5)
        with ThreadPoolExecutor(max_workers=1) as calls:
            cancellation = calls.submit(supervisor.cancel_recorded_artifact, publication)
            try:
                wait_for(publication.cancelled)
                assert not cancellation.done()
                assert supervisor.get_run(plan.run_id)['state'] == 'COMPLETED'
            finally:
                release.set()
            assert cancellation.result(timeout=5) is True
        Path(plan.artifact_directory, video.path).unlink()
        with pytest.raises(CancelledError):
            publication.result()
        assert video.id not in [a['id'] for a in supervisor.artifacts(plan.run_id)]


def test_cancelled_invalid_file_finishes_reading_without_failing_source_run(publication_source, recorded_trajectory, monkeypatch):
    catalog, plan = publication_source
    store_event(plan, recorded_trajectory)
    with Supervisor(catalog, worker_command=worker('completed')) as supervisor:
        supervisor.submit(plan)
        before = finished(supervisor, plan)
        entered, release = block_verification(monkeypatch)
        video = video_artifact(plan, recorded_trajectory)
        video.sha256 = '0' * 64
        publication = supervisor.publish_recorded_artifact(video)
        assert entered.wait(5)
        with ThreadPoolExecutor(max_workers=1) as calls:
            cancellation = calls.submit(supervisor.cancel_recorded_artifact, publication)
            try:
                wait_for(publication.cancelled)
            finally:
                release.set()
            assert cancellation.result(timeout=5) is True
        with pytest.raises(CancelledError):
            publication.result()
        assert supervisor.get_run(plan.run_id) == before


def test_registration_wins_cancellation_and_shutdown_waits_for_verification(publication_source, recorded_trajectory, monkeypatch):
    catalog, plan = publication_source
    store_event(plan, recorded_trajectory)
    supervisor = Supervisor(catalog, worker_command=worker('completed')).start()
    try:
        supervisor.submit(plan)
        finished(supervisor, plan)
        first = video_artifact(plan, recorded_trajectory)
        publication = supervisor.publish_recorded_artifact(first)
        publication.result(timeout=5)
        assert supervisor.cancel_recorded_artifact(publication) is False
        entered, release = block_verification(monkeypatch)
        second = video_artifact(plan, recorded_trajectory)
        publication = supervisor.publish_recorded_artifact(second)
        assert entered.wait(5)
        with ThreadPoolExecutor(max_workers=1) as calls:
            closing = calls.submit(supervisor.close)
            try:
                wait_for(lambda: supervisor._closing)
                assert not closing.done() and not supervisor._lock_file.closed
            finally:
                release.set()
            closing.result(timeout=5)
        assert publication.result(timeout=5)['id'] == second.id
        assert {first.id, second.id} <= {a['id'] for a in supervisor.artifacts(plan.run_id)}
        assert supervisor._lock_file.closed
    finally:
        if 'release' in locals():
            release.set()
        supervisor.close()


def test_cancellation_waits_when_registration_has_already_started(publication_source, recorded_trajectory, monkeypatch):
    catalog, plan = publication_source
    store_event(plan, recorded_trajectory)
    with Supervisor(catalog, worker_command=worker('completed')) as supervisor:
        supervisor.submit(plan)
        finished(supervisor, plan)
        video = video_artifact(plan, recorded_trajectory)
        entered, release, requested = threading.Event(), threading.Event(), threading.Event()
        save = supervisor._save

        def blocked_save(record):
            if any(a['id'] == video.id for a in record['artifacts']):
                entered.set()
                assert release.wait(10)
            return save(record)

        with monkeypatch.context() as patch:
            patch.setattr(supervisor, '_save', blocked_save)
            publication = supervisor.publish_recorded_artifact(video)
            assert entered.wait(5)

            def cancel():
                requested.set()
                return supervisor.cancel_recorded_artifact(publication)

            with ThreadPoolExecutor(max_workers=1) as calls:
                cancellation = calls.submit(cancel)
                try:
                    assert requested.wait(5)
                    assert not cancellation.done() and not publication.cancelled()
                finally:
                    release.set()
                assert cancellation.result(timeout=5) is False
        assert publication.result(timeout=5)['id'] == video.id
        assert video.id in [a['id'] for a in supervisor.artifacts(plan.run_id)]


@pytest.mark.parametrize('fault', ['run', 'id', 'path', 'size', 'hash', 'missing', 'symlink'])
def test_recorded_file_validation_rejects_invalid_final_reference(recording_source, recorded_trajectory, tmp_path, fault):
    from muse_server.storage import verify_recorded_file
    plan, _ = recording_source
    artifact = recorded_trajectory.model_copy()
    path = Path(plan.artifact_directory, artifact.path)
    if fault == 'run': artifact.run = 'other'
    if fault == 'id': artifact.id = f'{plan.run_id}:trajectory:not-uuid'
    if fault == 'path': artifact.path += '.tmp'
    if fault == 'size': artifact.size += 1
    if fault == 'hash': artifact.sha256 = '0' * 64
    if fault == 'missing': path.unlink()
    if fault == 'symlink':
        outside = tmp_path / 'outside.zip'
        path.rename(outside)
        path.symlink_to(outside)
    with pytest.raises(ValueError):
        verify_recorded_file(artifact, plan.run_id, Path(plan.artifact_directory))


def test_recorded_file_verification_accepts_both_fixed_artifact_kinds(recording_source, recorded_trajectory):
    from muse_server.storage import verify_recorded_file
    plan, _ = recording_source
    video = video_artifact(plan, recorded_trajectory)
    for artifact in (recorded_trajectory, video):
        assert verify_recorded_file(artifact, plan.run_id, Path(plan.artifact_directory)) == Path(plan.artifact_directory, artifact.path)
