"""历史CPU会话所有者；Server主进程不读取轨迹数组或导入图形实现。"""
from collections import deque
from concurrent.futures import CancelledError, Future
from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import select
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from muse_sdk.config import ConfigError
from muse_sdk.recording_protocol import RecordingArtifact
from muse_sdk.replay_protocol import RecordingRequest, ReplayCommand, ReplayRequest, ReplayState
from muse_sdk.view_protocol import ViewIntent

from .errors import ResourceNotFound


def replay_command(identity: str) -> list[str]:
    return [sys.executable, '-m', 'muse_server.viewers.replay_process', '--view-id', identity]


def recording_command(job_path: Path) -> list[str]:
    return [sys.executable, '-m', 'muse_server.viewers.recording_process', '--job', str(job_path)]


def _group_alive(pgid):
    # Linux进程组可能留下已退出、等待其父进程回收的zombie；它们不再持有资源。
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            continue
        if int(fields[2]) == pgid and fields[0] != 'Z':
            return True
    return False


def _stop_group(process, stop_timeout):
    """历史/编码两个真实消费者共用；确认组内没有仍持有资源的活进程。"""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + stop_timeout
        try:
            process.wait(timeout=stop_timeout)
        except subprocess.TimeoutExpired:
            continue
        while _group_alive(process.pid) and time.monotonic() < deadline:
            time.sleep(.01)
        if not _group_alive(process.pid):
            return
    raise TimeoutError('Viewer process group shutdown unconfirmed')


@dataclass
class _Export:
    identity: str
    state: dict
    prepared: dict | None = None
    staging: tempfile.TemporaryDirectory | None = None
    process: subprocess.Popen | None = None
    process_stopped: bool = False
    publication: Future | None = None
    owner: threading.Thread | None = None
    prepared_event: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    process_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class _Session:
    identity: str
    temporary: tempfile.TemporaryDirectory
    process: subprocess.Popen
    stop_timeout: float
    state: dict
    reader: threading.Thread | None = None
    timer: threading.Timer | None = None
    attached: bool = False
    stopping: bool = False
    disposed: bool = False
    events: deque = field(default_factory=deque)
    cleanup_lock: threading.Lock = field(default_factory=threading.Lock)
    export_job: _Export | None = None


class ReplayManager:
    def __init__(self, catalog, supervisor, *, attach_timeout=30., process_command=replay_command):
        self.catalog, self.supervisor = catalog, supervisor
        self.attach_timeout, self.process_command = attach_timeout, process_command
        self._lock = threading.RLock()
        self._sessions = {}

    def _session(self, identity):
        try:
            return self._sessions[identity]
        except KeyError as exc:
            raise ResourceNotFound(identity) from exc

    def create(self, request: ReplayRequest) -> dict:
        record = self.supervisor.get_run(request.run_id)
        value = next((item for item in record['artifacts'] if item['id'] == request.trajectory_artifact_id), None)
        if value is None:
            raise ResourceNotFound(request.trajectory_artifact_id)
        if value['kind'] != 'trajectory' or value['run'] != request.run_id:
            raise ConfigError('trajectory_artifact_id', 'Artifact must be a trajectory published by this Run')
        stop_timeout = self.supervisor.get_plan(request.run_id).timeouts.stop
        identity = uuid4().hex
        temporary = tempfile.TemporaryDirectory(prefix='replay-', dir=self.catalog.directory)
        try:
            with (Path(temporary.name) / 'worker.log').open('wb') as log:
                process = subprocess.Popen(self.process_command(identity), stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, start_new_session=True)
        except OSError:
            temporary.cleanup()
            raise
        state = ReplayState(session_id=identity, status='preparing', source=request).model_dump(mode='json')
        session = _Session(identity, temporary, process, stop_timeout, state)
        session.timer = threading.Timer(self.attach_timeout, self._expire, args=(identity,))
        session.timer.daemon = True
        with self._lock:
            self._sessions[identity] = session
            failure = None
            try:
                self._send(session, dict(kind='prepare', artifact=value,
                    run_directory=record['artifact_directory'], directory=temporary.name))
            except OSError as exc:
                failure = str(exc)
            session.reader = threading.Thread(target=self._read, args=(session, failure), daemon=True)
            session.reader.start()
            session.timer.start()
        # 即使child很快就准备成功，POST也返回这次创建的preparing快照。
        return state.copy()

    def _send(self, session, message):
        session.process.stdin.write(json.dumps(message, allow_nan=False) + '\n')
        session.process.stdin.flush()

    def _enqueue(self, session, message):
        if len(session.events) >= 256:
            raise ValueError('Replay control consumer is too slow; control receipt buffer overflow')
        session.events.append({**message, 'session_id': session.identity})

    def _read(self, session, failure=None):
        try:
            for line in session.process.stdout:
                message = json.loads(line)
                with self._lock:
                    if session.stopping:
                        break
                    kind = message['kind']
                    if kind == 'state':
                        state = message['state']
                        if state['viewer_url'] is not None:
                            url = urlsplit(state['viewer_url'])
                            host = self.supervisor.access_host
                            host = f'[{host}]' if ':' in host else host
                            state['viewer_url'] = urlunsplit(url._replace(netloc=f'{host}:{url.port}'))
                        session.state = state
                    elif kind in ('view', 'command_result'):
                        self._enqueue(session, message)
                    elif kind == 'export_prepared':
                        job = session.export_job
                        if job is not None and message['export_id'] == job.identity and not job.done.is_set():
                            job.prepared = message
                            job.prepared_event.set()
                    elif kind == 'error':
                        failure = message['message']
                        break
                    else:
                        raise ValueError('Unknown replay child event: ' + kind)
        except (OSError, ValueError, KeyError) as exc:
            failure = str(exc)
        finally:
            session.process.stdout.close()
            with self._lock:
                stopping = session.stopping
            if not stopping:
                self._dispose(session, remove=False, failure=failure)

    def get(self, identity: str) -> dict:
        with self._lock:
            return deepcopy(self._session(identity).state)

    def attach(self, identity: str) -> None:
        with self._lock:
            session = self._session(identity)
            if session.attached or session.stopping:
                raise ConfigError('session_id', 'Replay control is already attached or closing')
            session.attached = True
            session.timer.cancel()

    def control(self, identity: str, message: dict) -> None:
        overflow = None
        with self._lock:
            session = self._session(identity)
            if not session.attached:
                raise ConfigError('session_id', 'Replay control is not attached')
            kind = message.get('kind') if isinstance(message, dict) else None
            try:
                if not isinstance(message, dict):
                    raise ValueError('Expected replay control object')
                if kind == 'playback' and set(message) == {'kind', 'command'}:
                    value = ReplayCommand.model_validate(message['command'])
                    payload = dict(kind=kind, command=value.model_dump(mode='json'))
                elif kind == 'view' and set(message) == {'kind', 'intent'}:
                    value = ViewIntent.model_validate(message['intent'])
                    payload = dict(kind=kind, intent=value.model_dump(mode='json'))
                else:
                    raise ValueError('Expected playback command or view intent')
                if session.stopping or session.state['status'] != 'ready':
                    raise ValueError('Replay is not ready')
                self._send(session, payload)
            except (ValueError, OSError) as exc:
                raw = message.get('command') if isinstance(message, dict) else None
                event = (dict(kind='command_result', command_seq=raw.get('command_seq') if isinstance(raw, dict) else None,
                              status='error', error=str(exc), revision=session.state['revision'])
                         if kind == 'playback' else dict(kind='view', event=dict(kind='error', message=str(exc))))
                try:
                    self._enqueue(session, event)
                except ValueError as full:
                    overflow = str(full)
        if overflow is not None:
            self._dispose(session, remove=False, failure=overflow)

    def events(self, identity: str) -> list[dict]:
        with self._lock:
            session = self._session(identity)
            result = list(session.events)
            session.events.clear()
            return result

    def _expire(self, identity):
        with self._lock:
            session = self._sessions.get(identity)
        if session is not None:
            self._dispose(session, remove=True, unattached_only=True)

    def export(self, identity: str, request: RecordingRequest) -> dict:
        with self._lock:
            session = self._session(identity)
            if session.stopping or session.state['status'] != 'ready':
                raise RuntimeError('Replay is not ready')
            if not session.state['camera_ready']:
                raise RuntimeError('Replay camera is not paired and initialized')
            if not any(item['episode'] == request.episode
                       for item in session.state['episodes'].get(str(request.world_id), [])):
                raise ConfigError('episode', 'World/episode outside recorded range')
            if session.export_job is not None and not session.export_job.done.is_set():
                raise RuntimeError('Replay export is busy')
            export_id = uuid4().hex
            state = dict(export_id=export_id, status='preparing', frames_written=0, total_frames=0,
                         artifact_id=None, error=None)
            initial = state.copy()
            job = _Export(export_id, state)
            previous, session.export_job = session.export_job, job
            try:
                self._send(session, dict(kind='prepare_export', export=initial,
                                         request=request.model_dump(mode='json')))
            except OSError:
                session.export_job = previous
                raise
            job.owner = threading.Thread(target=self._run_export, args=(session, job), daemon=True)
            job.owner.start()
            return initial

    def _export_state(self, session, job, **updates):
        with self._lock:
            job.state.update(updates)
            if session.export_job is job and not session.stopping and session.process.poll() is None:
                # manager不拥有状态revision；由历史child顺序合并后发布。
                self._send(session, {'kind': 'export_state', 'export': job.state.copy()})

    def _stop_export_process(self, session, job):
        with job.process_lock:
            if not job.process_stopped:
                _stop_group(job.process, session.stop_timeout)
                job.process_stopped = True

    def _run_export(self, session, job):
        status, error, artifact_id = 'failed', None, None
        registered = False
        run_directory = None
        try:
            if not job.prepared_event.wait(session.stop_timeout):
                raise TimeoutError('Replay export snapshot timed out')
            if job.cancelled.is_set():
                raise CancelledError()
            prepared = job.prepared
            if prepared['error'] is not None:
                raise ValueError(prepared['error'])
            source = session.state['source']
            run_directory = Path(self.supervisor.get_run(source['run_id'])['artifact_directory'])
            job.staging = tempfile.TemporaryDirectory(prefix='recording-', dir=run_directory)
            job_path = Path(job.staging.name) / 'job.json'
            job_path.write_text(json.dumps(dict(directory=prepared['directory'], staging_directory=job.staging.name,
                artifact_directory=str(run_directory), export_id=job.identity, run_id=source['run_id'],
                source_trajectory_id=source['trajectory_artifact_id'], request=prepared['request'],
                camera=prepared['camera']), allow_nan=False))
            with self._lock:
                if job.cancelled.is_set():
                    raise CancelledError()
                with (Path(job.staging.name) / 'worker.log').open('wb') as log:
                    job.process = subprocess.Popen(recording_command(job_path), stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE, stderr=log, start_new_session=True)
            self._export_state(session, job, status='rendering')
            incoming, artifact = b'', None
            while not job.cancelled.is_set():
                readable = select.select([job.process.stdout], [], [], .05)[0]
                if not readable:
                    if job.process.poll() is not None:
                        break
                    continue
                chunk = os.read(job.process.stdout.fileno(), 65536)
                if not chunk:
                    break
                incoming += chunk
                while b'\n' in incoming:
                    line, incoming = incoming.split(b'\n', 1)
                    message = json.loads(line)
                    if message['kind'] == 'progress':
                        self._export_state(session, job, frames_written=message['frames_written'],
                                           total_frames=message['total_frames'])
                    elif message['kind'] == 'artifact':
                        # JSON数组→tuple是序列化边界语义，不能用严格Python dict校验代替。
                        artifact = RecordingArtifact.model_validate_json(json.dumps(message['artifact']))
                    elif message['kind'] == 'error':
                        raise RuntimeError(message['message'])
                    else:
                        raise ValueError('Unknown recording child event: ' + message['kind'])
            if not job.cancelled.is_set():
                job.process.wait(timeout=session.stop_timeout)
            self._stop_export_process(session, job)
            if job.cancelled.is_set():
                raise CancelledError()
            if incoming:
                raise ValueError('Recording child exited with incomplete JSONL')
            if job.process.returncode != 0:
                raise RuntimeError(f'Recording process exited {job.process.returncode}')
            if artifact is None:
                raise ValueError('Recording process exited without an artifact')
            with self._lock:
                if job.cancelled.is_set():
                    raise CancelledError()
                job.publication = self.supervisor.publish_recorded_artifact(artifact)
            self._export_state(session, job, status='publishing')
            while not job.publication.done() and not job.cancelled.wait(.05):
                pass
        except CancelledError:
            status = 'cancelled'
        except Exception as exc:
            # 有限owner线程的错误出口；根因发布给GUI，不改变源Run。
            error = str(exc)
        finally:
            if job.process is not None:
                self._stop_export_process(session, job)
                job.process.stdout.close()
            if job.publication is not None:
                cancelled = False
                if job.cancelled.is_set() or not job.publication.done():
                    cancelled = self.supervisor.cancel_recorded_artifact(job.publication)
                if cancelled:
                    status = 'cancelled' if job.cancelled.is_set() else 'failed'
                else:
                    try:
                        published = job.publication.result()
                    except Exception as exc:
                        status, error = 'failed', str(exc)
                    else:
                        registered, status, error = True, 'completed', None
                        artifact_id = published['id']
            if job.staging is not None:
                if not registered:
                    (run_directory / 'recordings' / (job.identity + '.mp4')).unlink(missing_ok=True)
                job.staging.cleanup()
            final = dict(status=status, error=error, artifact_id=artifact_id)
            if registered:
                count = published['metadata']['frame_count']
                final.update(frames_written=count, total_frames=count)
            self._export_state(session, job, **final)
            job.done.set()

    def _cancel_export_job(self, session, job):
        with self._lock:
            if not job.done.is_set():
                job.cancelled.set()
                job.prepared_event.set()
            process = job.process
        if process is not None:
            self._stop_export_process(session, job)
        job.owner.join()
        if not job.done.is_set():
            raise RuntimeError('Recording export cleanup unconfirmed')

    def cancel_export(self, identity: str, export_id: str) -> None:
        with self._lock:
            session = self._session(identity)
            job = session.export_job
            if job is None or job.identity != export_id:
                raise ResourceNotFound(export_id)
        self._cancel_export_job(session, job)

    def _dispose(self, session, *, remove, failure=None, unattached_only=False):
        # 外部close可能正在join本reader；reader不能反过来等待其cleanup锁。
        if not session.cleanup_lock.acquire(blocking=session.reader is not threading.current_thread()):
            return
        try:
            with self._lock:
                if unattached_only and session.attached:
                    return
                already_disposed = session.disposed
                if not already_disposed:
                    session.stopping = True
                    session.timer.cancel()
            if already_disposed:
                if session.reader is not threading.current_thread():
                    session.reader.join(timeout=session.stop_timeout)
                    if session.reader.is_alive():
                        raise TimeoutError('Replay protocol reader shutdown unconfirmed')
                if remove:
                    with self._lock:
                        self._sessions.pop(session.identity, None)
                return
            # 必须先收口export/FFmpeg及发布读线程，再删除历史reader依赖的目录。
            if session.export_job is not None:
                self._cancel_export_job(session, session.export_job)
            _stop_group(session.process, session.stop_timeout)
            try:
                session.process.stdin.close()
            except BrokenPipeError as exc:
                failure = failure or str(exc)
            if session.reader is not threading.current_thread():
                session.reader.join(timeout=session.stop_timeout)
                if session.reader.is_alive():
                    raise TimeoutError('Replay protocol reader shutdown unconfirmed')
            session.temporary.cleanup()
            with self._lock:
                session.disposed = True
                session.state = {**session.state, 'revision': session.state['revision'] + 1,
                    'status': 'closing' if remove else 'error', 'playing': False, 'camera_ready': False,
                    'viewer_url': None, 'error': None if remove else failure or f'Replay process exited {session.process.returncode}'}
                if session.export_job is not None:
                    session.state['export'] = session.export_job.state.copy()
                if remove:
                    self._sessions.pop(session.identity, None)
        finally:
            session.cleanup_lock.release()

    def close(self, identity: str) -> None:
        with self._lock:
            session = self._session(identity)
        self._dispose(session, remove=True)

    def shutdown(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            self._dispose(session, remove=True)
