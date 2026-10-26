"""历史CPU会话所有者；Server主进程不读取轨迹数组或导入图形实现。"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from muse_sdk.config import ConfigError
from muse_sdk.replay_protocol import ReplayCommand, ReplayRequest, ReplayState
from muse_sdk.view_protocol import ViewIntent


def replay_command(identity: str) -> list[str]:
    return [sys.executable, '-m', 'muse_server.viewers.replay_process', '--view-id', identity]


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


class ReplayManager:
    def __init__(self, catalog, supervisor, *, attach_timeout=30., process_command=replay_command):
        self.catalog, self.supervisor = catalog, supervisor
        self.attach_timeout, self.process_command = attach_timeout, process_command
        self._lock = threading.RLock()
        self._sessions = {}

    def create(self, request: ReplayRequest) -> dict:
        record = self.supervisor.get_run(request.run_id)
        value = next((item for item in record['artifacts'] if item['id'] == request.trajectory_artifact_id), None)
        if value is None:
            raise KeyError(request.trajectory_artifact_id)
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
            return deepcopy(self._sessions[identity].state)

    def attach(self, identity: str) -> None:
        with self._lock:
            session = self._sessions[identity]
            if session.attached or session.stopping:
                raise ConfigError('session_id', 'Replay control is already attached or closing')
            session.attached = True
            session.timer.cancel()

    def control(self, identity: str, message: dict) -> None:
        overflow = None
        with self._lock:
            session = self._sessions[identity]
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
            session = self._sessions[identity]
            result = list(session.events)
            session.events.clear()
            return result

    def _expire(self, identity):
        with self._lock:
            session = self._sessions.get(identity)
        if session is not None:
            self._dispose(session, remove=True, unattached_only=True)

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
            # 不持manager锁等待进程/线程/reader；TERM和必要KILL均覆盖本会话进程组。
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(session.process.pid, sig)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + session.stop_timeout
                try:
                    session.process.wait(timeout=session.stop_timeout)
                except subprocess.TimeoutExpired:
                    continue
                while _group_alive(session.process.pid) and time.monotonic() < deadline:
                    time.sleep(.01)
                if not _group_alive(session.process.pid):
                    break
            else:
                raise TimeoutError('Replay process group shutdown unconfirmed')
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
                if remove:
                    self._sessions.pop(session.identity, None)
        finally:
            session.cleanup_lock.release()

    def close(self, identity: str) -> None:
        with self._lock:
            session = self._sessions[identity]
        self._dispose(session, remove=True)

    def shutdown(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            self._dispose(session, remove=True)
