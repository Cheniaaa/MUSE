"""草稿CPU子进程所有者；只解析可信资源引用，不导入运行/图形实现。"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from uuid import uuid4

from muse_sdk.config import ConfigError, resolve_config
from muse_sdk.experiment import EntityModelRef
from muse_sdk.scene_edit import PreviewEntity, ScenePreviewRequest
from muse_sdk.view_protocol import ViewIntent

from .errors import ResourceNotFound


def preview_command(identity):
    return [sys.executable, '-m', 'muse_runtime.draft_session', '--view-id', identity]


@dataclass
class _Session:
    identity: str
    draft_id: str
    requested_revision: int
    temporary: tempfile.TemporaryDirectory
    process: subprocess.Popen
    state: dict
    reader: threading.Thread | None = None
    timer: threading.Timer | None = None
    inflight: tuple | None = None
    pending: tuple | None = None
    attached: bool = False
    closed: bool = False
    terminated: bool = False
    events: deque = field(default_factory=deque)
    cleanup_lock: threading.Lock = field(default_factory=threading.Lock)


class PreviewManager:
    def __init__(self, catalog, *, attach_timeout=30., process_command=preview_command):
        self.catalog = catalog
        self.attach_timeout, self.process_command = attach_timeout, process_command
        self._lock = threading.RLock()
        self._sessions = {}

    def _session(self, identity):
        try:
            return self._sessions[identity]
        except KeyError as exc:
            raise ResourceNotFound(identity) from exc

    def _resolve(self, payload):
        request = ScenePreviewRequest.model_validate(payload)
        environment = self.catalog.get_resource('environment', request.environment.resource_id,
                                                 request.environment.revision)
        if environment.sha256 != request.environment.sha256:
            raise ConfigError('environment.sha256', 'Environment content identity does not match')
        if request.environment_edit is not None and request.environment_edit.base != request.environment:
            raise ConfigError('environment_edit.base', 'Edit base must match the preview environment')
        builtins = self.catalog.builtins
        paths = {'environment': environment.path}
        entities, omissions = [], []
        environment_ids = set(environment.metadata['geometry_summary']['geoms'])
        if request.environment_edit:
            environment_ids.difference_update(request.environment_edit.removed)
            environment_ids.update(request.environment_edit.upserts)
        for index, item in enumerate(request.entities):
            path = f'entities.{index}'
            model = builtins.entity_models.get(item.model_id)
            if model is None or model.revision != item.model_revision:
                raise ConfigError(path + '.model_id', 'Unknown entity model revision')
            if item.object_id in environment_ids or item.object_id.startswith('@unnamed/'):
                raise ConfigError(path + '.object_id', 'Entity object ID conflicts with an environment object')
            outputs = []
            if item.sensor_profile_id is None:
                omissions.append(item.object_id)
            else:
                if item.sensor_profile_id not in model.metadata['sensor_profile_ids']:
                    raise ConfigError(path + '.sensor_profile_id', 'Unknown or incompatible sensor profile')
                profile = builtins.sensors[item.sensor_profile_id]
                config = resolve_config(profile['config_schema'], {}, item.sensor_config, require_complete=False)
                missing = [name for name in profile['config_schema'].get('required', []) if name not in config]
                if missing:
                    omissions.append(item.object_id + ': missing ' + ', '.join(missing))
                else:
                    outputs = list(profile['outputs'].values())
            reference = EntityModelRef(**model.model_dump(include={'resource_id', 'revision', 'sha256'}),
                physics_asset=model.metadata['physics_asset'], root_body=model.metadata['root_body'],
                support_geoms=model.metadata['support_geoms'], radio_devices=model.metadata.get('radio_devices', {}))
            entities.append(PreviewEntity(object_id=item.object_id, group_id=item.group_id, slot=item.slot,
                                           model=reference, pose=item.pose, sensor_outputs=outputs).model_dump(mode='json'))
            paths['entity_model:' + item.model_id] = model.path
        resolved = dict(draft_id=request.draft_id, revision=request.revision,
            environment=request.environment.model_dump(mode='json'), environment_scene=environment.metadata['scene'],
            environment_edit=request.environment_edit.model_dump(mode='json') if request.environment_edit else None,
            entities=entities)
        return resolved, paths, omissions

    def create(self, request):
        resolved = self._resolve(request)
        identity = uuid4().hex
        temporary = tempfile.TemporaryDirectory(prefix='preview-', dir=self.catalog.directory)
        try:
            with (Path(temporary.name) / 'worker.log').open('wb') as log:
                process = subprocess.Popen(self.process_command(identity), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=log, text=True, bufsize=1, start_new_session=True)
        except OSError:
            temporary.cleanup()
            raise
        state = dict(session_id=identity, status='preparing', applied_revision=None,
                     viewer_url=None, error=None, objects=[], sensor_omissions=resolved[2])
        session = _Session(identity, resolved[0]['draft_id'], resolved[0]['revision'], temporary, process, state)
        session.reader = threading.Thread(target=self._read, args=(session,), daemon=True)
        session.timer = threading.Timer(self.attach_timeout, self._close, args=(identity, True))
        session.timer.daemon = True
        with self._lock:
            self._sessions[identity] = session
            self._compile(session, resolved)
            session.reader.start()
            session.timer.start()
            return deepcopy(state)

    def _send(self, session, message):
        session.process.stdin.write(json.dumps(message, allow_nan=False) + '\n')
        session.process.stdin.flush()

    def _compile(self, session, resolved):
        request, paths, omissions = resolved
        destination = Path(session.temporary.name) / f"revision-{request['revision']}"
        session.inflight = (request['revision'], destination, omissions)
        self._send(session, dict(kind='compile', request=request, resource_paths=paths, destination=str(destination)))

    def update(self, identity, request):
        resolved = self._resolve(request)
        with self._lock:
            session = self._session(identity)
            if session.closed:
                raise ConfigError('session_id', 'Preview session is closing')
            if session.terminated:
                raise ConfigError('session_id', 'Preview session terminated; recreate it: ' + session.state['error'])
            if resolved[0]['draft_id'] != session.draft_id or resolved[0]['revision'] <= session.requested_revision:
                raise ConfigError('revision', 'Preview update requires the same draft and a newer revision')
            session.requested_revision = resolved[0]['revision']
            session.state.update(status='preparing', error=None)
            if session.inflight is not None:
                session.pending = resolved
            else:
                self._compile(session, resolved)
            return deepcopy(session.state)

    def _read(self, session):
        try:
            for line in session.process.stdout:
                message = json.loads(line)
                with self._lock:
                    if session.closed:
                        break
                    kind = message['kind']
                    if kind in ('ready', 'error'):
                        revision, destination, omissions = session.inflight
                        if message['revision'] != revision:
                            raise ValueError('Preview child revision does not match the active compilation')
                        if revision == session.requested_revision:
                            if kind == 'ready':
                                objects = message['objects']
                                if len({item['object_id'] for item in objects}) != len(objects):
                                    raise ValueError('Duplicate compiled preview object ID')
                                session.state.update(status='ready', applied_revision=revision, error=None,
                                    viewer_url=message['viewer_url'], sensor_omissions=omissions,
                                    objects=[{key: value for key, value in item.items() if key not in (
                                        'body_id', 'geom_ids', 'local_position', 'local_orientation_matrix')}
                                             for item in objects])
                            else:
                                session.state.update(status='error', error=message['message'])
                        session.inflight = None
                        if destination.exists():
                            shutil.rmtree(destination)
                        if session.pending is not None:
                            pending, session.pending = session.pending, None
                            self._compile(session, pending)
                    elif kind in ('view', 'gesture'):
                        if kind == 'gesture':
                            gesture = message['gesture']
                            if gesture['base_revision'] != session.state['applied_revision'] or session.state['status'] != 'ready':
                                continue
                            if gesture['phase'] == 'update' and session.events and session.events[-1]['kind'] == 'gesture':
                                previous = session.events[-1]['gesture']
                                if previous['phase'] == 'update' and previous['gesture_id'] == gesture['gesture_id']:
                                    session.events.pop()
                        if len(session.events) >= 256:
                            raise ValueError('Preview control consumer is too slow; reconnect the preview')
                        session.events.append({**message, 'session_id': session.identity})
                    else:
                        raise ValueError('Unknown preview child event: ' + kind)
            code = session.process.wait()
            with self._lock:
                if not session.closed:
                    session.terminated = True
                    session.inflight = session.pending = None
                    session.state.update(status='error', error=f'Preview process exited {code}')
        except (OSError, ValueError, KeyError) as exc:
            with self._lock:
                if not session.closed:
                    session.terminated = True
                    session.inflight = session.pending = None
                    session.state.update(status='error', error=str(exc))
        finally:
            session.process.stdout.close()

    def get(self, identity):
        with self._lock:
            return deepcopy(self._session(identity).state)

    def attach(self, identity):
        with self._lock:
            session = self._session(identity)
            if session.attached or session.closed:
                raise ConfigError('session_id', 'Preview control is already attached or closing')
            session.attached = True
            session.timer.cancel()

    def view_intent(self, identity, intent):
        value = ViewIntent.model_validate(intent)
        with self._lock:
            session = self._session(identity)
            if not session.attached:
                raise ConfigError('session_id', 'Preview control is not ready')
            if session.state['status'] != 'ready':
                # HTTP编译可先于旧WS意图到达；返回当前状态，不把旧意图重放到新图。
                return deepcopy(session.state)
            self._send(session, {'kind': 'intent', 'intent': value.model_dump(mode='json')})

    def events(self, identity):
        with self._lock:
            session = self._session(identity)
            result = list(session.events)
            session.events.clear()
            return result

    def _close(self, identity, unattached_only=False):
        with self._lock:
            session = self._sessions.get(identity)
        if session is None:
            return
        with session.cleanup_lock:
            with self._lock:
                if identity not in self._sessions or (unattached_only and session.attached):
                    return
                session.closed = True
                session.timer.cancel()
            if session.process.poll() is None:
                session.process.terminate()
            try:
                session.process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                with self._lock:
                    session.state.update(status='error', error='CPU preview shutdown unconfirmed')
                raise
            session.process.stdin.close()
            session.reader.join()
            session.temporary.cleanup()
            with self._lock:
                del self._sessions[identity]

    def close(self, identity):
        self._close(identity)

    def shutdown(self):
        with self._lock:
            identities = list(self._sessions)
        for identity in identities:
            self.close(identity)
