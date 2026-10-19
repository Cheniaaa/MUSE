"""每 Run 一份显示状态；所有调用由 Supervisor 锁串行化。"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import time
from uuid import uuid4

from muse_sdk.run_protocol import (DisplayPatch, DisplaySettings, PreviewApplied, PreviewFrame, PreviewSelection, PreviewSelections,
    PreviewModel, SceneObject, WorkerControl, check_preview_capacity, drain_preview, encode_preview,
    preview_socketpair, send_preview)
from muse_sdk.view_protocol import ViewIntent


class _ViewerSlot:
    def __init__(self, name, settings):
        self.name = name
        self.requested = settings
        self.effective = self.requested
        self.committed = settings
        self.sent = PreviewSelection(selection_seq=0, display=settings)
        self.selection_seq = self.effective_seq = 0
        self.status, self.error, self.viewer_url = 'ready', None, None
        self.viewer_error = None
        self.viewer_socket = None
        self.process = None
        self.loaded = False
        self.visible = None
        self.cursor = None
        self.pending = None
        self.outgoing = b''
        self.incoming = b''
        self.request_at = 0.
        self.launch_at = 0.

    def snapshot(self):
        value = dict(requested=self.requested.model_dump(), effective=self.effective.model_dump(),
            selection_seq=self.selection_seq, effective_seq=self.effective_seq,
            status=self.status, error=self.error, frame=self.visible.model_dump() if self.visible else None)
        if self.name == 'viser':
            value['viewer_url'] = self.viewer_url
        return value


class DisplaySession:
    def __init__(self, plan, send_control, *, access_host='127.0.0.1', viewer_port=8080, send_command,
                 on_log_started):
        self.plan, self.send_control = plan, send_control
        self.send_command = send_command
        self.on_log_started = on_log_started
        self.interaction = {'command': None, 'result': None}
        self.access_host, self.viewer_port = access_host, viewer_port
        self.slots = {name: _ViewerSlot(name, DisplaySettings(enabled=False,
            world_id=plan.experiment.preview.world_ids[0], rate_hz=plan.experiment.preview.rate_hz))
            for name in ('viser', 'native')}
        self.model = None
        self.model_error = None
        self.worker_socket = None
        self.retired = []
        self.ack_timeout = 10.
        self.worker_running_at = None
        self.released = False
        self._view_ids = set()
        self._view_events = {}
        self._entity_snapshot = None
        self._objects = []
        objects_path = Path(plan.artifact_directory) / 'preview_objects.json'
        if objects_path.exists():
            self._objects = [SceneObject.model_validate(item) for item in json.loads(objects_path.read_text())]

    def snapshot(self):
        return {**{name: slot.snapshot() for name, slot in self.slots.items()},
                'interaction': deepcopy(self.interaction)}

    def register_client(self):
        view_id = uuid4().hex
        self._view_ids.add(view_id)
        self._view_events.setdefault(view_id, [])
        self._state_to_viewer(self.slots['viser'])
        return {'view_id': view_id, 'connected': False}

    def remove_client(self, view_id):
        self._view_ids.discard(view_id)
        self._view_events.pop(view_id, None)
        self._state_to_viewer(self.slots['viser'])

    def client_events(self, view_id):
        if view_id not in self._view_ids:
            raise ValueError('Unknown view_id')
        events = self._view_events.setdefault(view_id, [])
        self._view_events[view_id] = []
        return events

    def view_intent(self, view_id, intent):
        if view_id not in self._view_ids:
            raise ValueError('Unknown view_id')
        intent = ViewIntent.model_validate(intent)
        if intent.kind == 'visibility' and (intent.object_id is None or intent.visible is None):
            raise ValueError('Visibility requires object_id and visible')
        slot = self.slots['viser']
        if slot.process is None:
            raise ValueError('Run viewer is not available')
        if intent.object_id is not None and intent.object_id not in {obj.object_id for obj in self._objects}:
            raise ValueError('Unknown scene object')
        if len(slot.outgoing) > 65536:
            raise ValueError('Run viewer control is busy')
        slot.outgoing += (json.dumps({'kind': 'view_intent', 'view_id': view_id,
                                      'intent': intent.model_dump(mode='json')}) + '\n').encode()

    def _public_objects(self):
        return [{key: value for key, value in item.model_dump(mode='json').items()
                 if key not in ('body_id', 'geom_ids', 'local_position', 'local_orientation_matrix')}
                for item in self._objects]

    def entities(self):
        if self._entity_snapshot is None:
            return {'objects': self._public_objects(), 'frame': None, 'available': False,
                    'reason': 'Run has ended' if self.released else 'No current preview frame'}
        slot = self.slots['viser']
        available = (not self.released and slot.requested.enabled and slot.effective.enabled
                     and slot.selection_seq == slot.effective_seq and slot.viewer_error is None)
        return {**deepcopy(self._entity_snapshot), 'available': available,
                'reason': None if available else 'Last snapshot; live preview is unavailable'}

    def _project_entities(self, header, arrays):
        values = struct.unpack('<' + 'f' * (header.nbody * 12), arrays)
        matrix_start = header.nbody * 3
        result = []
        for obj, public in zip(self._objects, self._public_objects(), strict=True):
            body = obj.body_id
            position = list(values[body * 3:body * 3 + 3])
            rotation = list(values[matrix_start + body * 9:matrix_start + body * 9 + 9])
            if obj.kind == 'environment':
                position = [position[i] + sum(rotation[i * 3 + j] * obj.local_position[j] for j in range(3))
                            for i in range(3)]
                rotation = [sum(rotation[i * 3 + k] * obj.local_orientation_matrix[k * 3 + j] for k in range(3))
                            for i in range(3) for j in range(3)]
            result.append({**public, 'pose': {'position': position, 'orientation_matrix': rotation}})
        self._entity_snapshot = {'objects': result, 'frame': header.model_dump(mode='json')}

    def set_interaction(self, interaction):
        self.interaction = deepcopy(interaction)
        self._state_to_viewer(self.slots['viser'])

    def _publish(self, slot, selection):
        choices = {name: selection if current is slot else current.sent for name, current in self.slots.items()}
        self.send_control(self.plan.run_id, WorkerControl(run_id=self.plan.run_id, kind='preview',
            preview=PreviewSelections(**choices)))
        slot.sent = selection
        slot.request_at = time.monotonic()

    def patch(self, patch):
        change = DisplayPatch.model_validate(patch)
        name, = change.model_fields_set
        slot = self.slots[name]
        settings = DisplaySettings.model_validate({**slot.requested.model_dump(),
            **getattr(change, name).model_dump(exclude_unset=True)})
        if settings.world_id >= self.plan.experiment.topology.world_count:
            raise ValueError('world_id outside Run world_count')
        if self.released:
            raise ValueError('Run display session has ended')
        seq = slot.selection_seq + 1
        selection = PreviewSelection(selection_seq=seq, display=settings)
        # Native上下文成功之前不发布启用请求；仍经过Worker存活准入。
        self._publish(slot, slot.sent if name == 'native' and settings.enabled and not slot.loaded else selection)
        slot.requested, slot.selection_seq = settings, seq
        slot.status, slot.error = 'pending', None
        slot.viewer_error = None
        if not settings.enabled:
            self._stop_viewer(slot)
        elif self.model_error:
            self.fail(self.model_error)
        elif self.model and slot.process is None:
            self._launch(slot)
        self._state_to_viewer(slot)
        return self.snapshot()

    def worker_event(self, kind, payload):
        if kind == 'running':
            self.worker_running_at = time.monotonic()
        elif kind == 'preview_model':
            model = PreviewModel.model_validate(payload)
            root = Path(self.plan.artifact_directory).resolve()
            path = (root / model.path).resolve()
            if Path(model.path).is_absolute() or not path.is_relative_to(root):
                raise ValueError('Preview model must be relative to Run artifact directory')
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != model.sha256:
                raise ValueError('Preview model hash mismatch')
            if self.model is not None and self.model != model:
                raise ValueError('Runtime model replacement is not supported in M1')
            self.model = model
            self._objects = model.objects
            for slot in self.slots.values():
                if slot.requested.enabled and slot.process is None:
                    self._launch(slot)
        elif kind == 'preview_error':
            self.model_error = str(payload['message'])
            self.fail(self.model_error)
        elif kind == 'preview_applied':
            applied = PreviewApplied.model_validate(payload)
            slot = self.slots[applied.viewer]
            if not slot.effective_seq < applied.selection_seq <= slot.sent.selection_seq:
                return
            if slot.name == 'native' and applied.display.enabled and (not slot.loaded or not slot.requested.enabled):
                return
            slot.effective, slot.effective_seq = applied.display, applied.selection_seq
            slot.visible = slot.cursor = slot.pending = None
            if slot.name == 'viser':
                self._entity_snapshot = None
            slot.status, slot.error = 'pending', None
            if self.model_error:
                self.fail(self.model_error)
            elif slot.viewer_error:
                slot.status, slot.error = 'error', slot.viewer_error
            elif slot.effective_seq == slot.selection_seq and not slot.effective.enabled:
                slot.status = 'ready'
                slot.committed = slot.effective
            self._state_to_viewer(slot)

    def fail(self, message):
        for slot in self.slots.values():
            slot.status, slot.error = 'error', message
            self._state_to_viewer(slot)

    def _state_to_viewer(self, slot):
        if slot.process is not None:
            message = {'kind': 'state', 'state': slot.snapshot()}
            if slot.name == 'viser':
                message.update(interaction=self.interaction, view_ids=sorted(self._view_ids))
            slot.outgoing += (json.dumps(message) + '\n').encode()

    def _launch(self, slot):
        parent, child = preview_socketpair()
        log = None
        try:
            check_preview_capacity(parent, self.model.nbody)
            directory = Path(self.plan.artifact_directory)
            log = (directory / f'logs/{slot.name}.log').open('ab', buffering=0)
            if slot.name == 'viser':
                self.on_log_started()
            command = [sys.executable, '-m', f'muse_server.viewers.{slot.name}_process',
                '--snapshot-fd', str(child.fileno()), '--model', str(directory / self.model.path),
                '--sha256', self.model.sha256, '--nbody', str(self.model.nbody),
                '--run-id', self.plan.run_id]
            if slot.name == 'viser':
                command += ['--objects', str(directory / 'preview_objects.json'),
                    '--world-count', str(self.plan.experiment.topology.world_count),
                    '--port', str(self.viewer_port), '--mode', self.plan.mode]
            slot.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=log, pass_fds=(child.fileno(),), start_new_session=True)
            os.set_blocking(slot.process.stdin.fileno(), False)
            os.set_blocking(slot.process.stdout.fileno(), False)
            slot.viewer_socket = parent
            slot.loaded = False
            slot.visible = slot.cursor = slot.pending = None
            slot.outgoing = slot.incoming = b''
            slot.launch_at = time.monotonic()
            self._state_to_viewer(slot)
        except (OSError, ValueError) as exc:
            parent.close()
            self._viewer_failed(slot, f'{slot.name.title()} launch failed: {exc}')
        finally:
            child.close()
            if log is not None:
                log.close()

    def _stop_viewer(self, slot):
        if slot.process is not None:
            if slot.process.poll() is None:
                slot.process.terminate()
            self.retired.append((slot.process, time.monotonic()))
            slot.process.stdin.close()
            slot.process.stdout.close()
            slot.process = None
        if slot.viewer_socket is not None:
            slot.viewer_socket.close()
            slot.viewer_socket = None
        slot.viewer_url = None
        slot.loaded = False
        slot.pending = slot.visible = slot.cursor = None
        slot.outgoing = slot.incoming = b''

    def _viewer_failed(self, slot, message):
        self._stop_viewer(slot)
        slot.viewer_error = slot.error = message
        slot.status = 'error'
        if slot.name == 'native':
            slot.requested = slot.committed.model_copy(update={'enabled': False})
            slot.selection_seq += 1
            self._publish(slot, PreviewSelection(selection_seq=slot.selection_seq, display=slot.requested))

    def release(self):
        self.released = True
        for slot in self.slots.values():
            self._stop_viewer(slot)
            slot.status, slot.error = 'error', 'Run has ended; display released'
        if self.worker_socket is not None:
            self.worker_socket.close()
            self.worker_socket = None

    @property
    def drained(self):
        return not self.retired and all(slot.process is None for slot in self.slots.values())

    def _viewer_event(self, slot, message):
        kind = message['kind']
        if kind == 'loaded':
            slot.loaded = True
            if slot.name == 'viser':
                if message['host'] != '127.0.0.1' or not 0 < message['port'] < 65536:
                    raise ValueError('Invalid Viser bind address')
                slot.viewer_url = f'http://{self.access_host}:{message["port"]}'
            else:
                self._publish(slot, PreviewSelection(selection_seq=slot.selection_seq, display=slot.requested))
        elif kind == 'frame':
            header = PreviewFrame.model_validate(message['header'])
            if (header.viewer, header.run_id, header.selection_seq, header.world_id, header.scene_hash) != (
                    slot.name, self.plan.run_id, slot.effective_seq, slot.effective.world_id, self.model.sha256):
                return
            if slot.visible is not None and (header.episode, header.decision) < (slot.visible.episode, slot.visible.decision):
                return
            slot.visible = header
            previous = slot.status, slot.error
            if (slot.loaded and slot.selection_seq == slot.effective_seq and slot.requested.enabled
                    and not self.model_error and not slot.viewer_error):
                slot.status, slot.error = 'ready', None
                slot.committed = slot.effective
            if previous == (slot.status, slot.error):
                return
        elif kind == 'patch':
            try:
                self.patch({'viser': message['patch']})
            except ValueError as exc:
                self._state_to_viewer(slot)
                slot.outgoing += (json.dumps({'kind': 'request_error', 'message': str(exc)}) + '\n').encode()
            return
        elif kind == 'command':
            try:
                self.send_command(self.plan.run_id, message['command'], message.get('world_id'))
            except ValueError as exc:
                self._state_to_viewer(slot)
                slot.outgoing += (json.dumps({'kind': 'command_error', 'message': str(exc)}) + '\n').encode()
            return
        elif kind == 'view':
            view_id = message.get('view_id')
            if view_id in self._view_ids:
                event = dict(message['event'])
                if event['kind'] == 'selection':
                    if self._entity_snapshot is None or event['frame'] is None:
                        return
                    current = self._entity_snapshot['frame']
                    if any(event['frame'][key] != current[key] for key in (
                            'viewer', 'run_id', 'world_id', 'selection_seq', 'scene_hash', 'episode')):
                        return
                event['view_id'] = view_id
                # 选择与配对均为当前状态；每类只保留最新，慢客户端不会积压。
                events = self._view_events[view_id]
                events[:] = [previous for previous in events if previous['kind'] != event['kind']]
                events.append(event)
            return
        elif kind == 'error':
            self._viewer_failed(slot, message['message'])
        elif kind == 'closed' and slot.name == 'native':
            self.patch({'native': {'enabled': False}})
        self._state_to_viewer(slot)

    def tick(self):
        for process, started in self.retired[:]:
            if process.poll() is not None:
                self.retired.remove((process, started))
            elif time.monotonic() - started > 1.:
                process.kill()
        if self.released:
            return
        try:
            if self.worker_socket is not None:
                latest = drain_preview(self.worker_socket, run_id=self.plan.run_id,
                    selections={name: PreviewSelection(selection_seq=slot.effective_seq, display=slot.effective)
                                for name, slot in self.slots.items()},
                    scene_hash=self.model.sha256 if self.model else '')
                for name, (header, arrays) in latest.items():
                    slot = self.slots[name]
                    if slot.viewer_socket is None:
                        continue
                    key = header.episode, header.decision
                    if slot.cursor is None or key > slot.cursor:
                        slot.cursor = key
                        slot.pending = encode_preview(header, arrays)
                        if name == 'viser':
                            self._project_entities(header, arrays)
        except (OSError, ValueError) as exc:
            self.model_error = f'Preview unavailable: {exc}'
            self.fail(self.model_error)
        for slot in self.slots.values():
            self._tick_viewer(slot)

    def _tick_viewer(self, slot):
        try:
            if slot.process is not None:
                # stdout在exit判断/写stdin之前drain，保留正常关窗回执。
                while slot.process is not None:
                    try:
                        chunk = os.read(slot.process.stdout.fileno(), 65536)
                    except BlockingIOError:
                        break
                    if not chunk:
                        break
                    slot.incoming += chunk
                    while b'\n' in slot.incoming:
                        line, slot.incoming = slot.incoming.split(b'\n', 1)
                        self._viewer_event(slot, json.loads(line))
                if slot.process is None:
                    return
                if slot.process.poll() is not None:
                    self._viewer_failed(slot, slot.viewer_error or f'{slot.name.title()} exited {slot.process.returncode}')
                    return
                if slot.outgoing:
                    try:
                        count = os.write(slot.process.stdin.fileno(), slot.outgoing)
                        slot.outgoing = slot.outgoing[count:]
                    except BlockingIOError:
                        pass
                slot.pending = send_preview(slot.viewer_socket, slot.pending)
                if not slot.loaded and time.monotonic() - slot.launch_at > 30:
                    self._viewer_failed(slot, f'{slot.name.title()} model load timeout')
            # 准备/编译期限归Supervisor；ACK只能从Worker可消费控制后开始计时。
            if (self.worker_running_at is not None and slot.sent.selection_seq > slot.effective_seq
                    and time.monotonic() - max(slot.request_at, self.worker_running_at) > self.ack_timeout):
                if slot.error != 'preview_applied timeout' and slot.viewer_error is None:
                    slot.status, slot.error = 'error', 'preview_applied timeout'
                    self._state_to_viewer(slot)
        except (OSError, ValueError, KeyError) as exc:
            self._viewer_failed(slot, f'Preview unavailable: {exc}')

    def close(self):
        self.release()
        for process, _ in self.retired:
            if process.poll() is None:
                process.kill()
            process.wait()
        self.retired.clear()
