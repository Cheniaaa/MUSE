"""每 Run 一份显示状态；所有调用由 Supervisor 锁串行化。"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from muse_sdk.run_protocol import (DisplayPatch, DisplaySettings, PreviewApplied, PreviewFrame,
    PreviewModel, WorkerControl, check_preview_capacity, drain_preview, encode_preview,
    preview_socketpair, send_preview)


class DisplaySession:
    def __init__(self, plan, send_control, *, access_host='127.0.0.1', viewer_port=8080, send_command,
                 on_log_started):
        self.plan, self.send_control = plan, send_control
        self.send_command = send_command
        self.on_log_started = on_log_started
        self.interaction = {'command': None, 'result': None}
        self.access_host, self.viewer_port = access_host, viewer_port
        self.requested = DisplaySettings(enabled=False, world_id=plan.experiment.preview.world_ids[0],
                                         rate_hz=plan.experiment.preview.rate_hz)
        self.effective = self.requested
        self.selection_seq = self.effective_seq = 0
        self.status, self.error, self.viewer_url = 'ready', None, None
        self.model = None
        self.model_error = None
        self.viewer_error = None
        self.worker_socket = None
        self.viewer_socket = None
        self.process = None
        self.retired = []
        self.loaded = False
        self.visible = None
        self.cursor = None
        self.pending = None
        self.outgoing = b''
        self.incoming = b''
        self.ack_timeout = 10.
        self.request_at = 0.
        self.launch_at = 0.
        self.released = False

    def snapshot(self):
        return dict(requested=self.requested.model_dump(), effective=self.effective.model_dump(),
                    selection_seq=self.selection_seq, effective_seq=self.effective_seq,
                    status=self.status, viewer_url=self.viewer_url, error=self.error,
                    interaction=deepcopy(self.interaction),
                    frame=self.visible.model_dump() if self.visible else None)

    def set_interaction(self, interaction):
        self.interaction = deepcopy(interaction)
        self._state_to_viewer()

    def patch(self, patch):
        change = DisplayPatch.model_validate(patch)
        settings = DisplaySettings.model_validate({**self.requested.model_dump(), **change.model_dump(exclude_unset=True)})
        if settings.world_id >= self.plan.experiment.topology.world_count:
            raise ValueError('world_id outside Run world_count')
        if self.released:
            raise ValueError('Run display session has ended')
        seq = self.selection_seq + 1
        control = WorkerControl(run_id=self.plan.run_id, kind='preview', selection_seq=seq, display=settings)
        self.send_control(self.plan.run_id, control)
        self.requested, self.selection_seq = settings, seq
        self.request_at = time.monotonic()
        self.status, self.error = 'pending', None
        self.viewer_error = None
        if not settings.enabled:
            self._stop_viewer()
        elif self.model_error:
            self.fail(self.model_error)
        elif self.model and self.process is None:
            self._launch()
        self._state_to_viewer()
        return self.snapshot()

    def worker_event(self, kind, payload):
        if kind == 'preview_model':
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
            if self.requested.enabled and self.process is None:
                self._launch()
        elif kind == 'preview_error':
            self.model_error = str(payload['message'])
            self.fail(self.model_error)
        elif kind == 'preview_applied':
            applied = PreviewApplied.model_validate(payload)
            if not self.effective_seq < applied.selection_seq <= self.selection_seq:
                return
            self.effective, self.effective_seq = applied.display, applied.selection_seq
            self.visible = self.cursor = self.pending = None
            self.status, self.error = 'pending', None
            if self.model_error:
                self.fail(self.model_error)
            elif self.viewer_error:
                self.fail(self.viewer_error)
            elif self.effective_seq == self.selection_seq and not self.effective.enabled:
                self.status = 'ready'
            self._state_to_viewer()

    def fail(self, message):
        self.status, self.error = 'error', message
        self._state_to_viewer()

    def _state_to_viewer(self):
        if self.process is not None:
            self.outgoing += (json.dumps({'kind': 'state', 'state': self.snapshot()}) + '\n').encode()

    def _launch(self):
        parent, child = preview_socketpair()
        log = None
        try:
            check_preview_capacity(parent, self.model.nbody)
            directory = Path(self.plan.artifact_directory)
            log = (directory / 'logs/viser.log').open('ab', buffering=0)
            self.on_log_started()
            command = [sys.executable, '-m', 'muse_server.viewers.viser_process',
                '--snapshot-fd', str(child.fileno()), '--model', str(directory / self.model.path),
                '--sha256', self.model.sha256, '--nbody', str(self.model.nbody),
                '--run-id', self.plan.run_id, '--world-count', str(self.plan.experiment.topology.world_count),
                '--port', str(self.viewer_port), '--mode', self.plan.mode]
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=log, pass_fds=(child.fileno(),), start_new_session=True)
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
            self.viewer_socket = parent
            self.loaded = False
            self.visible = self.cursor = self.pending = None
            self.outgoing = self.incoming = b''
            self.launch_at = time.monotonic()
            self._state_to_viewer()
        except (OSError, ValueError) as exc:
            parent.close()
            self.viewer_error = f'Viser launch failed: {exc}'
            self.fail(self.viewer_error)
        finally:
            child.close()
            if log is not None:
                log.close()

    def _stop_viewer(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            self.retired.append((self.process, time.monotonic()))
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None
        if self.viewer_socket is not None:
            self.viewer_socket.close()
            self.viewer_socket = None
        self.viewer_url = None
        self.loaded = False
        self.pending = self.visible = self.cursor = None
        self.outgoing = self.incoming = b''

    def release(self):
        self.released = True
        self._stop_viewer()
        if self.worker_socket is not None:
            self.worker_socket.close()
            self.worker_socket = None
        self.status, self.error = 'error', 'Run has ended; display released'

    def _viewer_event(self, message):
        kind = message['kind']
        if kind == 'loaded':
            if message['host'] != '127.0.0.1' or not 0 < message['port'] < 65536:
                raise ValueError('Invalid Viser bind address')
            self.loaded = True
            self.viewer_url = f'http://{self.access_host}:{message["port"]}'
        elif kind == 'frame':
            header = PreviewFrame.model_validate(message['header'])
            if (header.run_id, header.selection_seq, header.world_id, header.scene_hash) != (
                    self.plan.run_id, self.effective_seq, self.effective.world_id, self.model.sha256):
                return
            self.visible = header
            previous = self.status, self.error
            if (self.loaded and self.selection_seq == self.effective_seq and self.requested.enabled
                    and not self.model_error and not self.viewer_error):
                self.status, self.error = 'ready', None
            if previous == (self.status, self.error):
                return
        elif kind == 'patch':
            try:
                self.patch(message['patch'])
            except ValueError as exc:
                self._state_to_viewer()
                self.outgoing += (json.dumps({'kind': 'request_error', 'message': str(exc)}) + '\n').encode()
            return
        elif kind == 'command':
            try:
                self.send_command(self.plan.run_id, message['command'], message.get('world_id'))
            except ValueError as exc:
                self._state_to_viewer()
                self.outgoing += (json.dumps({'kind': 'command_error', 'message': str(exc)}) + '\n').encode()
            return
        elif kind == 'error':
            self.viewer_error = message['message']
            self.fail(self.viewer_error)
        self._state_to_viewer()

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
                    selection_seq=self.effective_seq, world_id=self.effective.world_id,
                    scene_hash=self.model.sha256 if self.model else '')
                if latest is not None and self.effective.enabled and self.viewer_socket is not None:
                    header, arrays = latest
                    key = header.episode, header.decision
                    if self.cursor is None or key > self.cursor:
                        self.cursor = key
                        self.pending = encode_preview(header, arrays)
            if self.process is not None:
                if self.process.poll() is not None:
                    code = self.process.returncode
                    self._stop_viewer()
                    self.viewer_error = self.viewer_error or f'Viser exited {code}'
                    self.fail(self.viewer_error)
                    return
                if self.outgoing:
                    try:
                        count = os.write(self.process.stdin.fileno(), self.outgoing)
                        self.outgoing = self.outgoing[count:]
                    except BlockingIOError:
                        pass
                while True:
                    try:
                        chunk = os.read(self.process.stdout.fileno(), 65536)
                    except BlockingIOError:
                        break
                    if not chunk:
                        break
                    self.incoming += chunk
                    while b'\n' in self.incoming:
                        line, self.incoming = self.incoming.split(b'\n', 1)
                        self._viewer_event(json.loads(line))
                self.pending = send_preview(self.viewer_socket, self.pending)
                if not self.loaded and time.monotonic() - self.launch_at > 30:
                    self._stop_viewer()
                    self.viewer_error = 'Viser model load timeout'
                    self.fail(self.viewer_error)
            if self.selection_seq > self.effective_seq and time.monotonic() - self.request_at > self.ack_timeout:
                if self.error != 'preview_applied timeout':
                    self.fail('preview_applied timeout')
        except (OSError, ValueError, KeyError) as exc:
            self._stop_viewer()
            self.viewer_error = f'Preview unavailable: {exc}'
            self.fail(self.viewer_error)

    def close(self):
        self.release()
        for process, _ in self.retired:
            if process.poll() is None:
                process.kill()
            process.wait()
        self.retired.clear()
