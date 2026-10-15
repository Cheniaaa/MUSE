"""专用JSONL fd；普通Python及native输出只进入日志pipe。"""
import os
import sys
import threading
import time
from collections import deque

from muse_sdk.run_protocol import WorkerControl, WorkerEvent


class WorkerProtocol:
    def __init__(self, run_id):
        sys.stdout.flush()
        sys.stderr.flush()
        self.fd = os.dup(1)
        os.set_inheritable(self.fd, False)
        os.dup2(2, 1)
        self.run_id = run_id
        self.seq = 0
        self.sim_time = 0.
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._error = None
        self._preview_controls = deque()
        self._commands = deque()
        self._control_lock = threading.Lock()
        self._control_thread = threading.Thread(target=self._read_controls, daemon=True)
        self._control_thread.start()

    def _read_controls(self):
        try:
            buffer = b''
            while chunk := os.read(0, 4096):
                buffer += chunk
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    control = WorkerControl.model_validate_json(line)
                    if control.run_id != self.run_id:
                        raise ValueError('Control run_id mismatch')
                    if control.kind == 'cancel':
                        self._cancel.set()
                    else:
                        with self._control_lock:
                            if control.kind == 'command':
                                self._commands.append(control.command)
                            else:
                                # 显示请求只消费最新设置；环境命令仍保留原顺序。
                                self._preview_controls.clear()
                                self._preview_controls.append(control)
                if len(buffer) > 65536:
                    raise ValueError('Control line exceeds 64 KiB')
            # Server消失后在下个安全边界停止，不继续无人托管的训练。
            self._cancel.set()
        except BaseException as exc:
            self._error = exc

    def read_control(self):
        if self._error is not None:
            raise self._error
        return 'cancel' if self._cancel.is_set() else None

    def take_command(self):
        with self._control_lock:
            return self._commands.popleft() if self._commands else None

    def take_preview_controls(self):
        with self._control_lock:
            controls = list(self._preview_controls)
            self._preview_controls.clear()
        return controls

    def emit(self, kind, payload=None, *, sim_time=None):
        with self._lock:
            if sim_time is not None:
                self.sim_time = sim_time
            self.seq += 1
            event = WorkerEvent(event_seq=self.seq, run_id=self.run_id, kind=kind,
                                sim_time=self.sim_time, wall_time=time.time(), payload=payload or {})
            data = (event.model_dump_json() + '\n').encode()
            while data:
                count = os.write(self.fd, data)
                data = data[count:]

    def close(self):
        with self._lock:
            if self.fd >= 0:
                os.close(self.fd)
                self.fd = -1
