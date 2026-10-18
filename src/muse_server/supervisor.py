"""单GPU进程队列。SQLite持久状态，pipe持续drain，终态等待EOF和exit。"""
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import sqlite3
import subprocess
import sys
import threading
import time

from muse_sdk.experiment import RunPlan
from muse_sdk.run_protocol import (CheckpointArtifact, CommandResult, InteractiveCommand,
                                   WorkerControl, WorkerEvent, preview_socketpair)
from .display import DisplaySession
from .storage import verify_checkpoint_file

TERMINAL_STATES = frozenset(('COMPLETED', 'CANCELLED', 'FAILED', 'LOST'))


def process_identity(pid):
    """None表示已死；读取失败抛出，不把权限错误当作退出。"""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip() + ':' + fields[19]
    except FileNotFoundError:
        return None


def worker_command(plan):
    return [sys.executable, '-m', 'muse_runtime.worker', '--plan',
            str(Path(plan.artifact_directory) / 'resolved_run_plan.json')]


class Supervisor:
    def __init__(self, catalog, *, worker_command=worker_command, access_host='127.0.0.1', viewer_port=8080):
        if access_host in ('0.0.0.0', '::'):
            raise ValueError('access_host must be a browser-accessible host, not a wildcard')
        self.catalog = catalog
        self.database = catalog.database
        self.worker_command = worker_command
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._closing = False
        self._thread = None
        self._lock_file = None
        self._active = None
        self._lost = []
        self._events = {}
        self._selector = selectors.DefaultSelector()
        self._fatal = None
        self._displays = {}
        self.access_host, self.viewer_port = access_host, viewer_port

    def _display(self, run_id):
        if run_id not in self._displays:
            session = DisplaySession(self.get_plan(run_id), self.send_control,
                access_host=self.access_host, viewer_port=self.viewer_port, send_command=self.viewer_command,
                on_log_started=self._viewer_log_started)
            session.set_interaction(self.get_run(run_id)['interaction'])
            self._displays[run_id] = session
            if self.get_run(run_id)['state'] in TERMINAL_STATES:
                session.release()
        return self._displays[run_id]

    def get_display(self, run_id):
        with self._lock:
            return self._display(run_id).snapshot()

    def patch_display(self, run_id, patch):
        with self._lock:
            return self._display(run_id).patch(patch)

    def _viewer_log_started(self):
        # 显示只在活动Worker期间启动；写入者通知真实打开事实，重启后仍可辨别缺失日志。
        record = self._active['record']
        record['viser_log_started'] = True
        self._save(record)

    def _connect(self):
        return sqlite3.connect(self.database)

    def start(self):
        with self._lock:
            if self._thread is not None:
                return self
            self._lock_file = (self.catalog.directory / 'supervisor.lock').open('a')
            try:
                fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self._lock_file.close()
                self._lock_file = None
                raise RuntimeError('Another supervisor owns this data directory') from exc
            with self._connect() as db:
                db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, record TEXT NOT NULL)')
            with self._connect() as db:
                records = [json.loads(row[0]) for row in db.execute('SELECT record FROM runs ORDER BY rowid')]
            for record in records:
                if record['state'] not in TERMINAL_STATES | {'QUEUED'}:
                    record['state'] = 'LOST'
                    record['failure'] = 'Server restarted; no Worker takeover'
                    self._publish_terminal(record)
                    self._save(record)
                if record['state'] == 'LOST' and record['pid'] is not None:
                    self._lost.append(record)
            self._thread = threading.Thread(target=self._loop, name='muse-supervisor', daemon=True)
            self._thread.start()
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()

    @property
    def gpu_busy(self):
        with self._lock:
            return self._active is not None or bool(self._lost)

    @property
    def error(self):
        return self._fatal

    def _save(self, record):
        with self._connect() as db:
            db.execute('INSERT INTO runs VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET record=excluded.record',
                       (record['run_id'], json.dumps(record, allow_nan=False)))

    def get_run(self, run_id):
        with self._lock, self._connect() as db:
            row = db.execute('SELECT record FROM runs WHERE id=?', (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            return json.loads(row[0])

    def list_runs(self, *, experiment_id=None, mode=None, state=None, offset=0, limit=50):
        filters, values = [], []
        for field, value in (('experiment_id', experiment_id), ('mode', mode), ('state', state)):
            if value is not None:
                filters.append(f"json_extract(record,'$.{field}')=?")
                values.append(value)
        where = ' WHERE ' + ' AND '.join(filters) if filters else ''
        with self._lock, self._connect() as db:
            rows = db.execute('SELECT record FROM runs' + where + ' ORDER BY rowid LIMIT ? OFFSET ?',
                              (*values, limit + 1, offset)).fetchall()
        return {'items': [json.loads(row[0]) for row in rows[:limit]],
                'next_offset': offset + limit if len(rows) > limit else None}

    def get_plan(self, run_id):
        record = self.get_run(run_id)
        return RunPlan.model_validate_json(Path(record['plan_path']).read_text())

    def submit(self, plan):
        with self._lock:
            if self._closing or self._fatal or self._thread is None:
                raise RuntimeError('Supervisor is not accepting runs')
            record = dict(run_id=plan.run_id, experiment_id=plan.experiment_id, state='QUEUED',
                          mode=plan.mode, max_decision_steps=plan.max_decision_steps,
                          checkpoint=plan.checkpoint.model_dump(mode='json') if plan.checkpoint else None,
                          viser_log_started=False,
                          device=plan.device, pid=None, pgid=None, process_start=None,
                          plan_path=str(Path(plan.artifact_directory) / 'resolved_run_plan.json'),
                          artifact_directory=plan.artifact_directory, created_at=time.time(),
                          heartbeat=None, exit_code=None, failure=None, worker_terminal=None,
                          cancel_requested=False, forced=False, event_seq=0, sim_time=0.,
                          metrics={}, artifacts=[], interaction={'command': None, 'result': None})
            with self._connect() as db:
                db.execute('INSERT INTO runs VALUES (?, ?)', (plan.run_id, json.dumps(record)))
            self._wake.set()
            return deepcopy(record)

    def send_control(self, run_id, control):
        """环境命令必须经过唯一准入；其他控制复用可靠写入点。"""
        if control.run_id != run_id:
            raise ValueError('Control run_id mismatch')
        if control.kind == 'command':
            return self.submit_command(run_id, control.command)
        return self._send_control(run_id, control)

    def _send_control(self, run_id, control):
        with self._lock:
            active = self._active
            if active is None or active['record']['run_id'] != run_id:
                raise ValueError('Run has no live Worker')
            active['control'] += (control.model_dump_json() + '\n').encode()
            self._wake.set()

    def submit_command(self, run_id, command: InteractiveCommand):
        """只保留最后一条命令；锁内完成准入、去重、可靠投递与记录。"""
        with self._lock:
            self.get_run(run_id)
            active = self._active
            if active is None or active['record']['run_id'] != run_id:
                raise ValueError('Run has no live Worker')
            record, plan = active['record'], active['plan']
            if plan.mode != 'INTERACTIVE':
                raise ValueError('Commands require INTERACTIVE mode')
            if (record['state'] not in ('PREPARING', 'COMPILING', 'RUNNING')
                    or record['cancel_requested'] or record['worker_terminal'] is not None
                    or record['failure'] is not None or active['stop_at'] is not None
                    or self._closing or self._fatal or active['process'].poll() is not None):
                raise ValueError('Run is terminating or has no live Worker')
            if command.world_id is not None and command.world_id >= plan.experiment.topology.world_count:
                raise ValueError('world_id outside Run world_count')
            interaction = record['interaction']
            previous = interaction['command']
            data = command.model_dump(mode='json')
            if previous is not None and command.command_seq == previous['command_seq']:
                if data != previous:
                    raise ValueError('Conflicting command for command_seq')
                return deepcopy(interaction)
            next_seq = previous['command_seq'] + 1 if previous else 1
            if command.command_seq != next_seq:
                raise ValueError(f'Expected command_seq {next_seq}')
            if previous is not None and interaction['result'] is None:
                raise ValueError('Previous command is pending')
            self._send_control(run_id, WorkerControl(run_id=run_id, kind='command', command=command))
            record['interaction'] = {'command': data, 'result': None}
            self._save(record)
            if run_id in self._displays:
                self._displays[run_id].set_interaction(record['interaction'])
            return deepcopy(record['interaction'])

    def viewer_command(self, run_id, command, world_id=None):
        """Viser 只传意图，序号分配与 API 请求共享同一把锁。"""
        with self._lock:
            previous = self.get_run(run_id)['interaction']['command']
            request = InteractiveCommand(command_seq=previous['command_seq'] + 1 if previous else 1,
                                         command=command, world_id=world_id)
            return self.submit_command(run_id, request)

    def cancel(self, run_id):
        with self._lock:
            record = self.get_run(run_id)
            if record['state'] in TERMINAL_STATES or record['cancel_requested']:
                return record
            record['cancel_requested'] = True
            if record['state'] == 'QUEUED':
                record['state'] = 'CANCELLED'
                self._publish_terminal(record)
            else:
                active = self._active
                record['state'] = 'STOP_REQUESTED'
                active['record'] = record
                active['cancel_at'] = time.monotonic()
                self.send_control(run_id, WorkerControl(run_id=run_id, kind='cancel'))
            self._save(record)
            self._wake.set()
            return deepcopy(record)

    def events(self, run_id, after_seq=0):
        """有界当前事件；metric/heartbeat只保留最新值，最终terminal不丢。无历史回放。"""
        with self._lock:
            record = self.get_run(run_id)
            events = self._events.get(run_id, {})
            if not events and record['state'] in TERMINAL_STATES:
                return [self._terminal_event(record)] if record['event_seq'] > after_seq else []
            return deepcopy(sorted((e for e in events.values() if e['event_seq'] > after_seq),
                                   key=lambda e: e['event_seq']))

    def artifacts(self, run_id):
        return self.get_run(run_id)['artifacts']

    def _terminal_event(self, record):
        return dict(event_seq=record['event_seq'], run_id=record['run_id'], kind='terminal',
                    sim_time=record['sim_time'], wall_time=record['finished_at'],
                    payload=dict(state=record['state'], exit_code=record['exit_code'],
                                 failure=record['failure'], **record['metrics']))

    def _publish_terminal(self, record):
        if record['run_id'] in self._displays:
            self._displays[record['run_id']].release()
        record['event_seq'] += 1
        record['finished_at'] = time.time()
        self._events.setdefault(record['run_id'], {})['terminal'] = self._terminal_event(record)

    def _alive(self, record):
        try:
            identity = process_identity(record['pid'])
        except OSError:
            return True
        return identity is not None and (record['process_start'] is None or identity == record['process_start'])

    def _signal(self, record, sig):
        # start identity必须匹配，永远不向复用PID发送signal。
        try:
            identity = process_identity(record['pid'])
            if identity is not None and identity == record['process_start']:
                os.killpg(record['pgid'], sig)
                return True
        except ProcessLookupError:
            pass
        return False

    def _reap_lost(self):
        for record in self._lost[:]:
            if not self._alive(record):
                self._lost.remove(record)
                continue
            if 'stop_at' not in record:
                self._signal(record, signal.SIGTERM)
                record['stop_at'] = time.monotonic()
            elif time.monotonic() - record['stop_at'] > self.get_plan(record['run_id']).timeouts.stop:
                self._signal(record, signal.SIGKILL)

    def _spawn(self, record):
        plan = self.get_plan(record['run_id'])
        directory = Path(plan.artifact_directory)
        (directory / 'logs').mkdir(exist_ok=True)
        parent, child = preview_socketpair()
        try:
            process = subprocess.Popen(self.worker_command(plan), stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                                       pass_fds=(child.fileno(),), env={**os.environ, 'MUSE_PREVIEW_FD': str(child.fileno())})
        except OSError as exc:
            parent.close()
            record.update(state='FAILED', failure=f'Worker spawn failed: {exc}')
            self._publish_terminal(record)
            self._index_artifacts(record)
            self._save(record)
            return
        finally:
            child.close()
        self._display(plan.run_id).worker_socket = parent
        record.update(state='PREPARING', pid=process.pid, pgid=process.pid,
                      process_start=process_identity(process.pid))
        now = time.monotonic()
        self._active = dict(process=process, record=record, plan=plan, started=now,
                            phase_at=now, heartbeat_at=now, cancel_at=None, stop_at=None,
                            buffer=b'', control=b'', eof=set(),
                            log=(directory / 'logs/worker.log').open('ab', buffering=0),
                            events=(directory / 'events.jsonl').open('ab', buffering=0),
                            metrics=(directory / 'metrics.jsonl').open('ab', buffering=0))
        for pipe, kind in ((process.stdout, 'protocol'), (process.stderr, 'log')):
            os.set_blocking(pipe.fileno(), False)
            self._selector.register(pipe, selectors.EVENT_READ, kind)
        os.set_blocking(process.stdin.fileno(), False)
        self._save(record)

    def _fail(self, message):
        active = self._active
        record = active['record']
        record['failure'] = record['failure'] or message
        if active['stop_at'] is None:
            record['state'] = 'STOPPING'
            self._signal(record, signal.SIGTERM)
            active['stop_at'] = time.monotonic()
        self._save(record)

    def _event(self, line):
        active = self._active
        record = active['record']
        event = WorkerEvent.model_validate_json(line)
        if event.run_id != record['run_id'] or event.event_seq <= record['event_seq']:
            raise ValueError('Invalid event identity/order')
        if event.kind == 'command_result':
            result = CommandResult.model_validate(event.payload)
            interaction = record['interaction']
            identity = result.model_dump(include={'command_seq', 'command', 'world_id'})
            if identity != interaction['command'] or interaction['result'] is not None:
                raise ValueError('Invalid command_result identity: no matching pending command')
        data = event.model_dump(mode='json')
        active['events'].write(line + b'\n')
        record.update(event_seq=event.event_seq, sim_time=event.sim_time)
        if event.kind == 'failure':
            record['failure'] = record['failure'] or str(event.payload.get('message', event.payload))
        elif event.kind == 'terminal':
            if record['worker_terminal'] is not None:
                raise ValueError('Duplicate terminal')
            if event.payload.get('state') not in ('COMPLETED', 'CANCELLED', 'FAILED'):
                raise ValueError('Invalid terminal state')
            record['worker_terminal'] = event.payload
            record['metrics'].update({k: event.payload[k] for k in ('steps', 'updates') if k in event.payload})
            if event.payload['state'] == 'FAILED':
                record['failure'] = record['failure'] or 'Worker terminal FAILED'
        elif event.kind in ('preparing', 'compiling', 'running'):
            if not record['cancel_requested'] and active['stop_at'] is None:
                record['state'] = event.kind.upper()
                active['phase_at'] = time.monotonic()
        elif event.kind == 'heartbeat':
            record['heartbeat'] = event.wall_time
            active['heartbeat_at'] = time.monotonic()
        elif event.kind == 'metric':
            record['metrics'] = event.payload
            active['metrics'].write(line + b'\n')
        elif event.kind == 'checkpoint':
            artifact = CheckpointArtifact.model_validate(event.payload)
            verify_checkpoint_file(artifact, record['run_id'], record['artifact_directory'])
            record['artifacts'].append(artifact.model_dump(mode='json'))
        elif event.kind == 'command_result':
            record['interaction']['result'] = result.model_dump(mode='json')
            if event.run_id in self._displays:
                self._displays[event.run_id].set_interaction(record['interaction'])
        elif event.kind.startswith('preview_'):
            session = self._display(event.run_id)
            try:
                session.worker_event(event.kind, event.payload)
            except (ValueError, OSError, KeyError) as exc:
                session.model_error = str(exc)
                session.fail(str(exc))
        if event.kind != 'terminal':
            self._events.setdefault(event.run_id, {})[event.kind] = data
        self._save(record)

    def _drain(self):
        active = self._active
        for key, _ in self._selector.select(timeout=0):
            try:
                chunk = os.read(key.fd, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                active['eof'].add(key.data)
                self._selector.unregister(key.fileobj)
                key.fileobj.close()
                if key.data == 'protocol' and active['buffer']:
                    self._fail('Truncated protocol at EOF')
                continue
            if key.data == 'log':
                active['log'].write(chunk)
                continue
            active['buffer'] += chunk
            while b'\n' in active['buffer']:
                line, active['buffer'] = active['buffer'].split(b'\n', 1)
                try:
                    self._event(line)
                except (ValueError, TypeError) as exc:
                    self._fail(f'Invalid Worker protocol: {exc}')
            if len(active['buffer']) > 1024 * 1024:
                active['buffer'] = b''
                self._fail('Worker protocol line exceeds 1 MiB')

    def _tick(self):
        for session in self._displays.values():
            session.tick()
        self._reap_lost()
        if self._active is None:
            if not self._lost and not self._closing:
                queued = self.list_runs(state='QUEUED', limit=1)['items']
                if queued:
                    self._spawn(queued[0])
            return
        self._drain()
        active = self._active
        record, process = active['record'], active['process']
        if active['control']:
            try:
                count = os.write(process.stdin.fileno(), active['control'])
                active['control'] = active['control'][count:]
            except BlockingIOError:
                pass
            except BrokenPipeError:
                active['control'] = b''
        code = process.poll()
        if code is not None and active['eof'] == {'protocol', 'log'}:
            record['exit_code'] = code
            terminal = record['worker_terminal']
            if record['failure'] is not None:
                record['state'] = 'FAILED'
            elif record['cancel_requested'] and (
                    (record['forced'] and code in (-signal.SIGTERM, -signal.SIGKILL))
                    or (terminal and terminal['state'] == 'CANCELLED' and code == 0)):
                record['state'] = 'CANCELLED'
            elif code != 0 or terminal is None:
                record['state'] = 'FAILED'
                record['failure'] = f'Worker exited {code}' if code != 0 else 'Worker exited without terminal'
            elif terminal['state'] == 'COMPLETED':
                record['state'] = 'COMPLETED'
            else:
                record['state'] = 'FAILED'
                record['failure'] = 'Unrequested cancellation'
            process.stdin.close()
            for name in ('log', 'events', 'metrics'):
                active[name].close()
            self._publish_terminal(record)
            self._index_artifacts(record)
            self._save(record)
            self._active = None
            return
        now = time.monotonic()
        limits = active['plan'].timeouts
        if active['stop_at'] is not None:
            if now - active['stop_at'] >= limits.stop:
                self._signal(record, signal.SIGKILL)
        elif record['cancel_requested']:
            if now - active['cancel_at'] >= limits.cancel:
                record.update(state='STOPPING', forced=self._signal(record, signal.SIGTERM))
                active['stop_at'] = now
                self._save(record)
        elif record['state'] == 'PREPARING' and now - active['phase_at'] >= limits.startup:
            self._fail('startup timeout')
        elif record['state'] == 'COMPILING' and now - active['phase_at'] >= limits.compile:
            self._fail('compile timeout')
        elif now - active['heartbeat_at'] >= limits.heartbeat:
            self._fail('heartbeat timeout')

    def _index_artifacts(self, record):
        directory = Path(record['artifact_directory'])
        for relative, kind in (('resolved_run_plan.json', 'run_plan'), ('logs/worker.log', 'log'),
                                ('logs/viser.log', 'viser_log'),
                               ('events.jsonl', 'events'), ('metrics.jsonl', 'metrics')):
            path = directory / relative
            if path.is_file():
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                record['artifacts'].append(dict(id=f"{record['run_id']}:{kind}", run=record['run_id'],
                                                kind=kind, path=relative, size=path.stat().st_size,
                                                sha256=digest))

    def _loop(self):
        try:
            while True:
                with self._lock:
                    self._tick()
                    if self._closing and self._active is None and not self._lost:
                        return
                self._wake.wait(.01)
                self._wake.clear()
        except BaseException as exc:
            self._fatal = repr(exc)
            # 不释放锁和GPU占位；故障必须显式可见。

    def close(self):
        with self._lock:
            if self._thread is None:
                return
            self._closing = True
            if self._active is not None:
                self.cancel(self._active['record']['run_id'])
            self._wake.set()
        self._thread.join()
        if self._fatal:
            raise RuntimeError(f'Supervisor shutdown unconfirmed: {self._fatal}')
        self._selector.close()
        for session in self._displays.values():
            session.close()
        fcntl.flock(self._lock_file, fcntl.LOCK_UN)
        self._lock_file.close()
        self._thread = None
