"""真实Server/双显示进程的隔离与收尾；CPU Native替身只替换窗口边界。"""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

import pytest

from test_display_session import display_run
from test_supervisor import wait_for


NATIVE_TRANSPORT = r'''
import json, os, select, signal, socket, sys
from pathlib import Path
from muse_sdk.run_protocol import PreviewSelection, drain_preview
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
root = Path(args['--model']).parent
sock = socket.socket(fileno=int(args['--snapshot-fd']))
sock.setblocking(False)
def emit(value):
    print(json.dumps(value), flush=True)
def stop(*_):
    print('native final bytes', file=sys.stderr, flush=True)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
emit({'kind': 'loaded'})
state = None
while True:
    if (root / 'native-close').exists():
        emit({'kind': 'closed'})
        stop()
    if select.select([sys.stdin], [], [], .01)[0]:
        line = sys.stdin.readline()
        if not line:
            stop()
        state = json.loads(line)['state']
    if state:
        latest = drain_preview(sock, run_id=args['--run-id'], scene_hash=args['--sha256'],
            selections={'native': PreviewSelection(selection_seq=state['effective_seq'], display=state['effective'])})
        if 'native' in latest:
            emit({'kind': 'frame', 'header': latest['native'][0].model_dump()})
'''


@pytest.fixture
def native_transport(display_run, monkeypatch):
    from muse_server import display
    spawn = display.subprocess.Popen
    def start(command, **kwargs):
        if command[1:3] == ['-m', 'muse_server.viewers.native_process']:
            command = [sys.executable, '-u', '-c', NATIVE_TRANSPORT, *command[3:]]
        return spawn(command, **kwargs)
    monkeypatch.setattr(display.subprocess, 'Popen', start)
    return display_run


def ready(supervisor, run_id, viewer):
    return wait_for(lambda: (slot if (slot := supervisor.get_display(run_id)[viewer])['status'] == 'ready'
                            else None), timeout=15)


def test_dual_display_patch_does_not_invalidate_other_viewer(native_transport):
    supervisor, plan = native_transport
    run_id = plan.run_id
    supervisor.patch_display(run_id, {'viser': {'enabled': True, 'world_id': 0}})
    before = ready(supervisor, run_id, 'viser')
    view = supervisor.register_view_client(run_id)
    with supervisor._lock:
        entities = supervisor.entities(run_id)
        frozen = supervisor.get_display(run_id)['viser']
        result = supervisor.patch_display(run_id, {'native': {'enabled': True, 'world_id': 1, 'rate_hz': 17.}})
        assert result['viser'] == frozen
        assert supervisor.entities(run_id) == entities
        assert view['view_id'] in supervisor._displays[run_id]._view_ids
    native = ready(supervisor, run_id, 'native')
    assert native['effective'] == {'enabled': True, 'world_id': 1, 'rate_hz': 17.}
    assert native['frame']['viewer'] == 'native'
    assert supervisor.get_display(run_id)['viser']['viewer_url'] == before['viewer_url']
    supervisor.patch_display(run_id, {'viser': {'rate_hz': 7.}})
    ready(supervisor, run_id, 'viser')
    after = supervisor.get_display(run_id)['native']
    assert (after['effective'], after['selection_seq']) == (native['effective'], native['selection_seq'])


def test_native_failure_late_ack_does_not_reopen_or_change_viser(display_run, monkeypatch):
    from muse_server import display
    supervisor, plan = display_run
    run_id = plan.run_id
    spawn = display.subprocess.Popen
    def start(command, **kwargs):
        if command[1:3] == ['-m', 'muse_server.viewers.native_process']:
            command = [sys.executable, '-c', "import sys; print('no display context', file=sys.stderr); sys.exit(1)"]
        return spawn(command, **kwargs)
    monkeypatch.setattr(display.subprocess, 'Popen', start)
    supervisor.patch_display(run_id, {'viser': {'enabled': True}})
    before = ready(supervisor, run_id, 'viser')
    original = supervisor.get_display(run_id)['native']['effective']
    request = supervisor.patch_display(run_id, {'native': {'enabled': True, 'world_id': 1, 'rate_hz': 17.}})
    failed = wait_for(lambda: (value if (value := supervisor.get_display(run_id))['native']['status'] == 'error' else None))
    assert failed['native']['requested'] == original
    session = supervisor._displays[run_id]
    with supervisor._lock:
        session.worker_event('preview_applied', {'viewer': 'native',
            'selection_seq': request['native']['selection_seq'], 'display': request['native']['requested']})
    final = wait_for(lambda: (value if not (value := supervisor.get_display(run_id))['native']['effective']['enabled'] else None))
    assert final['native']['status'] == 'error'
    assert session.slots['native'].process is None
    assert final['viser']['viewer_url'] == before['viewer_url']
    assert supervisor.get_run(run_id)['state'] == 'RUNNING'


def test_native_user_close_disables_only_native(native_transport):
    supervisor, plan = native_transport
    run_id = plan.run_id
    supervisor.patch_display(run_id, {'native': {'enabled': True, 'world_id': 1, 'rate_hz': 17.}})
    ready(supervisor, run_id, 'native')
    first = supervisor._displays[run_id].slots['native'].process
    (Path(plan.artifact_directory) / 'native-close').touch()
    value = wait_for(lambda: (slot if not (slot := supervisor.get_display(run_id)['native'])['requested']['enabled'] else None))
    assert value['requested'] == {'enabled': False, 'world_id': 1, 'rate_hz': 17.}
    ready(supervisor, run_id, 'native')
    wait_for(lambda: first.poll() is not None)
    assert supervisor.get_run(run_id)['state'] == 'RUNNING'
    (Path(plan.artifact_directory) / 'native-close').unlink()
    supervisor.patch_display(run_id, {'native': {'enabled': True}})
    value = ready(supervisor, run_id, 'native')
    assert value['effective']['world_id'] == 1 and value['effective']['rate_hz'] == 17.
    assert supervisor._displays[run_id].slots['native'].process.pid != first.pid


@pytest.mark.parametrize('stopped', [False, True])
def test_terminal_waits_for_native_log_writer_then_indexes_final_bytes(native_transport, stopped):
    supervisor, plan = native_transport
    run_id = plan.run_id
    supervisor.patch_display(run_id, {'native': {'enabled': True}})
    ready(supervisor, run_id, 'native')
    process = supervisor._displays[run_id].slots['native'].process
    if stopped:
        os.kill(process.pid, signal.SIGSTOP)
    supervisor.cancel(run_id)
    terminal = wait_for(lambda: (row if (row := supervisor.get_run(run_id))['state'] == 'CANCELLED' else None), timeout=10)
    assert process.poll() is not None
    assert supervisor._displays[run_id].drained
    artifact, = [item for item in terminal['artifacts'] if item['kind'] == 'native_log']
    raw = (Path(plan.artifact_directory) / artifact['path']).read_bytes()
    assert artifact['sha256'] == hashlib.sha256(raw).hexdigest() and artifact['size'] == len(raw)
    if not stopped:
        assert b'native final bytes' in raw


def test_cancel_during_display_drain_does_not_write_closed_worker(native_transport):
    supervisor, plan = native_transport
    run_id = plan.run_id
    supervisor.patch_display(run_id, {'native': {'enabled': True}})
    ready(supervisor, run_id, 'native')
    process = supervisor._displays[run_id].slots['native'].process
    os.kill(process.pid, signal.SIGSTOP)
    supervisor._active['process'].terminate()
    wait_for(lambda: supervisor._active and supervisor._active.get('finishing'))
    with supervisor._lock:
        record = supervisor.get_run(run_id)
        assert supervisor.cancel(run_id) == record
        with pytest.raises(ValueError, match='ended'):
            supervisor.patch_display(run_id, {'native': {'enabled': True}})
    final = wait_for(lambda: (row if (row := supervisor.get_run(run_id))['state'] == 'FAILED' else None))
    assert not final['cancel_requested'] and process.poll() is not None


def test_native_without_display_preserves_last_successful_settings(display_run, monkeypatch):
    supervisor, plan = display_run
    run_id = plan.run_id
    supervisor.patch_display(run_id, {'native': {'world_id': 1, 'rate_hz': 7.}})
    original = ready(supervisor, run_id, 'native')['effective']
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    supervisor.patch_display(run_id, {'native': {'enabled': True, 'world_id': 0, 'rate_hz': 17.}})
    failed = wait_for(lambda: (slot if (slot := supervisor.get_display(run_id)['native'])['status'] == 'error' else None))
    assert failed['requested'] == original
    assert 'GLFW' in failed['error']
    assert supervisor.get_run(run_id)['state'] == 'RUNNING'


@pytest.mark.skipif(os.environ.get('MUSE_TEST_NATIVE') != '1', reason='Explicit native process window validation')
def test_native_process_renders_then_exits_on_eof(tmp_path):
    import select
    import subprocess
    import mujoco
    from muse_runtime.preview import export_model
    from muse_sdk.run_protocol import PreviewFrame, encode_preview, preview_socketpair

    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/>'
        '<geom type="box" size=".1 .2 .3"/></body></worldbody></mujoco>')
    metadata = export_model(model, tmp_path, objects=[])
    data = mujoco.MjData(model)
    data.qpos[:3] = [1., 2., 3.]
    mujoco.mj_forward(model, data)
    arrays = data.xpos.astype('<f4').tobytes() + data.xmat.astype('<f4').tobytes()
    sender, receiver = preview_socketpair()
    process = None
    with (tmp_path / 'native.log').open('wb') as log:
        try:
            process = subprocess.Popen([sys.executable, '-m', 'muse_server.viewers.native_process',
                '--snapshot-fd', str(receiver.fileno()), '--model', str(tmp_path / metadata.path),
                '--sha256', metadata.sha256, '--nbody', str(model.nbody), '--run-id', 'native-window'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, pass_fds=(receiver.fileno(),))
            receiver.close()
            def message():
                assert select.select([process.stdout], [], [], 15)[0], (tmp_path / 'native.log').read_text()
                value = json.loads(process.stdout.readline())
                assert value['kind'] != 'error', value
                return value
            assert message()['kind'] == 'loaded'
            for seq, world in [(1, 0), (2, 1)]:
                settings = dict(enabled=True, world_id=world, rate_hz=10.)
                state = dict(requested=settings, effective=settings, selection_seq=seq,
                             effective_seq=seq, status='pending', error=None, frame=None)
                process.stdin.write((json.dumps({'kind': 'state', 'state': state}) + '\n').encode())
                process.stdin.flush()
                header = PreviewFrame(viewer='native', run_id='native-window', selection_seq=seq,
                    world_id=world, episode=1, decision=0, tick=0, sim_time=0., scene_hash=metadata.sha256, nbody=model.nbody)
                sender.send(encode_preview(header, arrays))
                ack = message()
                assert ack == {'kind': 'frame', 'header': header.model_dump()}
            process.stdin.close()
            assert process.wait(timeout=10) == 0
        finally:
            sender.close()
            receiver.close()
            if process is not None:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                process.stdout.close()
