"""Viser 消费快照并转发控制意图；不导入 Runtime，也不推进物理。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time

from muse_sdk.run_protocol import drain_preview


def load_model(path, sha256, nbody):
    import mujoco
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != sha256:
        raise ValueError('Preview model hash mismatch')
    model = mujoco.MjModel.from_binary_path(str(path))
    if model.nbody != nbody:
        raise ValueError('Preview model nbody mismatch')
    return model


def create_interaction_controls(gui, world, emit, *, interactive):
    if not interactive:
        return lambda message: None
    status = gui.add_text('Interaction', initial_value='running', disabled=True)
    buttons = {}
    interaction = {'command': None, 'result': None}
    awaiting_after = None

    def add_button(label, command):
        button = gui.add_button(label, disabled=command != 'pause')
        buttons[command] = button

        @button.on_click
        def clicked(event):
            nonlocal awaiting_after
            if event.client_id is None or button.disabled:
                return
            previous = interaction['command']
            awaiting_after = previous['command_seq'] if previous else 0
            for handle in buttons.values():
                handle.disabled = True
            status.value = f'{command}: pending'
            emit({'kind': 'command', 'command': command,
                  'world_id': int(world.value) if command == 'reset_world' else None})

    for label, command in (('Pause', 'pause'), ('Resume', 'resume'), ('Step', 'step'), ('Reset world', 'reset_world')):
        add_button(label, command)

    def update(message):
        nonlocal interaction, awaiting_after
        if message['kind'] == 'state':
            interaction = message['state']['interaction']
        elif message['kind'] != 'command_error':
            return
        else:
            awaiting_after = None
        command, result = interaction['command'], interaction['result']
        if awaiting_after is not None:
            if command is None or command['command_seq'] <= awaiting_after:
                return
            awaiting_after = None
        pending = command is not None and result is None
        paused = result is not None and result['paused']
        for name, button in buttons.items():
            button.disabled = pending or (paused if name == 'pause' else not paused)
        if message['kind'] == 'command_error':
            status.value = message['message']
        elif pending:
            status.value = f"#{command['command_seq']} {command['command']}: pending"
        elif result is not None:
            status.value = (f"#{result['command_seq']} {result['command']}: {result['status']} / "
                            f"{'paused' if paused else 'running'} / steps {result['steps']} / updates {result['updates']}"
                            + (f" / {result['error']}" if result['error'] else ''))
        else:
            status.value = 'running'
    return update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-fd', type=int, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--objects', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--nbody', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--world-count', type=int, required=True)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--mode', choices=('TRAIN', 'EVALUATE', 'INTERACTIVE'), default='EVALUATE')
    args = parser.parse_args()
    sys.stdout.flush()
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    lock = threading.Lock()

    def emit(message):
        data = (json.dumps(message, allow_nan=False) + '\n').encode()
        with lock:
            while data:
                data = data[os.write(protocol_fd, data):]

    server = None
    interaction = None
    sock = socket.socket(fileno=args.snapshot_fd)
    sock.setblocking(False)
    try:
        import numpy as np
        import viser
        from muse_sdk.view_protocol import SceneObject, ViewIntent
        from .scene_interaction import SceneInteraction
        model = load_model(args.model, args.sha256, args.nbody)
        objects = [SceneObject.model_validate(obj) for obj in json.loads(args.objects.read_text(encoding='utf-8'))]
        server = viser.ViserServer(host='127.0.0.1', port=args.port, verbose=False)
        interaction = SceneInteraction(server, model, objects, view_id=None, emit=emit)
        world = server.gui.add_dropdown('World', options=tuple(str(i) for i in range(args.world_count)), initial_value='0')
        rate = server.gui.add_slider('Rate (Hz)', min=1., max=30., step=1., initial_value=10.)
        status = server.gui.add_text('Display', initial_value='pending', disabled=True)
        clock = server.gui.add_text('Frame', initial_value='', disabled=True)
        update_interaction = create_interaction_controls(server.gui, world, emit, interactive=args.mode == 'INTERACTIVE')

        @world.on_update
        def change_world(event):
            if event.client_id is not None:
                emit({'kind': 'patch', 'patch': {'world_id': int(world.value)}})

        @rate.on_update
        def change_rate(event):
            if event.client_id is not None:
                emit({'kind': 'patch', 'patch': {'rate_hz': float(rate.value)}})

        emit({'kind': 'loaded', 'host': server.get_host(), 'port': server.get_port()})
        os.set_blocking(0, False)
        incoming = b''
        state = None
        view_ids = set()
        cursor = None
        while True:
            while True:
                try:
                    chunk = os.read(0, 65536)
                except BlockingIOError:
                    break
                if not chunk:
                    return 0
                incoming += chunk
                while b'\n' in incoming:
                    line, incoming = incoming.split(b'\n', 1)
                    message = json.loads(line)
                    if message['kind'] == 'view_intent':
                        target = message['view_id']
                        try:
                            interaction.apply_intent(ViewIntent.model_validate(message['intent']), view_id=target)
                        except ValueError as exc:
                            emit({'kind': 'view', 'view_id': target if target in view_ids else None,
                                  'event': {'kind': 'error', 'message': str(exc)}})
                        continue
                    update_interaction(message)
                    if message['kind'] == 'command_error':
                        continue
                    if message['kind'] == 'request_error':
                        status.value = message['message']
                        continue
                    view_ids = set(message['view_ids'])
                    interaction.set_view_ids(message['view_ids'])
                    updated = message['state']
                    if state is None or updated['effective_seq'] != state['effective_seq']:
                        cursor = None
                    state = updated
                    world.value = str(state['effective']['world_id'])
                    rate.value = state['effective']['rate_hz']
                    status.value = state['error'] or state['status']
            if state is not None:
                latest = drain_preview(sock, run_id=args.run_id, selection_seq=state['effective_seq'],
                    world_id=state['effective']['world_id'], scene_hash=args.sha256)
                if latest is not None and state['effective']['enabled']:
                    header, arrays = latest
                    key = header.episode, header.decision
                    if cursor is None or key > cursor:
                        if header.nbody != model.nbody:
                            raise ValueError('Preview frame nbody mismatch')
                        values = np.frombuffer(arrays, dtype='<f4')
                        if not np.isfinite(values).all():
                            raise ValueError('Nonfinite preview pose')
                        xpos = values[:model.nbody * 3].reshape(1, model.nbody, 3)
                        xmat = values[model.nbody * 3:].reshape(1, model.nbody, 3, 3)
                        interaction.update_frame(xpos[0], xmat[0], header.model_dump(mode='json'))
                        cursor = key
                        clock.value = f'World {header.world_id} / Episode {header.episode} / Tick {header.tick}'
                        emit({'kind': 'frame', 'header': header.model_dump()})
            time.sleep(.01)
    except Exception as exc:
        emit({'kind': 'error', 'message': str(exc)})
        return 1
    finally:
        sock.close()
        if interaction is not None:
            interaction.close()
        if server is not None:
            server.stop()
        os.close(protocol_fd)


if __name__ == '__main__':
    raise SystemExit(main())
