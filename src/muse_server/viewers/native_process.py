"""官方MuJoCo窗口进程；只消费快照和显示选择，不控制Run物理。"""


def main() -> int:
    import argparse
    import json
    import os
    from pathlib import Path
    import signal
    import socket
    import sys
    import time
    import traceback

    from muse_sdk.run_protocol import PreviewSelection, drain_preview
    from .native_viewer import NativeSnapshot, OfficialNativeViewer
    from .scene_data import load_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-fd', type=int, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--nbody', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    sys.stdout.flush()
    protocol_fd = os.dup(1)
    os.dup2(2, 1)

    def emit(message):
        data = (json.dumps(message, allow_nan=False) + '\n').encode()
        while data:
            data = data[os.write(protocol_fd, data):]

    window = None
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    sock = socket.socket(fileno=args.snapshot_fd)
    sock.setblocking(False)
    try:
        model = load_model(args.model, args.sha256, args.nbody)
        window = OfficialNativeViewer(NativeSnapshot(model), run_id=args.run_id)
        emit({'kind': 'loaded'})
        os.set_blocking(0, False)
        incoming = b''
        state = cursor = pending = None
        next_sync = 0.
        while not stopping and window.is_running():
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
                    updated = json.loads(line)['state']
                    if state is None or (updated['selection_seq'], updated['effective_seq']) != (
                            state['selection_seq'], state['effective_seq']):
                        cursor = pending = None
                    if state is None or updated['selection_seq'] != state['selection_seq']:
                        window.set_selection(PreviewSelection(selection_seq=updated['selection_seq'],
                                                              display=updated['requested']))
                        next_sync = 0.
                    state = updated
            if state is not None:
                selection = PreviewSelection(selection_seq=state['effective_seq'], display=state['effective'])
                latest = drain_preview(sock, run_id=args.run_id, selections={'native': selection}, scene_hash=args.sha256)
                if 'native' in latest and state['selection_seq'] == state['effective_seq']:
                    header, arrays = latest['native']
                    key = header.episode, header.decision
                    if cursor is None or key > cursor:
                        if header.nbody != model.nbody:
                            raise ValueError('Preview frame nbody mismatch')
                        cursor, pending = key, (header, arrays)
                if time.monotonic() >= next_sync:
                    if pending is not None:
                        window.update_frame(*pending)
                        pending = None
                    synchronized = window.sync()
                    if synchronized is not None:
                        emit({'kind': 'frame', 'header': synchronized.model_dump()})
                    next_sync = time.monotonic() + 1. / state['requested']['rate_hz']
            time.sleep(.01)
        if not stopping:
            emit({'kind': 'closed'})
        return 0
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit({'kind': 'error', 'message': str(exc)})
        return 1
    finally:
        sock.close()
        if window is not None:
            window.close()
        os.close(protocol_fd)


if __name__ == '__main__':
    raise SystemExit(main())
