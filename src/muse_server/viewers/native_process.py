"""MuJoCo只读刚体快照窗口；相机输入不接入任何物理控制。"""
import glfw
import mujoco

from .scene_data import apply_body_poses, geom_rotations


class NativeWindow:
    def __init__(self, model, *, title: str):
        self.model = model
        self.data = mujoco.MjData(model)
        self._rotations = geom_rotations(model)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.camera)
        self.option = mujoco.MjvOption()
        self.option.flags[:] = 0
        self.option.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 1
        self.option.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
        self.option.geomgroup[:] = 1
        self.option.sitegroup[:] = 0
        self.option.tendongroup[:] = 0
        self.model.light_active[:] = 0
        self.model.vis.headlight.active = 1
        self.scene = mujoco.MjvScene(model, maxgeom=max(model.ngeom, 1))
        self.frame = None
        self._cursor = None
        self.window = self.context = None
        if not glfw.init():
            raise RuntimeError('GLFW initialization failed')
        try:
            self.window = glfw.create_window(960, 640, title, None, None)
            if not self.window:
                raise RuntimeError('GLFW window creation failed')
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)
            self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
            glfw.set_cursor_pos_callback(self.window, self._move)
            glfw.set_scroll_callback(self.window, self._scroll)
        except Exception:
            self.close()
            raise

    def update_frame(self, xpos, xmat, frame: dict) -> None:
        apply_body_poses(self.model, self.data, xpos, xmat, self._rotations)
        if self.frame is None:
            moving = self.model.body_mocapid >= 0
            moving |= self.model.body_dofnum > 0
            centers = self.data.xpos[moving]
            if len(centers):
                self.camera.lookat[:] = centers.mean(axis=0)
        self.frame = frame

    def clear_frame(self) -> None:
        self.frame = None
        self.scene.ngeom = 0

    def _move(self, window, x, y):
        previous, self._cursor = self._cursor, (x, y)
        if previous is None:
            return
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            return
        height = glfw.get_window_size(window)[1]
        if height > 0:
            mujoco.mjv_moveCamera(self.model, action, (x - previous[0]) / height,
                                 (y - previous[1]) / height, self.camera)

    def _scroll(self, window, x, y):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0., -.05 * y, self.camera)

    def render(self) -> bool:
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width > 0 and height > 0:
            if self.frame is not None:
                mujoco.mjv_updateScene(self.model, self.data, self.option, None, self.camera,
                                      mujoco.mjtCatBit.mjCAT_ALL, self.scene)
                if self.scene.status:
                    raise RuntimeError('Native scene geometry capacity exceeded')
            else:
                self.scene.ngeom = 0
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjr_render(viewport, self.scene, self.context)
            text = 'Read-only | Rigid geometry only; auxiliary layers unavailable'
            if self.frame is not None:
                frame = self.frame
                text += (f"\nWorld {frame['world_id']} / Episode {frame['episode']} / "
                         f"Tick {frame['tick']} / Time {frame['sim_time']:.3f}")
            else:
                text += '\nWaiting for current selection frame'
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               viewport, text, '', self.context)
            glfw.swap_buffers(self.window)
        glfw.poll_events()
        return width > 0 and height > 0

    def is_running(self) -> bool:
        return self.window is not None and not glfw.window_should_close(self.window)

    def close(self) -> None:
        if self.context is not None:
            glfw.make_context_current(self.window)
            self.context.free()
            self.context = None
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()


def main() -> int:
    import argparse
    import json
    import os
    from pathlib import Path
    import socket
    import sys
    import time
    import traceback

    import numpy as np
    from muse_sdk.run_protocol import PreviewSelection, drain_preview
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
    sock = socket.socket(fileno=args.snapshot_fd)
    sock.setblocking(False)
    try:
        model = load_model(args.model, args.sha256, args.nbody)
        window = NativeWindow(model, title=f'MUSE Native | {args.run_id} | Read-only')
        emit({'kind': 'loaded'})
        os.set_blocking(0, False)
        incoming = b''
        state = cursor = presented = None
        while window.is_running():
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
                        window.clear_frame()
                        cursor = presented = None
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
                        values = np.frombuffer(arrays, dtype='<f4')
                        if not np.isfinite(values).all():
                            raise ValueError('Nonfinite preview pose')
                        window.update_frame(values[:model.nbody * 3].reshape(model.nbody, 3),
                            values[model.nbody * 3:].reshape(model.nbody, 9), header.model_dump())
                        cursor, presented = key, header
            if window.render() and presented is not None:
                emit({'kind': 'frame', 'header': presented.model_dump()})
                presented = None
            time.sleep(.01)
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
