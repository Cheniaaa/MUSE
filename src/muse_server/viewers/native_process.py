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

    def render(self) -> None:
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
