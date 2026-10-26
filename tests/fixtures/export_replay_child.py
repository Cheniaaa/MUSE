"""CPU生命周期fixture：保留真实历史reader/Viser，只提供确定的公开相机。"""
import sys
import threading
import time
from pathlib import Path

from muse_sdk.recording_protocol import ReplayCamera
from muse_server.viewers import scene_interaction

mode = sys.argv.pop(1)


class CameraInteraction(scene_interaction.SceneInteraction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialized = mode != 'unpaired' and not mode.startswith('delayed:')

        def notify():
            if mode.startswith('delayed:'):
                while not Path(mode.removeprefix('delayed:')).exists():
                    time.sleep(.005)
                self.initialized = True
            self.emit({'kind': 'view', 'event': {'kind': 'camera', 'ready': self.initialized}})

        threading.Thread(target=notify, daemon=True).start()

    def camera_ready(self, view_id):
        return self.initialized

    def camera_snapshot(self, view_id):
        if mode == 'snapshot_error':
            raise ValueError('camera disconnected before snapshot')
        return ReplayCamera(position=(3., -4., 2.), wxyz=(1., 0., 0., 0.), fov=.8, near=.01, far=100.)


scene_interaction.SceneInteraction = CameraInteraction
from muse_server.viewers.replay_process import main

raise SystemExit(main())
