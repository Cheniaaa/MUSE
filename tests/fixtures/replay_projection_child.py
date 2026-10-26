"""测试child：在真实公开投影之后用公开pick暴露CPU几何oracle。"""
from muse_server.viewers import scene_interaction


class ProjectedInteraction(scene_interaction.SceneInteraction):
    def update_frame(self, xpos, xmat, frame):
        super().update_frame(xpos, xmat, frame)
        hit = self.pick([-2., 0., 1.], [1., 0., 0.])
        self.emit({'kind': 'view', 'event': {'kind': 'projection', 'hit': hit}})


scene_interaction.SceneInteraction = ProjectedInteraction

from muse_server.viewers.replay_process import main

raise SystemExit(main())
