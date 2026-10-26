"""Worker拥有的诊断采集；Env安全点调用，不参与物理/学习调度。"""
import time

import numpy as np

from muse_sdk.recording_protocol import RecordingState
from muse_sdk.trajectory import TrajectoryWriter


class RuntimeRecorder:
    def __init__(self, plan, env, model, emit):
        self.plan, self.env, self.emit = plan, env, emit
        self.writer = TrajectoryWriter(plan, model)
        self.worlds = plan.recording.world_ids
        self.episodes = dict.fromkeys(self.worlds, 0)
        self.active = True
        self.statistics = dict(copied_bytes=0, copy_seconds=0., capture_seconds=0., seal_seconds=0.)

    def _state(self, status):
        self.emit('recording_state', RecordingState(status=status, config=self.plan.recording,
            row_bytes=self.writer.row_bytes, rows_per_world=self.writer.rows_per_world,
            static_bytes=self.writer.static_bytes).model_dump(mode='json'))

    def start(self):
        self._state('capturing')
        self._capture(self.worlds, phase=0)

    def before_step(self):
        if self.active and not self.writer.can_append(self.worlds, 2):
            self.finish('capacity')

    def after_step(self, actions, transition):
        if not self.active:
            return
        selected = [world for world in self.worlds if transition.world_reset[world].item()
                    or self.env.episode_length_buf[world].item() % self.plan.recording.every_n_decisions == 0]
        self._capture(selected, phase=1, actions=actions, transition=transition)

    def before_reset(self, world_mask, *, automatic):
        if not self.active or automatic:
            return
        selected = [world for world in self.worlds if world_mask[world].item()]
        if not self.writer.can_append(selected, 2):
            self.finish('capacity')
        else:
            self._capture(selected, phase=2, reset_source=2)

    def after_reset(self, world_mask, *, automatic):
        if not self.active:
            return
        selected = [world for world in self.worlds if world_mask[world].item()]
        for world in selected:
            self.episodes[world] += 1
        self._capture(selected, phase=3, reset_source=1 if automatic else 2)

    def _capture(self, worlds, *, phase, reset_source=0, actions=None, transition=None):
        started = time.perf_counter()
        env = self.env
        for world in worlds:
            values = []
            for field in self.writer.fields:
                name, role = field.name, field.role_id
                if name in ('xpos', 'xmat'):
                    source = getattr(env.sim.data, name)
                elif name in ('velocity_world', 'angular_velocity_body', 'crashed'):
                    source = getattr(env.physics_outputs, name)[role]
                elif name in ('present', 'alive', 'controllable'):
                    source = getattr(env.role_masks[role], name)
                elif phase != 1:
                    values.append(np.zeros(field.shape, dtype=field.dtype))
                    continue
                elif name == 'action_motion':
                    source = actions[role].motion
                elif name in ('action_power_w', 'action_channel'):
                    source = getattr(actions[role].radio, name.removeprefix('action_'))
                elif name == 'reward':
                    source = transition.rewards[role]
                elif name == 'role_cost':
                    source = transition.costs['roles'][role][field.cost_id]
                elif name == 'world_cost':
                    source = transition.costs['world'][field.cost_id]
                elif name == 'learning_done':
                    source = transition.learning_done[role]
                else:
                    source = getattr(transition, name)
                # 同流下先选world，阻塞CPU拷贝完成后才允许Env复用/reset源缓冲。
                before_copy = time.perf_counter()
                value = source[world:world + 1].detach().cpu().numpy().reshape(field.shape)
                self.statistics['copy_seconds'] += time.perf_counter() - before_copy
                self.statistics['copied_bytes'] += value.nbytes
                values.append(value)
            decision = int(env.episode_length_buf[world].item())
            self.writer.append(world, (self.writer.counts[world], env.common_step_counter,
                self.episodes[world], decision, decision * env.cfg.decimation, phase, reset_source), values)
        self.statistics['capture_seconds'] += time.perf_counter() - started

    def finish(self, reason):
        if not self.active:
            return
        self._state('publishing')
        started = time.perf_counter()
        artifact = self.writer.seal(reason, self.env.common_step_counter)
        self.statistics['seal_seconds'] = time.perf_counter() - started
        self.emit('trajectory', artifact.model_dump(mode='json'))
        self.emit('metric', {'source': 'recording', **self.statistics})
        self.active = False
        self.env.recorder = None
        self.writer.close()

    def close(self):
        self.env.recorder = None
        self.writer.close()
