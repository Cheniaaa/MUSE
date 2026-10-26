"""仅 selected world 的只读快照；不持有物理循环。"""
import time

from muse_sdk.run_protocol import (DisplaySettings, PreviewSelection, PreviewFrame,
    native_state_layout, encode_preview, send_preview)


class WorkerPreview:
    def __init__(self, run_id, env, sock, emit, *, model):
        import torch
        self.run_id, self.env, self.sock, self.emit = run_id, env, sock, emit
        compiled = env.sim.mj_model
        self.native_layout = native_state_layout(nq=compiled.nq, nv=compiled.nv, na=compiled.na, nu=compiled.nu,
            nbody=compiled.nbody, nmocap=compiled.nmocap, neq=compiled.neq, nsensordata=compiled.nsensordata)
        self.model = model
        self.selections = {name: PreviewSelection(selection_seq=0,
            display=DisplaySettings(enabled=False, world_id=env.experiment.preview.world_ids[0],
                                    rate_hz=env.experiment.preview.rate_hz)) for name in ('viser', 'native')}
        self.episodes = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.pending = dict.fromkeys(self.selections)
        self.next_sample = dict.fromkeys(self.selections, 0.)
        self._first = 'viser'
        self.error = None
        if self.model is not None:
            emit('preview_model', self.model.model_dump())

    def fail(self, message):
        self.error = message
        self.pending = dict.fromkeys(self.selections)
        self.emit('preview_error', {'message': message})

    def after_step(self):
        # 只在每次成功 step 后调用；本步自动 reset 的 world clock 已归零。
        reset = self.env.episode_length_buf == 0
        self.episodes.add_(reset)
        worlds = {selection.display.world_id for selection in self.selections.values()
                  if selection.display.enabled}
        for world in worlds:
            if reset[world].item():
                self._invalidate_world(world)

    def _invalidate_world(self, world_id):
        for name, selection in self.selections.items():
            if selection.display.world_id == world_id:
                self.pending[name] = None
                self.next_sample[name] = 0.

    def after_reset(self, world_id):
        self.episodes[world_id] += 1
        self._invalidate_world(world_id)

    def apply(self, control):
        choices = {name: getattr(control.preview, name) for name in self.selections}
        if any(selection.display.world_id >= self.env.num_envs for selection in choices.values()):
            self.fail('Invalid preview world')
            return
        for name, selection in choices.items():
            if selection.selection_seq == self.selections[name].selection_seq:
                continue
            self.selections[name] = selection
            self.pending[name] = None
            self.next_sample[name] = 0.
            self.emit('preview_applied', {'viewer': name, **selection.model_dump()})
        if self.error:
            self.emit('preview_error', {'message': self.error})

    def sample(self, force=False):
        if self.error or self.model is None:
            return
        now = time.monotonic()
        due = {name: selection for name, selection in self.selections.items()
               if selection.display.enabled and (force or now >= self.next_sample[name])}
        if due:
            import torch
            torch.cuda.current_stream(self.env.device).synchronize()
            for name, selection in due.items():
                world = selection.display.world_id
                fields = ('xpos', 'xmat') if name == 'viser' else (
                    field for field, size in self.native_layout if size)
                arrays = b''.join(getattr(self.env.sim.data, field)[world:world+1]
                    .detach().cpu().numpy().astype('<f4', copy=False).tobytes() for field in fields)
                episode = int(self.episodes[world].item())
                decision = int(self.env.episode_length_buf[world].item())
                tick = decision * self.env.cfg.decimation
                header = PreviewFrame(viewer=name, run_id=self.run_id, world_id=world,
                    selection_seq=selection.selection_seq, episode=episode, decision=decision, tick=tick,
                    sim_time=tick * self.env.physics_dt, scene_hash=self.model.sha256,
                    nbody=self.model.nbody, payload_floats=len(arrays) // 4)
                self.pending[name] = encode_preview(header, arrays)
                self.next_sample[name] = now + 1. / selection.display.rate_hz
        try:
            other = 'native' if self._first == 'viser' else 'viser'
            for name in (self._first, other):
                self.pending[name] = send_preview(self.sock, self.pending[name])
            self._first = other
        except OSError as exc:
            self.fail(f'Preview socket unavailable: {exc}')
