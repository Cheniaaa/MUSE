"""仅 selected world 的只读快照；不持有物理循环。"""
import hashlib
from pathlib import Path
import time

from muse_sdk.run_protocol import (DisplaySettings, PreviewFrame, PreviewModel,
    check_preview_capacity, encode_preview, send_preview)


def export_model(model, directory):
    import mujoco
    path = Path(directory) / 'preview.mjb'
    mujoco.mj_saveModel(model, filename=str(path))
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return PreviewModel(path=path.name, sha256=digest, nbody=model.nbody)


class WorkerPreview:
    def __init__(self, run_id, env, directory, sock, emit):
        import torch
        self.run_id, self.env, self.sock, self.emit = run_id, env, sock, emit
        self.model = None
        self.selection_seq = 0
        self.settings = DisplaySettings(enabled=False, world_id=env.experiment.preview.world_ids[0],
                                        rate_hz=env.experiment.preview.rate_hz)
        self.episodes = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.pending = None
        self.next_sample = 0.
        self.error = None
        # 仅 CPU 模型导出失败可隔离；GPU 访问和同步不在此异常边界内。
        try:
            check_preview_capacity(sock, env.sim.mj_model.nbody)
            self.model = export_model(env.sim.mj_model, directory)
        except Exception as exc:
            self.fail(str(exc))
        else:
            emit('preview_model', self.model.model_dump())

    def fail(self, message):
        self.error = message
        self.pending = None
        self.emit('preview_error', {'message': message})

    def after_step(self):
        # 只在每次成功 step 后调用；本步自动 reset 的 world clock 已归零。
        self.episodes.add_(self.env.episode_length_buf == 0)

    def after_reset(self, world_id):
        self.episodes[world_id] += 1
        self.pending = None
        self.next_sample = 0.

    def apply(self, control):
        if control.display.world_id >= self.env.num_envs:
            self.fail('Invalid preview world')
            return
        self.settings = control.display
        self.selection_seq = control.selection_seq
        self.pending = None
        self.next_sample = 0.
        self.emit('preview_applied', {'selection_seq': self.selection_seq,
                                     'display': self.settings.model_dump()})
        if self.error:
            self.emit('preview_error', {'message': self.error})

    def sample(self, force=False):
        if not self.settings.enabled or self.error or self.model is None:
            return
        now = time.monotonic()
        if force or now >= self.next_sample:
            import torch
            world = self.settings.world_id
            torch.cuda.current_stream(self.env.device).synchronize()
            pos = self.env.sim.data.xpos[world:world+1].detach().cpu().numpy().astype('<f4', copy=False)
            mat = self.env.sim.data.xmat[world:world+1].detach().cpu().numpy().astype('<f4', copy=False)
            decision = int(self.env.episode_length_buf[world].item())
            tick = decision * self.env.cfg.decimation
            header = PreviewFrame(run_id=self.run_id, world_id=world, selection_seq=self.selection_seq,
                episode=int(self.episodes[world].item()), decision=decision, tick=tick,
                sim_time=tick * self.env.physics_dt, scene_hash=self.model.sha256, nbody=self.model.nbody)
            self.pending = encode_preview(header, pos.tobytes() + mat.tobytes())
            self.next_sample = now + 1. / self.settings.rate_hz
        try:
            self.pending = send_preview(self.sock, self.pending)
        except OSError as exc:
            self.fail(f'Preview socket unavailable: {exc}')
