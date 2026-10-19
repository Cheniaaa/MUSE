"""仅 selected world 的只读快照；不持有物理循环。"""
import hashlib
import json
from pathlib import Path
import time

from muse_sdk.run_protocol import (DisplaySettings, PreviewSelection, PreviewFrame, PreviewModel,
    SceneObject, native_state_layout, encode_preview, send_preview)


def scene_objects(experiment, scene, model) -> list[SceneObject]:
    """从已编译 Run 的 spec 引用导出身份；不重新装配或读取设备状态。"""
    import mujoco
    import numpy as np
    from .scene import entity_name

    roles = {(binding.group_id, slot): role for role, binding in experiment.bindings.items()
             for slot in binding.role_slot_to_group_slot}
    objects, entity_geoms = [], set()
    for group in experiment.scenario.entity_groups:
        reference = experiment.entity_models[group.model_id]
        for slot in range(len(experiment.scenario.placements[group.placement_id])):
            name = entity_name(group.group_id, slot)
            # attach 保留元素引用，compile 后 id 属于本次 Scene；包含匿名 geom。
            geom_ids = sorted(geom.id for geom in scene.entities[name].spec.geoms)
            entity_geoms.update(geom_ids)
            objects.append(SceneObject(object_id=name, label=name, kind='entity',
                group_id=group.group_id, group_slot=slot, role_id=roles.get((group.group_id, slot)),
                model_id=reference.resource_id, body_id=model.body(f'{name}/{reference.root_body}').id,
                geom_ids=geom_ids))
    labels = experiment.environment_metadata.get('object_labels', {})
    # 当前 Scene 的基础资源没有 geom，其余未归实体的 geom 均来自冻结环境。
    for geom in scene.spec.geoms:
        if geom.id in entity_geoms:
            continue
        identity = f'environment:{geom.id}'
        name = geom.name.removeprefix('environment/')
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, model.geom_quat[geom.id])
        objects.append(SceneObject(object_id=identity, label=labels.get(name, name or identity),
            kind='environment', group_id=None, group_slot=None, role_id=None, model_id=None,
            body_id=int(model.geom_bodyid[geom.id]), geom_ids=[geom.id],
            local_position=model.geom_pos[geom.id].tolist(), local_orientation_matrix=matrix.tolist()))
    return objects


def export_model(model, directory, *, objects: list[SceneObject]):
    import mujoco
    path = Path(directory) / 'preview.mjb'
    mujoco.mj_saveModel(model, filename=str(path))
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    layout = native_state_layout(nq=model.nq, nv=model.nv, na=model.na, nu=model.nu,
        nbody=model.nbody, nmocap=model.nmocap, neq=model.neq, nsensordata=model.nsensordata)
    metadata = PreviewModel(path=path.name, sha256=digest, nbody=model.nbody,
                           native_payload_floats=sum(size for _, size in layout), objects=objects)
    (Path(directory) / 'preview_objects.json').write_text(json.dumps(
        [obj.model_dump(mode='json') for obj in metadata.objects], ensure_ascii=False, allow_nan=False),
        encoding='utf-8')
    return metadata


class WorkerPreview:
    def __init__(self, run_id, env, directory, sock, emit):
        import torch
        self.run_id, self.env, self.sock, self.emit = run_id, env, sock, emit
        model = env.sim.mj_model
        self.native_layout = native_state_layout(nq=model.nq, nv=model.nv, na=model.na, nu=model.nu,
            nbody=model.nbody, nmocap=model.nmocap, neq=model.neq, nsensordata=model.nsensordata)
        self.model = None
        self.selections = {name: PreviewSelection(selection_seq=0,
            display=DisplaySettings(enabled=False, world_id=env.experiment.preview.world_ids[0],
                                    rate_hz=env.experiment.preview.rate_hz)) for name in ('viser', 'native')}
        self.episodes = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.pending = dict.fromkeys(self.selections)
        self.next_sample = dict.fromkeys(self.selections, 0.)
        self._first = 'viser'
        self.error = None
        # 仅 CPU 模型导出失败可隔离；GPU 访问和同步不在此异常边界内。
        try:
            objects = scene_objects(env.experiment, env.scene, env.sim.mj_model)
            self.model = export_model(env.sim.mj_model, directory, objects=objects)
        except Exception as exc:
            self.fail(str(exc))
        else:
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
