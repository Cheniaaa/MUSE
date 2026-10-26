"""编译冷模型与对象身份一次导出，供实时preview和独立recording共用。"""
import hashlib
import json
from pathlib import Path

from muse_sdk.run_protocol import PreviewModel, SceneObject, native_state_layout


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
