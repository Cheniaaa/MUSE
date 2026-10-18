"""CPU 原生环境编辑；输入为 Catalog 已校验的不可变资源目录。"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import mujoco
import yaml

from muse_sdk.config import ConfigError
from muse_sdk.experiment import EnvironmentBundle, Pose, ResourceRef
from muse_sdk.scene_edit import BoxEdit, EnvironmentEdit, EnvironmentManifest, EnvironmentObject


def _referenced_geoms(model) -> dict[int, str]:
    """仅检查固定 MuJoCo 的原生 geom 引用数组，不解释任意插件字符串。"""
    references = {}
    for ids in (model.pair_geom1, model.pair_geom2):
        for geom_id in ids:
            references[int(geom_id)] = 'contact pair'
    for types, ids, reason in (
        (model.eq_objtype, model.eq_obj1id, 'equality'),
        (model.eq_objtype, model.eq_obj2id, 'equality'),
        (model.sensor_objtype, model.sensor_objid, 'sensor object'),
        (model.sensor_reftype, model.sensor_refid, 'sensor reference'),
        (model.tuple_objtype, model.tuple_objid, 'custom tuple'),
    ):
        for kind, geom_id in zip(types, ids):
            if kind == mujoco.mjtObj.mjOBJ_GEOM:
                references[int(geom_id)] = reason
    for kind, geom_id in zip(model.wrap_type, model.wrap_objid):
        if kind in (mujoco.mjtWrap.mjWRAP_SPHERE, mujoco.mjtWrap.mjWRAP_CYLINDER):
            references[int(geom_id)] = 'tendon wrap'
    return references


def _inspect(root: Path, reference: ResourceRef):
    document = yaml.safe_load((root / 'environment.yaml').read_text(encoding='utf-8'))
    bundle = EnvironmentBundle.model_validate(document)
    spec = mujoco.MjSpec.from_file(str(root / bundle.scene))
    # fusestatic 编译会改写 spec 树；识别边界必须保留作者原始层级。
    model = spec.copy().compile()
    references = _referenced_geoms(model)
    labels = bundle.metadata.get('object_labels', {})
    names = {geom.name for geom in spec.geoms if geom.name}
    objects, mapping = [], {}
    for index, geom in enumerate(spec.geoms):
        object_id = geom.name
        reason = None
        if not object_id:
            # 匿名对象只有冻结资源内的只读身份；绝不允许此编号成为新 geom 别名。
            object_id = f'@unnamed/{reference.sha256}/{index}'
            while object_id in names:
                object_id = '@' + object_id
            reason = 'readonly: Unnamed geom has no editable identity'
        elif geom.type != mujoco.mjtGeom.mjGEOM_BOX:
            reason = 'readonly: Only box geoms are editable'
        elif geom.parent.name != spec.worldbody.name:
            reason = 'readonly: Geom must be directly in worldbody'
        elif geom.frame is not None:
            reason = 'readonly: Geom uses a native frame, not direct worldbody coordinates'
        elif (geom.contype, geom.conaffinity) not in ((0, 0), (1, 1)):
            reason = 'readonly: Custom collision mask cannot be expressed as a boolean'
        elif model.geom(geom.name).id in references:
            reason = f'readonly: Native {references[model.geom(geom.name).id]} reference'
        label = labels.get(object_id, geom.name or 'Unnamed geom')
        box = None
        if reason is None:
            native = model.geom(geom.name)
            box = BoxEdit(label=label,
                pose=Pose(position=native.pos.tolist(), orientation_wxyz=native.quat.tolist()),
                size_m=(native.size * 2).tolist(), collision=bool(geom.contype),
                rf_blocking=bundle.rf_geometry is not None and geom.name in bundle.rf_geometry.rf_blocking)
        objects.append(EnvironmentObject(object_id=object_id, label=label, editable=reason is None,
                                         reason=reason, box=box))
        mapping[object_id] = geom
    return document, bundle, spec, EnvironmentManifest(base=reference, objects=objects), mapping


def inspect_editable_environment(root: Path, reference: ResourceRef) -> EnvironmentManifest:
    """具名 object_id 就是原生 geom 名称；标签不参与 geom 寻址。"""
    return _inspect(root, reference)[3]


def derive_environment(root: Path, edit: EnvironmentEdit, destination: Path) -> None:
    """完整副本编译成功后才发布到新目录；不改原包，不覆盖既有目标。"""
    if destination.exists():
        raise FileExistsError(destination)
    if destination.resolve().is_relative_to(root.resolve()):
        raise ConfigError('destination', 'Destination must be outside the immutable base')
    with tempfile.TemporaryDirectory(prefix='muse-edit-', dir=destination.parent) as temporary:
        work = Path(temporary) / 'content'
        shutil.copytree(root, work)
        document, bundle, spec, manifest, mapping = _inspect(work, edit.base)
        objects = {obj.object_id: obj for obj in manifest.objects}
        for object_id in (*edit.upserts, *edit.removed):
            if object_id in objects and not objects[object_id].editable:
                raise ConfigError(f'objects.{object_id}', objects[object_id].reason)
        for object_id in edit.removed:
            if object_id not in objects:
                raise ConfigError(f'removed.{object_id}', 'Unknown object')
        blockers = set(bundle.rf_geometry.rf_blocking) if bundle.rf_geometry is not None else set()
        labels = dict(bundle.metadata.get('object_labels', {}))
        for object_id in edit.removed:
            spec.delete(mapping[object_id])
            blockers.discard(object_id)
            labels.pop(object_id, None)
        for object_id, box in edit.upserts.items():
            geom = mapping[object_id] if object_id in mapping else spec.worldbody.add_geom(
                name=object_id, type=mujoco.mjtGeom.mjGEOM_BOX)
            geom.pos = box.pose.position
            geom.quat = box.pose.orientation_wxyz
            # 显式切换原生姿态输入；旧 euler/fromto 不能覆盖新的 world pose。
            geom.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
            geom.fromto[:] = float('nan')
            geom.size = [value / 2 for value in box.size_m]
            geom.contype = geom.conaffinity = int(box.collision)
            labels[object_id] = box.label
            if box.rf_blocking:
                blockers.add(object_id)
            else:
                blockers.discard(object_id)
        spec.compile()
        (work / bundle.scene).write_text(spec.to_xml(), encoding='utf-8')
        # 只新增编辑器确实消费的标签；保留源包其它 metadata 和资源文件。
        document.setdefault('metadata', {})['object_labels'] = labels
        document['rf_geometry'] = {'rf_blocking': sorted(blockers)}
        (work / 'environment.yaml').write_text(yaml.safe_dump(document, allow_unicode=True), encoding='utf-8')
        mujoco.MjModel.from_xml_path(str(work / bundle.scene))
        work.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    for operation in ('inspect', 'derive'):
        command = commands.add_parser(operation)
        command.add_argument('--root', type=Path, required=True)
        command.add_argument('--input', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        if operation == 'derive':
            command.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    source = args.input.read_text(encoding='utf-8')
    if args.operation == 'inspect':
        reference = ResourceRef.model_validate_json(source)
        output = inspect_editable_environment(args.root, reference).model_dump_json()
    else:
        edit = EnvironmentEdit.model_validate_json(source)
        derive_environment(args.root, edit, args.destination)
        output = json.dumps({'destination': str(args.destination)})
    args.output.write_text(output, encoding='utf-8')


if __name__ == '__main__':
    main()
