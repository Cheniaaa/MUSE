"""已解析草稿的独立 CPU 编译；不加载 Task、算法或运行期执行状态。"""
import argparse
import json
from pathlib import Path
import tempfile

import mujoco
from mjlab.scene import Scene

from muse_sdk.experiment import ResourceRef
from muse_sdk.scene_edit import EnvironmentEdit, PreviewEntity
from muse_sdk.view_protocol import SceneObject

from .environment_edit import derive_environment, inspect_editable_environment
from .preview import export_model
from .scene import entity_name, make_scene_config


def compile_preview(request: dict, resource_paths: dict, destination: Path) -> dict:
    """destination 为调用方选定的新目录；HTTP 路径及资源身份已由 Server 核对。"""
    entities = [PreviewEntity.model_validate(entity) for entity in request['entities']]
    reference = ResourceRef.model_validate(request['environment'])
    destination = destination.absolute()
    destination.mkdir()
    with tempfile.TemporaryDirectory(prefix='muse-preview-', dir=destination) as temporary:
        environment_root = Path(resource_paths['environment'])
        if request['environment_edit'] is not None:
            derived = Path(temporary) / 'environment'
            derive_environment(environment_root, EnvironmentEdit.model_validate(request['environment_edit']), derived)
            environment_root = derived
        manifest = inspect_editable_environment(environment_root, reference)
        cfg = make_scene_config(environment_root, request['environment_scene'], entities,
                                resource_paths, world_count=1)
        scene = Scene(cfg, device='cpu')
        # 保存实际原生元素引用：attach 不复制元素，compile 后其 id 就是本模型编号。
        native_geoms = list(scene.spec.geoms)
        model = scene.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        objects, entity_geoms = [], set()
        for entity in entities:
            name = entity_name(entity.group_id, entity.slot)
            geom_ids = sorted(geom.id for geom in scene.entities[name].spec.geoms)
            entity_geoms.update(geom_ids)
            objects.append(SceneObject(object_id=entity.object_id, label=entity.object_id, kind='entity',
                group_id=entity.group_id, group_slot=entity.slot, role_id=None,
                model_id=entity.model.resource_id, body_id=model.body(f'{name}/{entity.model.root_body}').id,
                geom_ids=geom_ids))
        # 固定 mjlab 的初始 scene.xml 只有视觉设置；其余 geom 全部来自显式环境。
        environment_geoms = [geom for geom in native_geoms if geom.id not in entity_geoms]
        for obj, geom in zip(manifest.objects, environment_geoms, strict=True):
            objects.append(SceneObject(object_id=obj.object_id, label=obj.label, kind='environment',
                group_id=None, group_slot=None, role_id=None, model_id=None,
                body_id=int(model.geom_bodyid[geom.id]), geom_ids=[geom.id]))
        exported = export_model(model, destination, objects=objects)
    return {'model_path': str(destination / exported.path), 'sha256': exported.sha256,
            'objects': [obj.model_dump(mode='json') for obj in objects],
            'applied_revision': request['revision'],
            'editable': [obj.object_id for obj in manifest.objects if obj.editable] +
                        [entity.object_id for entity in entities],
            'sensor_omissions': [entity.object_id for entity in entities if not entity.sensor_outputs]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--resource-paths', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compile_preview(json.loads(args.request.read_text(encoding='utf-8')),
                             json.loads(args.resource_paths.read_text(encoding='utf-8')), args.destination)
    # JSON 只写独立文件；库的 Python/native 日志不参与结果通道。
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
