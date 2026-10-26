"""Run 静态身份与同帧投影 oracle；只编译 CPU Scene，不创建 Simulation。"""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from muse_sdk.experiment import EntityModelRef, Pose


@pytest.fixture
def compiled_run(tmp_path):
    from mjlab.scene import Scene
    from muse_runtime.scene import assemble_scene

    environment = tmp_path / 'environment'
    environment.mkdir()
    (environment / 'scene.xml').write_text('''<mujoco><worldbody>
      <geom name="wall" type="box" size=".1 .2 .3" pos="2 3 4" quat=".5 .5 .5 .5"/>
      <geom type="sphere" size=".2" pos="-3 1 2"/>
      <body name="nested" mocap="true" pos="5 -2 1" quat=".7071067811865476 0 0 .7071067811865476">
        <geom type="box" size=".1 .2 .3" pos=".4 .5 .6" quat=".5 .5 .5 .5"/>
      </body>
    </worldbody></mujoco>''')
    models, paths = {}, {'environment': str(environment)}
    for name, joint in [('free', '<freejoint/>'), ('fixed', '')]:
        root = tmp_path / name
        root.mkdir()
        (root / 'model.xml').write_text(f'''<mujoco><worldbody><body name="root">{joint}
          <geom name="shell" type="box" size=".2 .3 .4"/>
          <geom type="sphere" size=".1" pos=".3 0 0"/>
          <body pos="0 .3 0"><geom size=".1"/></body>
        </body></worldbody></mujoco>''')
        models[name] = EntityModelRef(resource_id=name, revision=1, sha256='a' * 64,
            physics_asset='model.xml', root_body='root', support_geoms=[])
        paths['entity_model:' + name] = str(root)
    groups = [SimpleNamespace(group_id='movers', model_id='free', placement_id='moving'),
              SimpleNamespace(group_id='anchor', model_id='fixed', placement_id='fixed')]
    experiment = SimpleNamespace(entity_models=models, environment_scene='scene.xml',
        environment_metadata={'object_labels': {'wall': '墙'}}, sensors={}, activation={},
        scenario=SimpleNamespace(entity_groups=groups, placements={
            'moving': [Pose(position=[1., 2., 3.], orientation_wxyz=[1., 0., 0., 0.]),
                       Pose(position=[-2., 4., 1.], orientation_wxyz=[.5, .5, .5, .5])],
            'fixed': [Pose(position=[7., 1., 2.], orientation_wxyz=[0., 0., 0., 1.])]},
            scene=SimpleNamespace(physics_dt=.003, integrator='RK4', density=1.2, viscosity=.000018)),
        topology=SimpleNamespace(world_count=2), preview=SimpleNamespace(world_ids=[0], rate_hz=10.),
        bindings={'pilot_b': SimpleNamespace(group_id='movers', role_slot_to_group_slot=[1]),
                  'pilot_a': SimpleNamespace(group_id='movers', role_slot_to_group_slot=[0])})
    cfg, _ = assemble_scene(experiment, paths)
    scene = Scene(cfg, device='cpu')
    model = scene.compile()
    return experiment, scene, model


def test_run_objects_cover_actual_geoms_and_frozen_group_role_identity(compiled_run):
    from muse_runtime.scene_snapshot import scene_objects

    experiment, scene, model = compiled_run
    objects = scene_objects(experiment, scene, model)
    assert len({obj.object_id for obj in objects}) == len(objects)
    ids = [geom for obj in objects for geom in obj.geom_ids]
    assert len(ids) == len(set(ids)) == model.ngeom
    entities = {obj.object_id: obj for obj in objects if obj.kind == 'entity'}
    assert set(entities) == {'movers_0', 'movers_1', 'anchor_0'}
    for slot, role in [(0, 'pilot_a'), (1, 'pilot_b')]:
        obj = entities[f'movers_{slot}']
        assert (obj.group_id, obj.group_slot, obj.role_id, obj.model_id) == ('movers', slot, role, 'free')
        assert obj.body_id == model.body(f'movers_{slot}/root').id
        assert obj.geom_ids == sorted(geom.id for geom in scene.entities[obj.object_id].spec.geoms)
        assert len(obj.geom_ids) == 3  # 匿名 geom 和匿名子 body 不依赖 name 前缀。
        assert obj.local_position is None and obj.local_orientation_matrix is None
    anchor = entities['anchor_0']
    assert anchor.role_id is None and anchor.model_id == 'fixed'
    assert anchor.body_id == model.body('anchor_0/root').id
    environment = [obj for obj in objects if obj.kind == 'environment']
    assert len(environment) == 3
    assert all(obj.object_id == f'environment:{obj.geom_ids[0]}' for obj in environment)
    assert next(obj for obj in environment if obj.geom_ids == [model.geom('environment/wall').id]).label == '墙'
    assert all(obj.local_position is not None and obj.local_orientation_matrix is not None for obj in environment)
    assert [obj.model_dump() for obj in scene_objects(experiment, scene, model)] == [obj.model_dump() for obj in objects]


def test_local_geom_transform_projects_two_body_frames_without_false_world_zero(compiled_run):
    from muse_runtime.scene_snapshot import scene_objects

    experiment, scene, model = compiled_run
    objects = scene_objects(experiment, scene, model)
    data = mujoco.MjData(model)
    for position, quat in [([1., 2., 3.], [1., 0., 0., 0.]),
                           ([-4., 3., 2.], [.5, .5, .5, .5])]:
        root = model.body('movers_0/root').id
        address = model.jnt_qposadr[model.body_jntadr[root]]
        data.qpos[address:address+7] = [*position, *quat]
        mocap = model.body_mocapid[model.body('environment/nested').id]
        data.mocap_pos[mocap] = position
        data.mocap_quat[mocap] = quat
        mujoco.mj_forward(model, data)
        for obj in objects:
            body = obj.body_id
            if obj.kind == 'entity':
                if obj.object_id == 'movers_0':
                    np.testing.assert_allclose(data.xpos[body], position)
                continue
            geom, = obj.geom_ids
            rotation = data.xmat[body].reshape(3, 3)
            projected_position = data.xpos[body] + rotation @ obj.local_position
            projected_matrix = rotation @ np.array(obj.local_orientation_matrix).reshape(3, 3)
            np.testing.assert_allclose(projected_position, data.geom_xpos[geom], atol=1e-12)
            np.testing.assert_allclose(projected_matrix.ravel(), data.geom_xmat[geom], atol=1e-12)
            assert not np.allclose(projected_position, np.zeros(3))
        wall = next(obj for obj in objects if obj.geom_ids == [model.geom('environment/wall').id])
        assert wall.body_id == 0
        np.testing.assert_allclose(wall.local_position, [2., 3., 4.])


def test_worker_exports_real_objects_once_and_sidecar_survives_without_runtime(compiled_run, tmp_path, monkeypatch):
    import torch
    from muse_runtime.preview import WorkerPreview
    from preview_helpers import preview_model
    from muse_sdk.run_protocol import preview_socketpair

    experiment, scene, model = compiled_run
    def no_compile():
        pytest.fail('导出不得重新编译 Scene')
    def no_device_copy(*args, **kwargs):
        pytest.fail('静态对象导出不得读取 GPU stream 或复制 Tensor 到 CPU')
    monkeypatch.setattr(scene, 'compile', no_compile)
    monkeypatch.setattr(torch.cuda, 'current_stream', no_device_copy)
    monkeypatch.setattr(torch.Tensor, 'cpu', no_device_copy)
    env = SimpleNamespace(experiment=experiment, scene=scene, sim=SimpleNamespace(mj_model=model),
                          num_envs=2, device='cpu')
    events = []
    sender, receiver = preview_socketpair()
    try:
        preview = WorkerPreview('run', env, sender, lambda kind, payload: events.append((kind, payload)),
                                model=preview_model(env, tmp_path))
    finally:
        sender.close()
        receiver.close()
    assert preview.error is None
    assert preview.model.objects
    assert [kind for kind, _ in events] == ['preview_model']
    metadata = events[0][1]
    sidecar = tmp_path / 'preview_objects.json'
    assert json.loads(sidecar.read_text()) == metadata['objects']
    assert Path(tmp_path / metadata['path']).is_file()
    script = '''import json, sys
from pathlib import Path
from muse_sdk.view_protocol import SceneObject
objects = [SceneObject.model_validate(obj) for obj in json.loads(Path(sys.argv[1]).read_text())]
assert objects and any(obj.kind == 'environment' for obj in objects)
assert not {'mujoco', 'numpy', 'torch', 'warp'} & sys.modules.keys()
assert not any(name.startswith('muse_runtime') for name in sys.modules)
print('STATIC_OBJECTS_OK')
'''
    result = subprocess.run([sys.executable, '-c', script, str(sidecar)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'STATIC_OBJECTS_OK' in result.stdout


def test_real_frozen_plan_objects_from_compiled_scene(pipeline):
    from mjlab.scene import Scene
    from muse_runtime.scene_snapshot import scene_objects
    from muse_runtime.scene import assemble_scene
    from muse_server.resolver import resolve_experiment

    catalog, source = pipeline
    report = resolve_experiment(source, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment, source=source)
    plan = catalog.create_run_plan(identity, mode='EVALUATE', device='cpu', max_decision_steps=1)
    cfg, _ = assemble_scene(plan.experiment, plan.resource_paths)
    scene = Scene(cfg, device='cpu')
    model = scene.compile()
    objects = scene_objects(plan.experiment, scene, model)
    entity = next(obj for obj in objects if obj.object_id == 'scout_group_0')
    binding = next(binding for binding in plan.experiment.bindings.values() if binding.group_id == entity.group_id)
    assert entity.role_id == binding.role_id
    assert entity.model_id == plan.experiment.entity_models[binding.model_id].resource_id
    assert entity.body_id == model.body('scout_group_0/cf2').id
    assert len({geom for obj in objects for geom in obj.geom_ids}) == model.ngeom


def test_preview_model_import_orders_remain_valid():
    for modules in [('experiment', 'run_protocol', 'view_protocol'),
                    ('run_protocol', 'view_protocol', 'experiment'),
                    ('view_protocol', 'experiment', 'run_protocol')]:
        script = '\n'.join(f'import muse_sdk.{module}' for module in modules)
        script += '\nfrom muse_sdk.run_protocol import PreviewModel\n'
        script += "PreviewModel(path='preview.mjb', sha256='a'*64, nbody=1, native_payload_floats=6, objects=[])\n"
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
