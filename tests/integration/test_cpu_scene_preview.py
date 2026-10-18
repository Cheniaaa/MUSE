"""不需要研究配置的真实 CPU 场景编译；不创建 Run 或 GPU context。"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from muse_sdk.content import hash_package
from muse_sdk.experiment import EntityModelRef, ResourceRef


@pytest.fixture(scope='module')
def preview_inputs(tmp_path_factory):
    from muse_builtins.metadata import export_metadata

    root = tmp_path_factory.mktemp('cpu-preview-inputs')
    project = Path(__file__).resolve().parents[2]
    builtins = export_metadata(project / 'examples/basic_flight/controller.json')
    environment = root / 'environment'
    environment.mkdir()
    (environment / 'environment.yaml').write_text('''scene: scene.xml
rf_geometry: {rf_blocking: [wall]}
metadata: {object_labels: {wall: 原墙}}
''')
    (environment / 'scene.xml').write_text('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="10 10 .1"/>
      <geom name="wall" type="box" size=".1 1 1" pos="2 0 1"/>
      <geom type="box" size=".2 .2 .2" pos="0 4 1"/>
      <body name="nested" pos="0 7 1"><geom size=".15" pos="0 .7 0"/></body>
    </worldbody></mujoco>''')
    reference = ResourceRef(resource_id='environment', revision=1, sha256=hash_package(environment))
    paths, refs = {'environment': str(environment)}, {}
    for model_id in ('bitcraze_crazyflie_2', 'radio_base_station'):
        record = builtins.entity_models[model_id]
        path = root / model_id
        shutil.copytree(record.path, path)
        paths['entity_model:' + model_id] = str(path)
        refs[model_id] = EntityModelRef(resource_id=model_id, revision=record.revision,
            sha256=record.sha256, **{key: record.metadata[key]
                for key in ('physics_asset', 'root_body', 'support_geoms', 'radio_devices')})
    q = 2 ** -.5
    entities = [dict(object_id='drone', group_id='scout_group', slot=0,
        model=refs['bitcraze_crazyflie_2'].model_dump(mode='json'),
        pose=dict(position=[1.2, -.8, 1.7], orientation_wxyz=[.5, .5, .5, .5]), sensor_outputs=[])]
    for slot, position, orientation in [(0, [0., 1., 1.], [q, 0., 0., q]),
                                        (1, [3., 2., 1.5], [0., 0., 0., 1.])]:
        entities.append(dict(object_id=f'anchor-{slot}', group_id='anchor_group', slot=slot,
            model=refs['radio_base_station'].model_dump(mode='json'),
            pose=dict(position=position, orientation_wxyz=orientation), sensor_outputs=[]))
    request = dict(draft_id='draft-without-research', revision=7, environment=reference.model_dump(mode='json'),
                   environment_scene='scene.xml', environment_edit=None, entities=entities)
    return request, paths, builtins.sensors['cf2_imu_rangefinder_down']['outputs']['range']


def test_incomplete_research_can_preview_geometry(preview_inputs, tmp_path):
    from muse_runtime.scene_preview import compile_preview
    from muse_sdk.view_protocol import SceneObject

    request, paths, _ = preview_inputs
    before = copy.deepcopy(request)
    hashes = {key: hash_package(Path(path)) for key, path in paths.items()}
    result = compile_preview(request, paths, tmp_path / 'compiled')
    assert request == before
    assert result['applied_revision'] == 7
    path = Path(result['model_path'])
    assert path.is_absolute() and path.is_file() and path.suffix == '.mjb'
    assert result['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    model = mujoco.MjModel.from_binary_path(str(path))
    assert model.geom('environment/wall').size.tolist() == [.1, 1., 1.]
    objects = [SceneObject.model_validate(obj) for obj in result['objects']]
    assert {obj.object_id for obj in objects if obj.kind == 'entity'} == {'drone', 'anchor-0', 'anchor-1'}
    assert all(obj.role_id is None for obj in objects)
    assert {key: hash_package(Path(path)) for key, path in paths.items()} == hashes


@pytest.mark.parametrize('with_sensor', [False, True])
def test_fixed_and_free_root_placements_match_assemble_scene(preview_inputs, tmp_path, with_sensor):
    from mjlab.scene import Scene
    from muse_runtime.scene import assemble_scene
    from muse_runtime.scene_preview import compile_preview
    from muse_sdk.experiment import Pose, Topology

    source, paths, range_output = preview_inputs
    request = copy.deepcopy(source)
    if with_sensor:
        request['entities'][0]['sensor_outputs'] = [range_output]
    result = compile_preview(request, paths, tmp_path / 'compiled')
    preview_model = mujoco.MjModel.from_binary_path(result['model_path'])
    preview_data = mujoco.MjData(preview_model)
    mujoco.mj_forward(preview_model, preview_data)
    groups = [SimpleNamespace(group_id='scout_group', model_id='bitcraze_crazyflie_2', placement_id='scout'),
              SimpleNamespace(group_id='anchor_group', model_id='radio_base_station', placement_id='anchors')]
    entities = request['entities']
    experiment = SimpleNamespace(scenario=SimpleNamespace(entity_groups=groups,
        placements={'scout': [Pose.model_validate(entities[0]['pose'])],
                    'anchors': [Pose.model_validate(entity['pose']) for entity in entities[1:]]},
        scene=SimpleNamespace(physics_dt=.003, integrator='RK4', density=1.2, viscosity=.000018)),
        topology=Topology(world_count=3), entity_models={entity['model']['resource_id']:
            EntityModelRef.model_validate(entity['model']) for entity in entities},
        sensors={'scout_group': {'outputs': {'range': range_output}}},
        activation=['sensor:scout_group'] if with_sensor else [], environment_scene='scene.xml')
    cfg, sim_cfg = assemble_scene(experiment, paths)
    assert cfg.num_envs == 3
    assert sim_cfg.nconmax == 128 and sim_cfg.njmax == 512
    assert sim_cfg.mujoco.timestep == .003 and sim_cfg.mujoco.flight_integrator == 'RK4'
    assert sim_cfg.mujoco.density == 1.2 and sim_cfg.mujoco.viscosity == .000018
    runtime_cpu = Scene(cfg, device='cpu').compile()
    runtime_data = mujoco.MjData(runtime_cpu)
    mujoco.mj_forward(runtime_cpu, runtime_data)
    np.testing.assert_array_equal(preview_model.sensor_type, runtime_cpu.sensor_type)
    np.testing.assert_array_equal(preview_model.sensor_objid, runtime_cpu.sensor_objid)
    np.testing.assert_array_equal(preview_model.sensor_intprm, runtime_cpu.sensor_intprm)
    np.testing.assert_allclose(preview_data.xpos, runtime_data.xpos, atol=1e-12)
    np.testing.assert_allclose(preview_data.xquat, runtime_data.xquat, atol=1e-12)
    objects = {obj['object_id']: obj for obj in result['objects']}
    for entity in entities:
        body = objects[entity['object_id']]['body_id']
        np.testing.assert_allclose(preview_data.xpos[body], entity['pose']['position'], atol=1e-12)
        np.testing.assert_allclose(preview_data.xquat[body], entity['pose']['orientation_wxyz'], atol=1e-12)
        if entity['group_id'] == 'scout_group':
            joint = preview_model.body_jntadr[body]
            address = preview_model.jnt_qposadr[joint]
            np.testing.assert_allclose(preview_data.qpos[address:address+7],
                [*entity['pose']['position'], *entity['pose']['orientation_wxyz']], atol=1e-12)
        else:
            assert preview_model.body_jntnum[body] == 0
            assert preview_model.body_mocapid[preview_model.body_parentid[body]] >= 0


def test_resolved_rangefinder_installs_but_omission_does_not_invent_one(preview_inputs, tmp_path):
    from muse_runtime.scene_preview import compile_preview

    source, paths, range_output = preview_inputs
    request = copy.deepcopy(source)
    request['entities'][0]['sensor_outputs'] = [range_output]
    result = compile_preview(request, paths, tmp_path / 'with-sensor')
    model = mujoco.MjModel.from_binary_path(result['model_path'])
    sensor = model.sensor('scout_group_0/rangefinder_down')
    assert sensor.type[0] == mujoco.mjtSensor.mjSENS_RANGEFINDER
    assert model.site('scout_group_0/rangefinder_down').pos.tolist() == [0., 0., -.01]
    omitted = compile_preview(source, paths, tmp_path / 'without-sensor')
    model = mujoco.MjModel.from_binary_path(omitted['model_path'])
    assert 'scout_group_0/rangefinder_down' not in [model.sensor(i).name for i in range(model.nsensor)]
    # 缺 profile 不删除资产本身已经声明的传感器。
    assert model.sensor('scout_group_0/body_gyro').id >= 0


def test_environment_objects_use_manifest_identity_and_actual_compiled_ids(preview_inputs, tmp_path):
    from muse_runtime.environment_edit import inspect_editable_environment
    from muse_runtime.scene_preview import compile_preview

    request, paths, _ = preview_inputs
    manifest = inspect_editable_environment(Path(paths['environment']), ResourceRef.model_validate(request['environment']))
    result = compile_preview(request, paths, tmp_path / 'compiled')
    model = mujoco.MjModel.from_binary_path(result['model_path'])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    objects = [obj for obj in result['objects'] if obj['kind'] == 'environment']
    assert [obj['object_id'] for obj in objects] == [obj.object_id for obj in manifest.objects]
    assert next(obj for obj in objects if obj['object_id'] == 'wall')['label'] == '原墙'
    anonymous_positions = []
    for obj in objects:
        geom, = obj['geom_ids']
        assert obj['body_id'] == model.geom_bodyid[geom]
        if model.geom(geom).name:
            assert model.geom(geom).name == 'environment/' + obj['object_id']
        else:
            anonymous_positions.append(data.geom_xpos[geom].tolist())
    assert anonymous_positions == [[0., 4., 1.], [0., 7.7, 1.]]
    ids = [geom for obj in result['objects'] for geom in obj['geom_ids']]
    assert len(ids) == len(set(ids)) == model.ngeom
    for entity in request['entities']:
        obj = next(obj for obj in result['objects'] if obj['object_id'] == entity['object_id'])
        assert obj['model_id'] == entity['model']['resource_id']
        assert obj['group_id'] == entity['group_id'] and obj['group_slot'] == entity['slot']


def test_environment_edit_is_temporary_and_base_resources_are_immutable(preview_inputs, tmp_path):
    from muse_runtime.scene_preview import compile_preview

    source, paths, _ = preview_inputs
    hashes = {key: hash_package(Path(path)) for key, path in paths.items()}
    request = copy.deepcopy(source)
    request['environment_edit'] = dict(base=request['environment'], removed=[], upserts={
        'wall': dict(label='临时墙', pose=dict(position=[4., 0., 1.], orientation_wxyz=[1., 0., 0., 0.]),
                     size_m=[2., 4., 6.], collision=False, rf_blocking=True)})
    destination = tmp_path / 'compiled'
    result = compile_preview(request, paths, destination)
    model = mujoco.MjModel.from_binary_path(result['model_path'])
    assert model.geom('environment/wall').pos.tolist() == [4., 0., 1.]
    assert model.geom('environment/wall').size.tolist() == [1., 2., 3.]
    assert next(obj for obj in result['objects'] if obj['object_id'] == 'wall')['label'] == '临时墙'
    assert {key: hash_package(Path(path)) for key, path in paths.items()} == hashes
    assert {path.name for path in destination.iterdir()} == {'preview.mjb', 'preview_objects.json'}
    assert json.loads((destination / 'preview_objects.json').read_text()) == result['objects']


def test_environment_without_any_entities_can_preview(preview_inputs, tmp_path):
    from muse_runtime.scene_preview import compile_preview

    source, paths, _ = preview_inputs
    request = {**source, 'entities': []}
    result = compile_preview(request, paths, tmp_path / 'compiled')
    assert all(obj['kind'] == 'environment' for obj in result['objects'])


def test_real_frozen_run_plan_still_compiles_through_shared_cpu_path(pipeline):
    from mjlab.scene import Scene
    from muse_runtime.scene import assemble_scene
    from muse_server.resolver import resolve_experiment

    catalog, source = pipeline
    report = resolve_experiment(source, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment, source=source)
    plan = catalog.create_run_plan(identity, mode='EVALUATE', device='cpu', max_decision_steps=1)
    cfg, sim_cfg = assemble_scene(plan.experiment, plan.resource_paths)
    model = Scene(cfg, device='cpu').compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.body('scout_group_0/cf2').id > 0
    assert sim_cfg.mujoco.timestep == plan.experiment.scenario.scene.physics_dt


def test_public_preview_dtos_do_not_accept_paths_or_model_metadata(preview_inputs):
    from muse_sdk.scene_edit import ScenePreviewRequest
    from muse_sdk.view_protocol import EditGesture, ViewIntent

    source, _, _ = preview_inputs
    entity = source['entities'][0]
    payload = dict(draft_id=source['draft_id'], revision=0, environment=source['environment'],
        entities=[dict(object_id=entity['object_id'], group_id=entity['group_id'], slot=0,
            model_id=entity['model']['resource_id'], model_revision=1, pose=entity['pose'])])
    request = ScenePreviewRequest.model_validate(payload)
    assert request.entities[0].sensor_profile_id is None and request.entities[0].sensor_config == {}
    assert request.environment_edit is None
    with pytest.raises(ValueError):
        ScenePreviewRequest.model_validate({**payload, 'resource_paths': {'environment': '/arbitrary'}})
    with pytest.raises(ValueError):
        ScenePreviewRequest.model_validate({**payload, 'entities': [{**payload['entities'][0], 'model': entity['model']}]})
    with pytest.raises(ValueError, match='unit quaternion'):
        ScenePreviewRequest.model_validate({**payload, 'entities': [{**payload['entities'][0],
            'pose': {**entity['pose'], 'orientation_wxyz': [2., 0., 0., 0.]}}]})
    with pytest.raises(ValueError, match='Duplicate preview object ID'):
        ScenePreviewRequest.model_validate({**payload, 'entities': payload['entities'] * 2})
    with pytest.raises(ValueError, match='Duplicate preview group/slot'):
        ScenePreviewRequest.model_validate({**payload, 'entities': [payload['entities'][0],
            {**payload['entities'][0], 'object_id': 'other-object'}]})
    assert ViewIntent(kind='reset_camera').object_id is None
    assert EditGesture(base_revision=0, object_id='drone', gesture_id='gesture', phase='end',
                       pose=request.entities[0].pose).phase == 'end'


def test_cpu_cli_never_initializes_cuda_or_research_runtime(preview_inputs, tmp_path):
    request, paths, _ = preview_inputs
    request_file, paths_file = tmp_path / 'request.json', tmp_path / 'paths.json'
    request_file.write_text(json.dumps(request))
    paths_file.write_text(json.dumps(paths))
    result_file = tmp_path / 'result.json'
    script = '''import ctypes, json, os, runpy, sys, threading
def guard(frame, event, arg):
    if event != 'call': return
    path, name = frame.f_code.co_filename.replace(chr(92), '/'), frame.f_code.co_name
    forbidden = (path.endswith('/mjlab/scene/scene.py') and name in ('initialize', 'reset')) or (
        path.endswith('/mjlab/sim/sim.py') and name == '__init__') or (
        path.endswith('/muse_runtime/env.py')) or (
        path.endswith('/torch/cuda/__init__.py') and name in ('init', '_lazy_init')) or (
        '/warp/' in path and name == 'init')
    assert not forbidden, (path, name)
sys.setprofile(guard)
threading.setprofile(guard)
runpy.run_module('muse_runtime.scene_preview', run_name='__main__')
import torch
import warp._src.context as warp_context
assert not torch.cuda.is_initialized()
assert warp_context.runtime is None
assert not {'muse_runtime.env', 'muse_runtime.package_loader', 'muse_runtime.validation'} & sys.modules.keys()
driver = ctypes.CDLL('libcuda.so.1')
driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
driver.cuCtxGetCurrent.restype = ctypes.c_int
context = ctypes.c_void_p()
status = driver.cuCtxGetCurrent(ctypes.byref(context))
assert status == 3 and context.value is None, (status, context.value)
for name in os.listdir('/proc/self/fd'):
    path = '/proc/self/fd/' + name
    if os.path.exists(path):
        target = os.readlink(path)
        assert '/dev/dxg' not in target and '/dev/nvidia' not in target, target
print('CPU_AUDIT_OK no CUDA context, Warp runtime, GPU FD, initialize/reset/Simulation/MuseEnv')
'''
    result = subprocess.run([sys.executable, '-c', script, '--request', str(request_file),
        '--resource-paths', str(paths_file), '--destination', str(tmp_path / 'compiled'),
        '--output', str(result_file)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CPU_AUDIT_OK' in result.stdout
    output = json.loads(result_file.read_text())
    assert output['applied_revision'] == 7
    assert Path(output['model_path']).is_file()
