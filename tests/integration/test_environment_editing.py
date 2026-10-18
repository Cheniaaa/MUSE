"""独立真实环境包验证受限编辑；Catalog/HTTP 接线由主代理覆盖。"""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import mujoco
import numpy as np
import pytest
import yaml

from muse_sdk.content import hash_package
from muse_sdk.experiment import ResourceRef


@pytest.fixture
def environment(tmp_path):
    root = tmp_path / 'base'
    root.mkdir()
    (root / 'environment.yaml').write_text(
        'scene: scene.xml\nrf_geometry: {rf_blocking: [wall]}\nmetadata: {author: preserved}\n')
    (root / 'scene.xml').write_text('''<mujoco><worldbody>
      <geom name="wall" type="box" size=".1 1 1" pos="2 0 1"/>
      <geom name="floor" type="plane" size="10 10 .1"/>
      <geom name="off" type="box" size=".2 .2 .2" pos="0 5 1" contype="0" conaffinity="0"/>
    </worldbody></mujoco>''')
    return root, ResourceRef(resource_id='environment', revision=1, sha256=hash_package(root))


def box():
    from muse_sdk.scene_edit import BoxEdit

    return BoxEdit.model_validate(dict(
        label='编辑墙', pose=dict(position=[3., 0., 3.], orientation_wxyz=[1., 0., 0., 0.]),
        size_m=[2., 4., 6.], collision=True, rf_blocking=True))


def test_catalog_http_derivation_freezes_source_and_checks_base(environment, pipeline, tmp_path):
    from fastapi.testclient import TestClient
    from flight_helpers import archive
    from muse_server.app import create_app
    root, _ = environment
    catalog, _ = pipeline
    original = catalog.register_archive('environment', archive(root, tmp_path / 'base.zip'))
    reference = original.model_dump(include={'resource_id', 'revision', 'sha256'})
    with TestClient(create_app(catalog.directory)) as client:
        editing = client.get(f'/catalog/environments/{original.resource_id}/{original.revision}/editing')
        assert editing.status_code == 200, editing.text
        assert editing.json()['base'] == reference
        edit = dict(base=reference, upserts={'edited_box': box().model_dump()}, removed=['wall'])
        response = client.post('/catalog/environments/derive', json=edit)
        assert response.status_code == 201, response.text
        record = response.json()
        assert record['resource_id'] != original.resource_id
        assert record['metadata']['derived_from'] == reference
        assert record['sha256'] != original.sha256
        assert hash_package(Path(original.path)) == original.sha256
        model = mujoco.MjModel.from_xml_path(str(Path(record['path']) / record['metadata']['scene']))
        assert model.geom('edited_box').size.tolist() == [1., 2., 3.]
        assert model.geom('edited_box').contype[0] == 1
        assert 'wall' not in record['metadata']['rf_geometry']['rf_blocking']
        count = len(catalog.snapshot().resources)
        edit['base'] = {**reference, 'sha256': '0' * 64}
        assert client.post('/catalog/environments/derive', json=edit).status_code == 422
        edit.update(base=reference, removed=['floor'])
        failed = client.post('/catalog/environments/derive', json=edit)
        assert failed.status_code == 422 and 'readonly' in failed.text
        assert len(catalog.snapshot().resources) == count
        assert hash_package(Path(original.path)) == original.sha256
        assert client.get('/catalog/schema').json()['environment_edit']['additionalProperties'] is False


def test_box_edit_roundtrip_keeps_base_immutable(environment, tmp_path):
    from muse_runtime.environment_edit import derive_environment, inspect_editable_environment
    from muse_sdk.scene_edit import EnvironmentEdit

    root, reference = environment
    manifest = inspect_editable_environment(root, reference)
    objects = {obj.object_id: obj for obj in manifest.objects}
    assert objects['wall'].editable and objects['wall'].box.size_m == [.2, 2., 2.]
    assert objects['off'].box.collision is False and not objects['off'].box.rf_blocking
    assert not objects['floor'].editable and objects['floor'].reason
    edit = EnvironmentEdit(base=reference, upserts={'edited_box': box(), 'off': box()}, removed=['wall'])
    destination = tmp_path / 'derived'
    derive_environment(root, edit, destination)
    model = mujoco.MjModel.from_xml_path(str(destination / 'scene.xml'))
    assert model.geom('edited_box').size.tolist() == [1., 2., 3.]
    assert 'wall' not in [model.geom(i).name for i in range(model.ngeom)]
    assert hash_package(root) == reference.sha256
    assert hash_package(destination) != reference.sha256
    doc = yaml.safe_load((destination / 'environment.yaml').read_text())
    assert doc['rf_geometry']['rf_blocking'] == ['edited_box', 'off']
    assert doc['metadata']['author'] == 'preserved'
    derived_ref = reference.model_copy(update={'revision': 2, 'sha256': hash_package(destination)})
    after = inspect_editable_environment(destination, derived_ref)
    objects = {obj.object_id: obj for obj in after.objects}
    assert objects['edited_box'].box == box()
    assert objects['edited_box'].label == '编辑墙'
    assert after == inspect_editable_environment(destination, derived_ref)


def test_mjcf_scene_extension_uses_the_same_native_xml_contract(environment, tmp_path):
    from muse_runtime.environment_edit import derive_environment, inspect_editable_environment
    from muse_sdk.scene_edit import EnvironmentEdit
    root, reference = environment
    (root / 'scene.xml').rename(root / 'scene.mjcf')
    path = root / 'environment.yaml'
    path.write_text(path.read_text().replace('scene.xml', 'scene.mjcf'))
    reference = reference.model_copy(update={'sha256': hash_package(root)})
    manifest = inspect_editable_environment(root, reference)
    assert next(obj for obj in manifest.objects if obj.object_id == 'wall').editable
    destination = tmp_path / 'derived-mjcf'
    derive_environment(root, EnvironmentEdit(base=reference, upserts={'wall': box()}, removed=[]), destination)
    assert mujoco.MjModel.from_xml_path(str(destination / 'scene.mjcf')).geom('wall').size.tolist() == [1., 2., 3.]


@pytest.mark.parametrize('collision,rf_blocking', [(False, True), (True, False), (False, False), (True, True)])
def test_collision_and_rf_are_independent(environment, tmp_path, collision, rf_blocking):
    from muse_runtime.environment_edit import derive_environment, inspect_editable_environment
    from muse_sdk.scene_edit import EnvironmentEdit

    root, reference = environment
    edited = box().model_copy(update={'collision': collision, 'rf_blocking': rf_blocking})
    destination = tmp_path / 'derived'
    derive_environment(root, EnvironmentEdit(base=reference, upserts={'wall': edited}, removed=[]), destination)
    model = mujoco.MjModel.from_xml_path(str(destination / 'scene.xml'))
    assert (int(model.geom('wall').contype[0]), int(model.geom('wall').conaffinity[0])) == (
        int(collision), int(collision))
    assert yaml.safe_load((destination / 'environment.yaml').read_text())['rf_geometry']['rf_blocking'] == (
        ['wall'] if rf_blocking else [])
    result = {obj.object_id: obj for obj in inspect_editable_environment(destination, reference).objects}
    assert result['wall'].box.collision is collision
    assert result['wall'].box.rf_blocking is rf_blocking


def test_referenced_and_nested_geoms_are_readonly(tmp_path):
    from muse_runtime.environment_edit import derive_environment, inspect_editable_environment
    from muse_sdk.scene_edit import EnvironmentEdit

    root = tmp_path / 'base'
    root.mkdir()
    (root / 'environment.yaml').write_text('scene: scene.xml\n')
    (root / 'scene.xml').write_text('''<mujoco>
      <asset><mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/></asset>
      <worldbody>
        <geom name="paired" type="box" size="1 1 1"/>
        <geom name="pair_other" type="box" size="1 1 1" pos="5 0 0"/>
        <geom name="sensed" type="box" size="1 1 1" pos="10 0 0"/>
        <geom name="sensor_reference" type="box" size="1 1 1" pos="15 0 0"/>
        <geom name="tuple_member" type="box" size="1 1 1" pos="20 0 0"/>
        <geom name="custom_mask" type="box" size="1 1 1" contype="2" conaffinity="4"/>
        <geom name="asymmetric_mask" type="box" size="1 1 1" contype="1" conaffinity="0"/>
        <geom type="box" size="1 1 1"/>
        <body name="nested"><geom name="mesh" type="mesh" mesh="tetra"/>
          <geom name="nested_box" type="box" size="1 1 1"/></body>
        <body name="moving"><freejoint/><geom name="dynamic" type="box" size="1 1 1"/></body>
        <frame pos="1 2 3"><geom name="framed" type="box" size="1 1 1"/></frame>
      </worldbody>
      <contact><pair geom1="paired" geom2="pair_other"/></contact>
      <sensor><framepos name="position" objtype="geom" objname="sensed"
          reftype="geom" refname="sensor_reference"/></sensor>
      <custom><tuple name="selection"><element objtype="geom" objname="tuple_member"/></tuple></custom>
    </mujoco>''')
    reference = ResourceRef(resource_id='readonly', revision=1, sha256=hash_package(root))
    manifest = inspect_editable_environment(root, reference)
    assert manifest == inspect_editable_environment(root, reference)
    objects = {obj.object_id: obj for obj in manifest.objects}
    for obj in objects.values():
        assert not obj.editable and obj.reason and obj.box is None
    for name, reason in [('paired', 'contact'), ('pair_other', 'contact'), ('sensed', 'sensor'),
                         ('sensor_reference', 'sensor'), ('tuple_member', 'tuple'),
                         ('custom_mask', 'mask'), ('asymmetric_mask', 'mask'), ('nested_box', 'worldbody'),
                         ('dynamic', 'worldbody'), ('mesh', 'box'), ('framed', 'frame')]:
        assert reason in objects[name].reason
    for name in objects:
        destination = tmp_path / 'invalid'
        with pytest.raises(ValueError, match='readonly'):
            derive_environment(root, EnvironmentEdit(base=reference, upserts={}, removed=[name]), destination)
        assert not destination.exists()
    assert hash_package(root) == reference.sha256


def test_unedited_assets_survive_serialization(environment, tmp_path):
    from muse_runtime.environment_edit import derive_environment
    from muse_sdk.scene_edit import EnvironmentEdit

    root, _ = environment
    (root / 'assets').mkdir()
    asset = Path(__file__).resolve().parents[2] / 'src/muse_builtins/assets/bitcraze_crazyflie_2/assets/cf2_0.obj'
    shutil.copy2(asset, root / 'assets/cf2.obj')
    (root / 'part.xml').write_text('''<mujocoinclude>
      <asset><mesh name="visual_mesh" file="assets/cf2.obj"/>
        <material name="paint" rgba=".15 .4 .7 .9" specular=".37" shininess=".63" reflectance=".11"/></asset>
      <worldbody><geom name="visual" type="mesh" mesh="visual_mesh" material="paint" pos="0 6 2"/>
        <geom name="pair1" type="box" size=".2 .3 .4" pos="0 8 1"/>
        <geom name="pair2" type="box" size=".2 .3 .4" pos="0 10 1"/>
      </worldbody><contact><pair geom1="pair1" geom2="pair2" friction=".7 .7 .01 .001 .001"/></contact>
    </mujocoinclude>''')
    scene = root / 'scene.xml'
    scene.write_text(scene.read_text().replace('<worldbody>', '<include file="part.xml"/><worldbody>'))
    reference = ResourceRef(resource_id='included', revision=1, sha256=hash_package(root))
    before = mujoco.MjModel.from_xml_path(str(scene))
    destination = tmp_path / 'derived'
    derive_environment(root, EnvironmentEdit(base=reference, upserts={'wall': box()}, removed=[]), destination)
    after = mujoco.MjModel.from_xml_path(str(destination / 'scene.xml'))
    for name in ('floor', 'off', 'visual', 'pair1', 'pair2'):
        for field in ('type', 'pos', 'quat', 'size', 'contype', 'conaffinity', 'rgba', 'friction', 'dataid', 'matid'):
            np.testing.assert_allclose(getattr(before.geom(name), field), getattr(after.geom(name), field),
                                       atol=2e-6, rtol=2e-6)
    for field in ('mesh_vert', 'mesh_face', 'mesh_normal', 'mesh_texcoord', 'mesh_pos', 'mesh_quat', 'mesh_scale',
                  'mat_rgba', 'mat_specular', 'mat_shininess', 'mat_reflectance',
                  'pair_geom1', 'pair_geom2', 'pair_friction', 'pair_solref', 'pair_solimp'):
        np.testing.assert_allclose(getattr(before, field), getattr(after, field), atol=2e-6, rtol=2e-6)
    for field in ('timestep', 'integrator', 'gravity', 'density', 'viscosity'):
        np.testing.assert_allclose(getattr(before.opt, field), getattr(after.opt, field), atol=2e-6, rtol=2e-6)
    assert (destination / 'part.xml').read_bytes() == (root / 'part.xml').read_bytes()
    assert (destination / 'assets/cf2.obj').read_bytes() == (root / 'assets/cf2.obj').read_bytes()
    assert '<include' not in (destination / 'scene.xml').read_text()
    assert hash_package(root) == reference.sha256


def test_edit_pose_and_size_validation_is_at_dto_boundary():
    from muse_sdk.scene_edit import BoxEdit, EnvironmentEdit

    payload = box().model_dump()
    for size in ([0., 1., 1.], [-1., 1., 1.], [1., 2.], [1., 2., 3., 4.], [float('inf'), 1., 1.]):
        with pytest.raises(ValueError):
            BoxEdit.model_validate({**payload, 'size_m': size})
    payload['pose']['orientation_wxyz'] = [2., 0., 0., 0.]
    with pytest.raises(ValueError, match='unit quaternion'):
        BoxEdit.model_validate(payload)
    reference = ResourceRef(resource_id='test', revision=1, sha256='a' * 64)
    with pytest.raises(ValueError, match='both'):
        EnvironmentEdit(base=reference, upserts={'wall': box()}, removed=['wall'])


def test_compiler_fusestatic_does_not_make_nested_box_editable(environment):
    from muse_runtime.environment_edit import inspect_editable_environment

    root, reference = environment
    (root / 'scene.xml').write_text('''<mujoco><compiler fusestatic="true"/>
      <worldbody><body name="nested"><geom name="nested_box" type="box" size="1 1 1"/></body>
      </worldbody></mujoco>''')
    obj, = inspect_editable_environment(root, reference).objects
    assert not obj.editable and 'worldbody' in obj.reason


def test_rotation_and_label_change_do_not_change_native_identity(environment, tmp_path):
    from muse_runtime.environment_edit import derive_environment, inspect_editable_environment
    from muse_sdk.scene_edit import EnvironmentEdit

    root, reference = environment
    path = root / 'scene.xml'
    path.write_text(path.read_text().replace('name="wall"', 'name="wall" euler="0 0 30"'))
    q = 2 ** -.5
    edited = box().model_copy(update={'label': '不是原生名称', 'pose': box().pose.model_copy(
        update={'orientation_wxyz': [q, 0., 0., q]})})
    destination = tmp_path / 'derived'
    derive_environment(root, EnvironmentEdit(base=reference, upserts={'wall': edited}, removed=[]), destination)
    model = mujoco.MjModel.from_xml_path(str(destination / 'scene.xml'))
    np.testing.assert_allclose(model.geom('wall').quat, [q, 0., 0., q], atol=1e-6)
    objects = {obj.object_id: obj for obj in inspect_editable_environment(destination, reference).objects}
    assert objects['wall'].label == '不是原生名称'
    assert '不是原生名称' not in objects


def test_fromto_box_is_updated_using_world_pose_and_actual_size(environment, tmp_path):
    from muse_runtime.environment_edit import derive_environment
    from muse_sdk.scene_edit import EnvironmentEdit

    root, reference = environment
    (root / 'scene.xml').write_text('''<mujoco><worldbody>
      <geom name="wall" type="box" size=".1 .2" fromto="0 0 0 0 0 2"/>
      </worldbody></mujoco>''')
    destination = tmp_path / 'derived'
    derive_environment(root, EnvironmentEdit(base=reference, upserts={'wall': box()}, removed=[]), destination)
    model = mujoco.MjModel.from_xml_path(str(destination / 'scene.xml'))
    np.testing.assert_allclose(model.geom('wall').pos, [3., 0., 3.])
    np.testing.assert_allclose(model.geom('wall').size, [1., 2., 3.])


def test_cpu_contacts_and_native_ray_follow_the_same_derived_wall(environment, tmp_path):
    from muse_runtime.environment_edit import derive_environment
    from muse_sdk.scene_edit import EnvironmentEdit

    root, _ = environment
    path = root / 'scene.xml'
    path.write_text(path.read_text().replace('</worldbody>', '''
      <body name="probe" pos="1.85 0 1"><freejoint/><geom name="probe_geom" size=".1" mass="1"/></body>
      </worldbody>'''))
    reference = ResourceRef(resource_id='contact', revision=1, sha256=hash_package(root))
    destination = tmp_path / 'derived'
    moved = box().model_copy(update={'pose': box().pose.model_copy(update={'position': [2., 3., 1.]}),
                                     'size_m': [.2, 2., 2.]})
    derive_environment(root, EnvironmentEdit(base=reference, upserts={'wall': moved}, removed=[]), destination)
    contacts, rays = [], []
    for directory in (root, destination):
        model = mujoco.MjModel.from_xml_path(str(directory / 'scene.xml'))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        contacts.append(any({model.geom(int(c.geom1)).name, model.geom(int(c.geom2)).name} ==
                            {'wall', 'probe_geom'} for c in data.contact))
        hit = np.array([-1], dtype=np.int32)
        rays.append(mujoco.mj_ray(model, data, np.array([0., 0., 1.]), np.array([1., 0., 0.]),
                                  None, True, model.body('probe').id, hit))
    assert contacts == [True, False]
    np.testing.assert_allclose(rays, [1.9, -1.])
    assert hash_package(root) == reference.sha256


def test_unknown_removal_and_existing_destination_do_not_modify_base(environment, tmp_path):
    from muse_runtime.environment_edit import derive_environment
    from muse_sdk.scene_edit import EnvironmentEdit

    root, reference = environment
    with pytest.raises(ValueError, match='Unknown'):
        derive_environment(root, EnvironmentEdit(base=reference, upserts={}, removed=['unknown']), tmp_path / 'bad')
    with pytest.raises(FileExistsError):
        derive_environment(root, EnvironmentEdit(base=reference, upserts={}, removed=[]), root)
    with pytest.raises(ValueError, match='base'):
        derive_environment(root, EnvironmentEdit(base=reference, upserts={}, removed=[]), root / 'nested')
    assert hash_package(root) == reference.sha256
    assert not (tmp_path / 'bad').exists()


def test_cli_inspect_and_derive_are_cpu_only(environment, tmp_path):
    from muse_sdk.scene_edit import EnvironmentEdit, EnvironmentManifest

    root, reference = environment
    reference_file, edit_file = tmp_path / 'reference.json', tmp_path / 'edit.json'
    reference_file.write_text(reference.model_dump_json())
    edit_file.write_text(EnvironmentEdit(base=reference, upserts={'edited_box': box()}, removed=[]).model_dump_json())
    # 同一真实 CLI 入口；导入/执行都不能进入训练或 GPU 模块。
    script = '''import runpy, sys
runpy.run_module('muse_runtime.environment_edit', run_name='__main__')
assert not {'torch', 'warp', 'mjlab', 'mujoco_warp', 'muse_runtime.env'} & sys.modules.keys()
'''
    manifest_file, result_file, destination = tmp_path / 'manifest.json', tmp_path / 'result.json', tmp_path / 'derived'
    for operation, input_file, output_file, extra in [
        ('inspect', reference_file, manifest_file, []),
        ('derive', edit_file, result_file, ['--destination', str(destination)]),
    ]:
        result = subprocess.run([sys.executable, '-c', script, operation, '--root', str(root),
                                 '--input', str(input_file), '--output', str(output_file), *extra],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert result.stdout == ''
    assert EnvironmentManifest.model_validate_json(manifest_file.read_text()).base == reference
    assert json.loads(result_file.read_text()) == {'destination': str(destination)}
    assert mujoco.MjModel.from_xml_path(str(destination / 'scene.xml')).geom('edited_box').size.tolist() == [1., 2., 3.]
