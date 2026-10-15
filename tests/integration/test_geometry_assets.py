"""真实Catalog登记必须从原生资产产生事实，而非信任用户metadata。"""
import json
import subprocess
import sys

import pytest
import yaml
from flight_helpers import archive


def test_registration_inspects_include_mesh_and_moving_ancestors(pipeline, tmp_path):
    catalog, _ = pipeline
    root = tmp_path / 'assets'
    root.mkdir()
    (root / 'environment.yaml').write_text(yaml.safe_dump({
        'scene': 'scene.xml', 'rf_geometry': {'rf_blocking': ['wall', 'mesh']},
        'metadata': {'geometry_summary': {'geoms': {'joint_child': {'is_static': True}}}}}))
    (root / 'scene.xml').write_text('''<mujoco><include file="part.xml"/></mujoco>''')
    (root / 'part.xml').write_text('''<mujocoinclude>
      <asset><mesh name="tetra" scale="2 3 4" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/></asset>
      <worldbody>
        <body name="fixed"><geom name="wall" type="box" size="1 1 1"/>
          <geom name="mesh" type="mesh" mesh="tetra"/></body>
        <body name="moving"><joint/><geom size=".1"/>
          <body name="child"><geom name="joint_child" size=".1"/></body></body>
        <body name="mocap" mocap="true"><body name="mocap_child">
          <geom name="tracked" size=".1"/></body></body>
        <flexcomp name="cloth" type="grid" count="2 2 1" spacing=".2 .2 .2" dim="2"/>
      </worldbody></mujocoinclude>''')
    record = catalog.register_archive('environment', archive(root, tmp_path / 'env.zip'))
    facts = record.metadata['geometry_summary']
    assert facts['geoms']['wall'] == dict(geom_type='box', body='fixed', is_static=True, mesh=None)
    assert facts['geoms']['mesh']['mesh'] == 'tetra'
    assert facts['geoms']['mesh']['is_static']
    assert not facts['geoms']['joint_child']['is_static']
    assert not facts['geoms']['tracked']['is_static']
    assert facts['flex_names'] == ['cloth']
    assert '' not in facts['geoms']
    assert next(r for r in catalog.snapshot().resources if r.resource_id == record.resource_id) == record
    assert (root / 'part.xml').read_bytes() == (__import__('pathlib').Path(record.path) / 'part.xml').read_bytes()


def test_invalid_native_scene_is_rejected_at_registration(pipeline, tmp_path):
    catalog, _ = pipeline
    root = tmp_path / 'bad'
    root.mkdir()
    (root / 'environment.yaml').write_text('scene: scene.xml\n')
    (root / 'scene.xml').write_text('<mujoco><worldbody><geom type="missing"/></worldbody></mujoco>')
    with pytest.raises(ValueError, match='environment.scene'):
        catalog.register_archive('environment', archive(root, tmp_path / 'bad.zip'))


def test_builtin_radios_have_verified_body_and_explicit_ideal_assumption(pipeline):
    catalog, _ = pipeline
    for name, body in [('bitcraze_crazyflie_2', 'cf2'), ('radio_base_station', 'station')]:
        device = catalog.builtins.entity_models[name].metadata['radio_devices']['ideal_point_radio']
        assert device['body'] == body
        assert device['position_m'] == [0., 0., 0.]
        assert device['can_transmit'] and device['can_receive']
        assert device['assumption']


@pytest.mark.parametrize('invalid_model,missing_body', [
    ('bitcraze_crazyflie_2', 'cf2'), ('radio_base_station', 'station')])
def test_metadata_reports_missing_radio_body_from_native_inspection(tmp_path, monkeypatch, invalid_model, missing_body):
    from flight_helpers import EXAMPLE
    from muse_builtins import metadata

    # 仅替换资源根，真实exporter与原生编译处理临时MJCF；不改内置资产。
    package = tmp_path / 'muse_builtins'
    for name, filename, body in [('bitcraze_crazyflie_2', 'cf2.xml', 'cf2'),
                               ('ugv_diff_drive', 'car.xml', 'car'),
                               ('radio_base_station', 'station.xml', 'station')]:
        root = package / 'assets' / name
        root.mkdir(parents=True)
        body = 'incorrect_body' if name == invalid_model else body
        (root / filename).write_text(f'<mujoco><worldbody><body name="{body}"><geom size=".1"/></body></worldbody></mujoco>')
    monkeypatch.setattr(metadata, '__file__', str(package / 'metadata.py'))
    with pytest.raises(KeyError, match=missing_body):
        metadata.export_metadata(EXAMPLE / 'controller.json')


def test_inspection_process_is_cpu_only(tmp_path):
    (tmp_path / 'scene.xml').write_text('<mujoco><worldbody><geom name="wall" size="1 1 1" type="box"/></worldbody></mujoco>')
    result = subprocess.run([sys.executable, '-c', '''
import sys
from pathlib import Path
from muse_runtime.resource_inspection import inspect_environment
print(inspect_environment(Path(sys.argv[1]), 'scene.xml').model_dump_json())
assert not {'torch', 'warp', 'mjlab', 'mujoco_warp'} & sys.modules.keys()
''', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['geoms']['wall']['is_static']
