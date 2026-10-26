"""T1 的真实注册、CPU resolve 与资产保真；不启动 GPU 物理。"""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from muse_sdk.run_protocol import RunRequest
from flight_helpers import archive


@pytest.mark.skipif(os.environ.get('MUSE_TEST_GPU') != '1', reason='Explicit serial GPU acceptance')
def test_real_mixed_api_checkpoint_roundtrip(tmp_path):
    import hashlib
    import io
    import socket
    import httpx2
    import torch
    from test_runs_api import zip_body
    from test_supervisor import wait_for
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    with (tmp_path / 'server.log').open('wb') as log:
        server = subprocess.Popen([sys.executable, '-m', 'muse_server.app', '--port', str(port),
                                   '--data-dir', str(tmp_path / 'data')], stdout=log, stderr=log)
        try:
            with httpx2.Client(base_url=f'http://127.0.0.1:{port}', timeout=30.) as client:
                def healthy():
                    try:
                        return client.get('/health').is_success
                    except httpx2.HTTPError:
                        return False
                wait_for(healthy, timeout=30)
                request = json.loads((EXAMPLE / 'request_train.json').read_text())
                for kind, route, directory in [('environment', 'environments', 'environment'),
                                               ('training_package', 'training-packages', 'training_package')]:
                    response = client.post('/catalog/' + route, content=zip_body(EXAMPLE / directory),
                                           headers={'Content-Type': 'application/zip'})
                    assert response.status_code == 201, response.text
                    item = response.json()
                    request[kind + '_id'], request[kind + '_revision'] = item['resource_id'], item['revision']
                response = client.post('/experiments', json=request)
                assert response.status_code == 201, response.text
                identity = response.json()['experiment_id']
                reference = None
                for mode, steps, cumulative in [('TRAIN', 8, 2), ('TRAIN', 8, 4), ('EVALUATE', 3, 4)]:
                    response = client.post('/runs', json=dict(experiment_id=identity, mode=mode,
                        max_decision_steps=steps, checkpoint_artifact_id=reference))
                    assert response.status_code == 201, response.text
                    run_id = response.json()['run_id']
                    result = wait_for(lambda: (r if (r := client.get(f'/runs/{run_id}').json())['state']
                        in ('COMPLETED', 'FAILED', 'LOST', 'CANCELLED') else None), timeout=180)
                    assert result['state'] == 'COMPLETED', (result, Path(result['artifact_directory'], 'logs/worker.log').read_text())
                    assert result['exit_code'] == 0 and result['metrics']['steps'] == steps
                    assert result['metrics']['updates'] == (2 if mode == 'TRAIN' else 0)
                    saved = [a for a in client.get(f'/runs/{run_id}/artifacts').json() if a['kind'] == 'checkpoint']
                    if mode == 'TRAIN':
                        assert len(saved) == 1
                        item = saved[0]
                        content = client.get(f"/runs/{run_id}/artifacts/{item['id']}").content
                        assert len(content) == item['size'] and hashlib.sha256(content).hexdigest() == item['sha256']
                        state = torch.load(io.BytesIO(content), weights_only=True)['learning_state']
                        assert state['updates'] == cumulative and set(state['roles']) == {'scout', 'rover'}
                        reference = item['id']
                    else:
                        assert saved == []
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=15)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / 'examples/heterogeneous'
ORIGINAL_CAR = Path('/home/chen/MuJoCo/mujoco-3.12.0/model/car/car.xml')


def mixed_pipeline(pipeline, tmp_path):
    catalog, _ = pipeline
    environment = catalog.register_archive('environment', archive(EXAMPLE / 'environment', tmp_path / 'env.zip'))
    package = catalog.register_archive('training_package', archive(EXAMPLE / 'training_package', tmp_path / 'pkg.zip'))
    request = json.loads((EXAMPLE / 'request_train.json').read_text())
    request.update(environment_id=environment.resource_id, environment_revision=environment.revision,
                   training_package_id=package.resource_id, training_package_revision=package.revision)
    return catalog, request


def test_real_mixed_registration_and_heads(pipeline, tmp_path):
    from muse_server.resolver import resolve_experiment
    catalog, request = mixed_pipeline(pipeline, tmp_path)
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    exp = report.resolved_experiment
    assert exp.action_specs['scout'].shape == [2, 1, 4]
    drive = exp.action_specs['rover']
    assert drive.interface == 'diff_drive' and drive.shape == [2, 1, 2]
    assert drive.fields == ['v_forward', 'yaw_rate']
    assert drive.units == ['m/s', 'rad/s']
    assert drive.frames == ['body_flu', 'world_heading']
    assert drive.minimum == [-.1, -.8] and drive.maximum == [.1, .8]
    assert [(x.start, x.stop, x.shape) for x in exp.action_layout] == [(0, 4, [2, 1, 4]), (4, 6, [2, 1, 2])]
    assert {r: sum(x.tensor.shape[-1] for x in items) for r, items in exp.algorithm_inputs.actors.items()} == {
        'scout': 17, 'rover': 9}
    for model_id, root, support in [('bitcraze_crazyflie_2', 'cf2', []),
                                  ('ugv_diff_drive', 'car', ['front wheel', 'left_wheel', 'right_wheel'])]:
        model = exp.entity_models[model_id]
        assert model.root_body == root and model.support_geoms == support
    assert exp.controllers['rover_group']['config'] == {
        'wheel_radius_m': .03, 'wheel_separation_m': .12,
        'wheel_damping_nm_s': .03, 'wheel_velocity_kp_nm_s': .01}
    assert exp.training_schedule.collection_steps == 4
    rover_terms = {x.selection_id: x for x in exp.selections if x.owner_path == ('roles', 'rover')
                   and x.term_kind == 'observation'}
    assert {name: term.output_spec.shape for name, term in rover_terms.items()} == {
        'target': [2, 1, 2], 'velocity': [2, 1, 2], 'heading': [2, 1, 2],
        'yaw_rate': [2, 1, 1], 'previous_action': [2, 1, 2]}
    assert rover_terms['yaw_rate'].output_spec.frame == 'body_flu'
    assert rover_terms['heading'].input_fields[0].path == ['physics_outputs', 'orientation_wxyz', 'rover']
    assert rover_terms['previous_action'].input_fields[0].tensor.shape == [2, 1, 2]
    assert [x.selection_id for x in exp.selections if x.term_kind == 'termination'] == [
        'scout_collision', 'rover_collision', 'episode_limit']
    identity = catalog.save_experiment(exp, source=request)
    plan = catalog.create_run_plan(RunRequest(experiment_id=identity, mode='TRAIN', max_decision_steps=8), device='cpu')
    assert plan.max_decision_steps == 8
    assert plan.experiment.model_dump_json() == exp.model_dump_json()


@pytest.mark.parametrize('key,value', [('units', ['m/s', 'm/s']), ('fields', ['vx_b', 'yaw_rate']),
                                     ('frames', ['world_z_up', 'world_heading']), ('minimum', [-.1])])
def test_drive_motion_contract_rejected(pipeline, tmp_path, key, value):
    from muse_server.resolver import resolve_experiment
    catalog, request = mixed_pipeline(pipeline, tmp_path)
    snapshot = catalog.snapshot()
    snapshot.builtins.controllers['diff_drive_controller']['motion'][key] = value
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert any('action_specs.rover' in e['field'] for e in report.errors), report.errors


@pytest.mark.parametrize('mutation,field', [
    ('interface', 'role_config.rover'), ('binding', 'role_config.rover'),
    ('algorithm', 'algorithm_id'), ('previous_action', 'input_fields')])
def test_incompatible_mixed_contract_rejected(pipeline, tmp_path, mutation, field):
    from muse_server.resolver import resolve_experiment
    catalog, request = mixed_pipeline(pipeline, tmp_path)
    snapshot = catalog.snapshot()
    record = next(r for r in snapshot.resources if r.resource_id == request['training_package_id'])
    rover = record.metadata['compositions']['default']['agent_roles']['rover']
    if mutation == 'interface':
        rover['action_interfaces']['motion'] = 'body_velocity'
    elif mutation == 'binding':
        request['role_config']['rover']['group_id'] = 'scout_group'
    elif mutation == 'algorithm':
        snapshot.builtins.descriptor.algorithms['ippo'].action_requirements = ['body_velocity']
    else:
        rover['observation_terms'][-1]['term'] = 'builtin:previous_action'
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert any(field in e['field'] for e in report.errors), report.errors


def test_algorithm_supported_but_unselected_interface_is_allowed(pipeline):
    from muse_server.resolver import resolve_experiment
    catalog, request = pipeline
    snapshot = catalog.snapshot()
    snapshot.builtins.descriptor.algorithms['constant_velocity'].action_requirements = ['body_velocity', 'diff_drive']
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors


def test_planar_terms_read_declared_cache_with_exact_dimensions():
    import torch
    from muse_builtins import terms
    registrations = terms.registrations()
    assert {'planar_target', 'planar_velocity', 'heading', 'yaw_rate', 'previous_drive_action'} <= registrations.keys()
    env = SimpleNamespace(
        role_positions_world={'rover': torch.tensor([[[1., 2., .03]]])},
        commands=SimpleNamespace(target_world={'rover': torch.tensor([[[4., 6., 9.]]])}),
        physics_outputs=SimpleNamespace(
            velocity_world={'rover': torch.tensor([[[.2, -.3, 8.]]])},
            orientation_wxyz={'rover': torch.tensor([[[2 ** -.5, 0., 0., 2 ** -.5]]])},
            angular_velocity_body={'rover': torch.tensor([[[5., 6., -.7]]])},
            previous_action={'rover': torch.tensor([[[.05, -.4]]])}))
    for name, expected in [('planar_target', [3., 4.]), ('planar_velocity', [.2, -.3]),
                           ('heading', [0., 1.]), ('yaw_rate', [-.7]), ('previous_drive_action', [.05, -.4])]:
        actual = getattr(terms, name)(env, role_id='rover')
        torch.testing.assert_close(actual, torch.tensor([[expected]]), atol=1e-6, rtol=1e-6)


def test_split_car_preserves_compiled_physics():
    import mujoco
    import numpy as np
    asset = ROOT / 'src/muse_builtins/assets/ugv_diff_drive/car.xml'
    assert asset.is_file(), '缺少拆分 car 资产'
    original = mujoco.MjModel.from_xml_path(str(ORIGINAL_CAR))
    split = mujoco.MjModel.from_xml_path(str(asset))
    assert (split.nq, split.nv, split.nu, split.ntendon) == (9, 8, 2, 2)
    for name in ('body_mass', 'body_inertia', 'body_pos', 'body_quat', 'qpos0', 'dof_damping',
                 'jnt_type', 'jnt_pos', 'jnt_axis', 'jnt_actfrclimited', 'jnt_actfrcrange',
                 'tendon_adr', 'tendon_num', 'wrap_type', 'wrap_objid', 'wrap_prm',
                 'actuator_trntype', 'actuator_trnid', 'actuator_gear', 'actuator_dyntype',
                 'actuator_gaintype', 'actuator_biastype', 'actuator_dynprm', 'actuator_gainprm',
                 'actuator_biasprm', 'actuator_ctrllimited', 'actuator_ctrlrange',
                 'actuator_forcelimited', 'actuator_forcerange'):
        np.testing.assert_array_equal(getattr(split, name), getattr(original, name), err_msg=name)
    assert split.ngeom == original.ngeom - 1
    for name in ('geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_friction', 'geom_condim',
                 'geom_priority', 'geom_solref', 'geom_solimp', 'geom_contype', 'geom_conaffinity'):
        np.testing.assert_array_equal(getattr(split, name), getattr(original, name)[1:], err_msg=name)
    assert split.geom('left_wheel').bodyid == split.body('left wheel').id
    assert split.geom('right_wheel').bodyid == split.body('right wheel').id


def test_mixed_resolve_and_metadata_without_gpu_import(pipeline, tmp_path):
    catalog, request = mixed_pipeline(pipeline, tmp_path)
    request_path = tmp_path / 'request.json'
    request_path.write_text(json.dumps(request))
    script = '''
import importlib.abc, json, sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mujoco_warp', 'mjlab', 'warp'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from muse_builtins.metadata import export_metadata
metadata = export_metadata(Path(sys.argv[1]))
assert metadata.entity_models['ugv_diff_drive'].metadata['root_body'] == 'car'
from muse_server.catalog import Catalog
from muse_server.resolver import resolve_experiment
catalog = Catalog(Path(sys.argv[2]), Path(sys.argv[3]))
report = resolve_experiment(json.loads(Path(sys.argv[4]).read_text()), catalog.snapshot())
assert report.valid, report.errors
'''
    result = subprocess.run([sys.executable, '-c', script, str(ROOT / 'examples/basic_flight/controller.json'),
                             str(catalog.directory), str(catalog.builtin_path), str(request_path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
