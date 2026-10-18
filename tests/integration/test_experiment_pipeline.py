import copy
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
from flight_helpers import EXAMPLE, archive


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module', autouse=True)
def source_environment():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('PYTHONPATH', str(ROOT / 'src'))
        patch.setenv('PYTHONDONTWRITEBYTECODE', '1')
        yield


def test_real_registration_and_frozen_flight_experiment(pipeline):
    from muse_server.resolver import resolve_experiment
    catalog, request = pipeline
    snapshot = catalog.snapshot()
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors
    exp = report.resolved_experiment
    assert exp.topology.world_count == 2
    assert exp.bindings['scout'].role_slot_to_group_slot == [0]
    assert exp.bindings['guard'].role_slot_to_group_slot == [0, 1]
    assert exp.action_specs['guard'].shape == [2, 2, 4]
    controller = json.loads((EXAMPLE / 'controller.json').read_text())
    for key, value in controller['motion'].items():
        assert getattr(exp.action_specs['guard'], key) == value
    assert exp.scenario.scene.physics_dt == .002 and exp.scenario.scene.control_substeps == 20
    assert exp.scenario.scene.integrator == 'RK4'
    assert exp.scenario.scene.density == 1.225 and exp.scenario.scene.viscosity == 1.8e-5
    assert exp.training_schedule is None
    assert exp.model_bindings == {} and exp.costs.model_dump() == {'roles': {}, 'world': {}}
    assert not any(x in exp.activation for x in ('radio', 'sensing', 'queue'))
    rewards = [s for s in exp.selections if s.term_kind == 'reward']
    assert [(s.config['scale'], s.weight) for s in rewards] == [(2.0, 1.0), (1.0, 0.5)]
    assert rewards[0].output_spec.shape == [2, 1]
    assert rewards[1].output_spec.shape == [2, 2]
    before = exp.model_dump_json()
    request_copy = copy.deepcopy(request)
    request_copy['term_config'] = {'roles': {'scout': {'reward': {'distance': {'config': {'scale': 9.0}}}}}}
    other = resolve_experiment(request_copy, snapshot)
    assert other.valid, other.errors
    assert exp.model_dump_json() == before
    identity = catalog.save_experiment(exp, source=request)
    assert catalog.get_experiment(identity).model_dump_json() == before
    plan = catalog.create_run_plan(identity, mode='EVALUATE', device='cpu')
    assert plan.experiment.model_dump_json() == before
    assert Path(plan.resource_paths['environment']).is_dir()
    assert Path(plan.artifact_directory, 'resolved_run_plan.json').is_file()
    with pytest.raises(ValueError, match='TRAIN'):
        catalog.create_run_plan(identity, mode='TRAIN', device='cpu')


@pytest.mark.parametrize('mutation,field', [
    (lambda r: r['scenario']['entity_groups'][0].pop('placement_id'), 'placement'),
    (lambda r: r['scenario']['entity_groups'][0].pop('controller_id'), 'controller'),
    (lambda r: r['scenario']['entity_groups'][0].update(controller_id='absent'), 'controller'),
    (lambda r: r.update(unknown=True), 'unknown'),
    (lambda r: r['role_config']['guard'].update(group_id='scout_group', slots=[0]), 'role_config'),
    (lambda r: r['scenario']['entity_groups'][0].update(count=2), 'placement'),
])
def test_invalid_requests_are_field_errors(pipeline, mutation, field):
    from muse_server.resolver import resolve_experiment
    catalog, source = pipeline
    request = copy.deepcopy(source)
    mutation(request)
    report = resolve_experiment(request, catalog.snapshot())
    assert not report.valid and report.resolved_experiment is None
    assert any(field in error['field'] for error in report.errors), report.errors


@pytest.mark.parametrize('name,content', [('../escape', 'x'), ('/absolute', 'x'),
    ('bad.json', '{"x":1,"x":2}'), ('bad.yaml', 'x: 1\nx: 2\n')])
def test_archive_entry_rejected(pipeline, tmp_path, name, content):
    catalog, _ = pipeline
    path = tmp_path / 'bad.zip'
    with zipfile.ZipFile(path, 'w') as stream:
        stream.writestr('environment.yaml', 'scene: scene.xml\n')
        stream.writestr(name, content)
    with pytest.raises(ValueError):
        catalog.register_archive('environment', path)


def test_duplicate_and_symlink_archive_rejected(pipeline, tmp_path):
    catalog, _ = pipeline
    for link in (False, True):
        path = tmp_path / f'bad-{link}.zip'
        with zipfile.ZipFile(path, 'w') as stream:
            if link:
                info = zipfile.ZipInfo('link')
                info.create_system = 3
                info.external_attr = 0o120777 << 16
                stream.writestr(info, '/tmp/elsewhere')
            else:
                stream.writestr('duplicate', 'a')
                with pytest.warns(UserWarning):
                    stream.writestr('duplicate', 'b')
        with pytest.raises(ValueError):
            catalog.register_archive('environment', path)


@pytest.mark.parametrize('batch', [1, 2])
def test_binding_order_and_dimensions_are_not_request_dictionary_order(pipeline, batch):
    from muse_server.resolver import resolve_experiment
    catalog, source = pipeline
    request = copy.deepcopy(source)
    request['topology']['world_count'] = batch
    request['role_config'] = dict(reversed(list(request['role_config'].items())))
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    exp = report.resolved_experiment
    assert [entry.role_id for entry in exp.action_layout] == ['scout', 'guard']
    assert exp.action_specs['scout'].shape == [batch, 1, 4]
    assert exp.action_specs['guard'].shape == [batch, 2, 4]
    assert exp.environment_scene == 'scene/scene.mjcf'


@pytest.mark.parametrize('key,value', [('units', ['m/s'] * 4), ('fields', ['x', 'y', 'z']),
                                     ('frames', ['world_z_up'] * 4)])
def test_platform_action_contract_mismatch_is_rejected(pipeline, key, value):
    from muse_server.resolver import resolve_experiment
    catalog, request = pipeline
    snapshot = catalog.snapshot()
    snapshot.builtins.controllers['cf2_body_velocity_v1']['motion'][key] = value
    report = resolve_experiment(request, snapshot)
    assert not report.valid and any('action_specs' in e['field'] for e in report.errors)


@pytest.mark.parametrize('mutation', [
    lambda field: field.update(phase='post_observation'),
    lambda field: field['tensor'].update(units='km'),
    lambda field: field['tensor'].update(shape=['B', 'N_role', 2]),
    lambda field: field.update(path=['network', 'unsupported']),
])
def test_unavailable_input_contract_rejected(pipeline, mutation):
    from muse_sdk.descriptor import InputField
    from muse_server.resolver import resolve_experiment
    catalog, request = pipeline
    snapshot = catalog.snapshot()
    term = snapshot.builtins.descriptor.terms['relative_target']
    field = term.input_fields[0].model_dump(mode='json')
    mutation(field)
    term.input_fields[0] = InputField.model_validate(field)
    report = resolve_experiment(request, snapshot)
    assert not report.valid and any('input_fields' in e['field'] for e in report.errors)


def test_custom_package_uses_same_worker_and_resolver(pipeline, tmp_path):
    from muse_server.resolver import resolve_experiment
    catalog, source = pipeline
    path = tmp_path / 'custom.zip'
    with zipfile.ZipFile(path, 'w') as stream:
        stream.writestr('__init__.py', '''
from muse_sdk.registration import TrainingPackageDefinition
from muse_builtins.rules import ConstantVelocity
from muse_builtins.terms import registrations
def build_training_package():
    return TrainingPackageDefinition(registrations(), {}, {'constant_velocity': ConstantVelocity}, {}, {})
''')
        stream.writestr('training_package.yaml',
                       (EXAMPLE / 'training_package/training_package.yaml').read_text().replace('builtin:', ''))
    record = catalog.register_archive('training_package', path)
    request = copy.deepcopy(source)
    request.update(training_package_id=record.resource_id, training_package_revision=record.revision)
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    assert report.resolved_experiment.algorithm_id == 'constant_velocity'


def test_resource_revision_and_experiment_remain_independent(pipeline, tmp_path):
    from muse_server.resolver import resolve_experiment
    catalog, request = pipeline
    before = resolve_experiment(request, catalog.snapshot()).resolved_experiment.model_dump_json()
    revision = catalog.register_archive('training_package',
        archive(EXAMPLE / 'training_package', tmp_path / 'revision.zip'), resource_id=request['training_package_id'])
    assert revision.revision == 2 and revision.resource_id == request['training_package_id']
    assert resolve_experiment(request, catalog.snapshot()).resolved_experiment.model_dump_json() == before


def test_environment_cannot_reference_external_assets(pipeline, tmp_path):
    catalog, _ = pipeline
    path = tmp_path / 'external.zip'
    with zipfile.ZipFile(path, 'w') as stream:
        stream.writestr('environment.yaml', 'scene: scene.xml\n')
        stream.writestr('scene.xml', '<mujoco><include file="/home/chen/old.xml"/></mujoco>')
    with pytest.raises(ValueError, match='bundled'):
        catalog.register_archive('environment', path)


def test_schemas_are_saved_for_worker_consumers(pipeline):
    catalog, _ = pipeline
    for name in ('experiment.schema.json', 'run-plan.schema.json'):
        schema = json.loads((catalog.directory / name).read_text())
        assert schema['additionalProperties'] is False


def test_server_and_sdk_import_boundaries(pipeline, tmp_path):
    catalog, request = pipeline
    request_path = tmp_path / 'request.json'
    request_path.write_text(json.dumps(request))
    script = '''
import importlib.abc, json, sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'muse_runtime','muse_builtins','torch','mujoco','mjlab','warp'} or fullname.startswith('_muse_package_'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from muse_sdk.runtime_types import ActResult
from muse_server.catalog import Catalog
from muse_server.resolver import resolve_experiment
catalog = Catalog(Path(sys.argv[1]), Path(sys.argv[2]))
report = resolve_experiment(json.loads(Path(sys.argv[3]).read_text()), catalog.snapshot())
assert report.valid, report.errors
'''
    result = subprocess.run([sys.executable, '-c', script, str(catalog.directory), str(catalog.builtin_path),
                             str(request_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_disabled_selection_does_not_require_runtime_role_binding(pipeline):
    from muse_server.resolver import resolve_experiment
    catalog, request = pipeline
    snapshot = catalog.snapshot()
    package = next(r for r in snapshot.resources if r.resource_id == request['training_package_id'] and r.revision == 1)
    package.metadata['compositions']['default']['world_termination_terms'].append({
        'selection_id': 'unused', 'term': 'builtin:collision', 'default_enabled': False,
        'config': {'controlled_role': 'not_in_scenario'},
    })
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors
    unused = report.resolved_experiment.selections[-1]
    assert not unused.enabled and unused.input_fields == [] and unused.output_spec is None
    assert report.resolved_experiment.termination_reasons[4] == ('world', 'termination', 'unused')


def test_reward_can_read_later_declared_role_observation_at_fixed_stage(pipeline):
    from muse_sdk.descriptor import InputField
    from muse_server.resolver import resolve_experiment
    catalog, request = pipeline
    snapshot = catalog.snapshot()
    term = snapshot.builtins.descriptor.terms['distance']
    # 声明是兼容性oracle，不将测试替身作为实际builtin发布。
    term.input_fields = [InputField(path=['observations', 'actors', 'guard', 'target'],
        phase='post_observation', tensor={'shape': ['B', 2, 3], 'dtype': 'float32',
                                         'units': 'm', 'frame': 'world_z_up', 'rate': 'decision'})]
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors


def test_two_same_role_selections_have_independent_resolved_values(pipeline):
    from muse_server.resolver import resolve_experiment
    catalog, source = pipeline
    snapshot = catalog.snapshot()
    package = next(r for r in snapshot.resources if r.resource_id == source['training_package_id'] and r.revision == 1)
    package.metadata['compositions']['default']['agent_roles']['scout']['reward_terms'].append({
        'selection_id': 'distance_aux', 'term': 'builtin:distance', 'config': {'scale': 4.0}, 'weight': 0.25})
    request = copy.deepcopy(source)
    request['term_config'] = {'roles': {'scout': {'reward': {'distance': {'config': {'scale': 3.0}}}}}}
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors
    selections = [s for s in report.resolved_experiment.selections if s.term_kind == 'reward']
    assert [(s.config['scale'], s.weight) for s in selections] == [(3., 1.), (4., .25), (1., .5)]
    snapshot.builtins.controllers['cf2_body_velocity_v1']['config']['tilt_limit_rad'] = 100.
    assert report.resolved_experiment.controllers['scout_group']['config']['tilt_limit_rad'] == .2


@pytest.mark.parametrize('mode,checkpoint,message', [
    ('UNKNOWN', None, 'Unknown Run mode'), ('EVALUATE', 'checkpoint-id', 'Rule EVALUATE'),
    ('INTERACTIVE', 'checkpoint-id', 'Rule INTERACTIVE')])
def test_unsupported_run_mode_or_rule_checkpoint_cannot_create_plan(pipeline, mode, checkpoint, message):
    from muse_server.resolver import resolve_experiment
    catalog, request = pipeline
    identity = catalog.save_experiment(resolve_experiment(request, catalog.snapshot()).resolved_experiment, source=request)
    with pytest.raises(ValueError, match=message):
        catalog.create_run_plan(identity, mode=mode, device='cuda:0', checkpoint_artifact_id=checkpoint)


def test_builtin_metadata_export_is_cpu_only_and_self_contained(tmp_path):
    script = '''
import importlib.abc, sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','mujoco_warp','mjlab','warp'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from muse_builtins.metadata import export_metadata
# 原生库导入可能由GLFW探测动态库；此处约束exporter的业务检查不再派生进程。
def check_process_boundary(event, args):
    if event == 'subprocess.Popen':
        raise AssertionError('Metadata exporter must perform CPU inspection in its own process')
sys.addaudithook(check_process_boundary)
metadata = export_metadata(Path(sys.argv[1]))
assert set(metadata.descriptor.algorithms) == {'constant_velocity', 'ippo', 'mappo'}
assert metadata.descriptor.algorithms['constant_velocity'].training_schedule is None
schedule = metadata.descriptor.algorithms['ippo'].training_schedule
assert metadata.descriptor.algorithms['ippo'].config_schema['properties'][schedule.collection_steps]['type'] == 'integer'
'''
    result = subprocess.run([sys.executable, '-c', script, str(EXAMPLE / 'controller.json')],
                             capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_request_dto_roundtrip_preserves_public_model_config_name(pipeline):
    from muse_sdk.experiment import ExperimentRequest
    _, request = pipeline
    encoded = json.loads(ExperimentRequest.model_validate(request).model_dump_json())
    assert 'model_config' in encoded and 'model_parameters' not in encoded
    assert json.loads(ExperimentRequest.model_validate(encoded).model_dump_json()) == encoded


def test_nonzip_archive_rejected_even_when_content_is_valid(pipeline, tmp_path):
    import io
    import tarfile
    catalog, _ = pipeline
    path = tmp_path / 'upload.zip'
    with tarfile.open(path, 'w') as stream:
        for name, content in [('environment.yaml', b'scene: scene.xml\n'), ('scene.xml', b'<mujoco/>')]:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            stream.addfile(member, io.BytesIO(content))
    with pytest.raises(ValueError, match='ZIP'):
        catalog.register_archive('environment', path)


def test_resolved_role_requirements_are_frozen(pipeline):
    from muse_server.resolver import resolve_experiment
    catalog, request = pipeline
    snapshot = catalog.snapshot()
    package = next(r for r in snapshot.resources if r.resource_id == request['training_package_id'] and r.revision == 1)
    requirement = {'path': ['role_positions_world', {'binding': 'role_id'}], 'phase': 'post_physics',
                   'tensor': {'shape': ['B', 'N_role', 3], 'dtype': 'float32', 'units': 'm',
                              'frame': 'world_z_up', 'rate': 'decision'}}
    package.metadata['compositions']['default']['agent_roles']['guard']['input_requirements'] = [requirement]
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors
    assert report.resolved_experiment.role_inputs['guard'][0].path == ['role_positions_world', 'guard']
    assert report.resolved_experiment.role_inputs['guard'][0].tensor.shape == [2, 2, 3]


def test_environment_metadata_type_is_checked_at_registration(pipeline, tmp_path):
    catalog, _ = pipeline
    path = tmp_path / 'wrong-metadata.zip'
    with zipfile.ZipFile(path, 'w') as stream:
        stream.writestr('environment.yaml', 'scene: scene.xml\nmetadata: not-an-object\n')
        stream.writestr('scene.xml', '<mujoco/>')
    with pytest.raises(ValueError, match='metadata'):
        catalog.register_archive('environment', path)


def test_disabling_algorithm_required_observation_is_rejected(pipeline):
    from muse_sdk.descriptor import InputField
    from muse_server.resolver import resolve_experiment
    catalog, source = pipeline
    snapshot = catalog.snapshot()
    package = next(r for r in snapshot.resources if r.resource_id == source['training_package_id'] and r.revision == 1)
    package.metadata['compositions']['default']['agent_roles']['scout']['observation_terms'][0]['required'] = False
    algorithm = snapshot.builtins.descriptor.algorithms['constant_velocity']
    algorithm.actor_observation = {'scout': [InputField(
        path=['observations', 'actors', {'binding': 'role_id'}, 'target'], phase='post_observation',
        tensor={'shape': ['B', 'N_role', 3], 'dtype': 'float32', 'units': 'm',
                'frame': 'world_z_up', 'rate': 'decision'})]}
    assert resolve_experiment(source, snapshot).valid
    request = copy.deepcopy(source)
    request['term_config'] = {'roles': {'scout': {'observation': {'target': {'enabled': False}}}}}
    report = resolve_experiment(request, snapshot)
    assert not report.valid and any('algorithm' in e['field'] for e in report.errors)


def learning_report(pipeline, *, reference='builtin:ippo', horizon=4, empty_observations=False,
                    world_reward=False, implementation=None):
    import yaml
    from muse_builtins.algorithms.ippo import IPPO
    from muse_sdk.config import DataModel
    from muse_sdk.descriptor import build_descriptor
    from muse_sdk.registration import TermRegistration, TrainingPackageDefinition
    from muse_server.resolver import resolve_experiment

    catalog, source = pipeline
    snapshot = catalog.snapshot()
    record = next(r for r in snapshot.resources if r.resource_id == source['training_package_id'])
    composition = yaml.safe_load((EXAMPLE / 'training_package_ippo/training_package.yaml').read_text())['compositions']['default']
    composition['algorithm'] = reference
    if empty_observations:
        for role in composition['agent_roles'].values():
            role['observation_terms'] = []
    terms = {}
    if world_reward:
        terms['team_bonus'] = TermRegistration(lambda env: None, 'reward', 'world', DataModel, (),
            {'shape': ['B'], 'dtype': 'float32', 'units': '1', 'frame': 'none', 'rate': 'decision'})
        composition['world_reward_terms'] = [{'selection_id': 'team_bonus', 'term': 'team_bonus', 'weight': 1.}]
    algorithms = {} if reference.startswith('builtin:') else {reference: implementation or IPPO}
    descriptor = build_descriptor(TrainingPackageDefinition(terms, {}, algorithms, {}, {}),
                                  {'default': composition}, record.sha256, snapshot.builtins.descriptor, default_composition_id='default')
    record.metadata = descriptor.model_dump(mode='json')
    request = copy.deepcopy(source)
    request.update(algorithm_config={'collection_steps': horizon})
    return resolve_experiment(request, snapshot)


@pytest.mark.parametrize('reference', ['builtin:ippo', 'ippo'])
def test_algorithm_registration_preserves_schedule_and_inputs(pipeline, reference):
    report = learning_report(pipeline, reference=reference)
    assert report.valid, report.errors
    exp = report.resolved_experiment
    assert exp.training_schedule.collection_steps == exp.algorithm_config['collection_steps'] == 4
    assert {r: len(items) for r, items in exp.algorithm_inputs.actors.items()} == {'scout': 5, 'guard': 5}


@pytest.mark.parametrize('reference', ['builtin:ippo', 'ippo'])
def test_learning_requires_declared_local_observations(pipeline, reference):
    report = learning_report(pipeline, reference=reference, empty_observations=True)
    assert not report.valid
    assert any('actor_observation' in e['field'] for e in report.errors)


@pytest.mark.parametrize('reference', ['builtin:ippo', 'ippo'])
def test_ippo_rejects_unconsumed_world_reward(pipeline, reference):
    report = learning_report(pipeline, reference=reference, world_reward=True)
    assert not report.valid
    assert any('team_bonus' in e['field'] and 'reward' in e['message'].lower() for e in report.errors)


def test_ippo_consumes_resolved_observations_not_all_selections(pipeline):
    from types import SimpleNamespace
    import torch
    from muse_builtins.algorithms.ippo import IPPO
    from muse_sdk.runtime_types import Observations

    class TargetOnly(IPPO):
        @classmethod
        def describe(cls):
            return {**super().describe(), 'algorithm_id': 'target_only', 'actor_observation': {
                role: [{'path': ['observations', 'actors', {'binding': 'role_id'}, 'target'],
                        'phase': 'post_observation', 'tensor': {'shape': ['B', 'N_role', 3],
                        'dtype': 'float32', 'units': 'm', 'frame': 'world_z_up', 'rate': 'decision'}}]
                for role in ('scout', 'guard')}, 'critic_inputs': {}}

    report = learning_report(pipeline, reference='target_only', horizon=24, implementation=TargetOnly)
    assert report.valid, report.errors
    exp = report.resolved_experiment
    algorithm = TargetOnly(exp.algorithm_config)
    algorithm.initialize(exp, SimpleNamespace(device='cpu'), None)
    obs = Observations({role: {'target': torch.zeros(2, n, 3)} for role, n in [('scout', 1), ('guard', 2)]}, {})
    result = algorithm.act(obs, {}, None, False)
    assert result.role_actions['guard'].motion.shape == (2, 2, 4)


def test_user_algorithm_declares_its_own_schedule_parameter_and_reward_scope(pipeline):
    from pydantic import Field
    from muse_sdk.config import DataModel
    from muse_sdk.descriptor import build_descriptor
    from muse_sdk.registration import TrainingPackageDefinition
    from muse_server.resolver import resolve_experiment

    class Config(DataModel):
        rollout_length: int = Field(default=7, gt=0)

    from muse_sdk.interfaces import Algorithm

    class UserLearner(Algorithm):
        @classmethod
        def describe(cls):
            return dict(algorithm_id='user_learner', config_schema=Config, observation_requirements=[],
                        action_requirements=['body_velocity'], actor_observation='all_local', critic_inputs={},
                        reward_inputs=['role', 'world'], act_aux={}, diagnostics={},
                        training_schedule={'collection_steps': 'rollout_length'})

    catalog, source = pipeline
    snapshot = catalog.snapshot()
    record = next(r for r in snapshot.resources if r.resource_id == source['training_package_id'])
    composition = copy.deepcopy(record.metadata['compositions']['default'])
    composition.update(algorithm='user_learner', algorithm_config={})
    descriptor = build_descriptor(TrainingPackageDefinition({}, {}, {'user_learner': UserLearner}, {}, {}),
                                  {'default': composition}, record.sha256, snapshot.builtins.descriptor, default_composition_id='default')
    record.metadata = descriptor.model_dump(mode='json')
    request = {**source, 'algorithm_config': {'rollout_length': 11}}
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors
    assert report.resolved_experiment.training_schedule.collection_steps == 11


def test_world_reward_is_accepted_when_algorithm_declares_consumption(pipeline):
    from types import SimpleNamespace
    import torch
    from muse_builtins.algorithms.ippo import IPPO
    from muse_sdk.runtime_types import Observations, RoleMask, TransitionView

    class TeamLearner(IPPO):
        @classmethod
        def describe(cls):
            return {**super().describe(), 'algorithm_id': 'team_learner', 'reward_inputs': ['role', 'world']}

        def process_env_step(self, transition, act_aux):
            rewards = {role: values + transition.world_reward[:, None] for role, values in transition.rewards.items()}
            return super().process_env_step(transition._replace(rewards=rewards), act_aux)

    report = learning_report(pipeline, reference='team_learner', world_reward=True, implementation=TeamLearner)
    assert report.valid, report.errors
    exp = report.resolved_experiment
    algorithm = TeamLearner(exp.algorithm_config)
    algorithm.initialize(exp, SimpleNamespace(device='cpu'), None)
    obs = Observations({role: {item.path[3]: torch.zeros(item.tensor.shape) for item in items}
                        for role, items in exp.algorithm_inputs.actors.items()}, {})
    masks = {role: RoleMask(*(torch.ones(spec.shape[:2], dtype=torch.bool) for _ in range(3)))
             for role, spec in exp.action_specs.items()}
    result = algorithm.act(obs, masks, None, False)
    done = torch.zeros(2, dtype=torch.bool)
    transition = TransitionView(masks, obs, {r: torch.zeros_like(m.alive, dtype=torch.float32) for r, m in masks.items()},
        torch.tensor([10., 20.]), done, done, torch.zeros(2, dtype=torch.int64),
        {r: torch.zeros_like(m.alive) for r, m in masks.items()}, done, {'roles': {}, 'world': {}}, wireless=None)
    algorithm.process_env_step(transition, result.act_aux)
    torch.testing.assert_close(algorithm.children['guard'].storage.rewards[0, :, 0], torch.tensor([10., 10., 20., 20.]))


def test_declared_entity_asset_is_frozen_and_used(pipeline, tmp_path):
    from muse_server.resolver import resolve_experiment
    from muse_runtime.scene import assemble_scene

    catalog, source = pipeline
    snapshot = catalog.snapshot()
    model = snapshot.builtins.entity_models['bitcraze_crazyflie_2']
    model.metadata['physics_asset'] = 'renamed.xml'
    report = resolve_experiment(source, snapshot)
    assert report.valid, report.errors
    # CPU MjSpec即可验证文件选择，不创建Simulation或占用GPU。
    (tmp_path / 'renamed.xml').write_text('<mujoco><worldbody><body name="cf2"><freejoint/><geom size=".1"/></body></worldbody></mujoco>')
    environment = next(record for record in snapshot.resources
                       if record.resource_id == source['environment_id'] and record.revision == source['environment_revision'])
    scene, _ = assemble_scene(report.resolved_experiment, {'entity_model:bitcraze_crazyflie_2': str(tmp_path),
                                                         'environment': environment.path})
    assert scene.entities['scout_group_0'].spec_fn().body('cf2') is not None


def test_checkpoint_contract_ignores_only_runtime_batch(pipeline):
    from muse_sdk.experiment import learning_contract
    from muse_server.resolver import resolve_experiment
    catalog, source = pipeline
    first = resolve_experiment(source, catalog.snapshot()).resolved_experiment
    changed = copy.deepcopy(source)
    changed['topology']['world_count'] = 1
    changed['root_seed'] += 1
    second = resolve_experiment(changed, catalog.snapshot()).resolved_experiment
    before = first.model_dump_json()
    assert learning_contract(first) == learning_contract(second)
    assert first.model_dump_json() == before
    second.algorithm_config['shape'] = [1, 3]
    assert learning_contract(first) != learning_contract(second)


@pytest.mark.parametrize('mutation', [
    lambda e: e.algorithm_inputs.actors['scout'].reverse(),
    lambda e: setattr(e.algorithm_inputs.actors['scout'][0].tensor, 'units', 'different'),
    lambda e: e.action_specs['guard'].shape.__setitem__(1, 3),
    lambda e: e.algorithm_config.update(hidden_dims=[16]),
    lambda e: e.bindings['guard'].role_slot_to_group_slot.reverse(),
    lambda e: e.selections.reverse(),
    lambda e: setattr(e.training_package, 'revision', 2),
])
def test_checkpoint_contract_distinguishes_learning_semantics(pipeline, mutation):
    from muse_sdk.experiment import learning_contract
    report = learning_report(pipeline)
    assert report.valid, report.errors
    exp = report.resolved_experiment
    changed = exp.model_copy(deep=True)
    mutation(changed)
    assert learning_contract(exp) != learning_contract(changed)


def test_checkpoint_reference_survives_run_plan_json(pipeline):
    from muse_sdk.experiment import RunPlan
    from muse_sdk.run_protocol import CheckpointReference
    catalog, _ = pipeline
    exp = learning_report(pipeline).resolved_experiment
    plan = RunPlan(run_id='target', experiment_id='exp', mode='TRAIN', device='cpu',
        platform_revision=exp.platform_revision, artifact_directory=str(catalog.directory),
        resource_paths={}, experiment=exp, max_decision_steps=8, checkpoint_interval_updates=1,
        checkpoint=CheckpointReference(artifact_id='source:checkpoint:abc', path='/tmp/abc.pt',
                                       size=10, sha256='a' * 64))
    loaded = RunPlan.model_validate_json(plan.model_dump_json())
    assert loaded.checkpoint.artifact_id == 'source:checkpoint:abc'
    assert loaded.checkpoint_interval_updates == 1
