"""CPU descriptor → Experiment 的合作学习及 Task 契约。"""
import copy
import subprocess
import sys

import pytest

from muse_sdk.descriptor import InputField, RoleBinding, parse_descriptor_json
from muse_sdk.experiment import ExperimentDefinition, learning_contract
from muse_server.resolver import resolve_experiment


def tensor(shape, dtype='float32', units='1', frame='none'):
    return dict(shape=shape, dtype=dtype, units=units, frame=frame, rate='decision')


def field(path, shape, phase='post_physics', **kwargs):
    return dict(path=path, phase=phase, tensor=tensor(shape, **kwargs))


def test_real_builtin_mappo_descriptor_roundtrip():
    from flight_helpers import EXAMPLE
    from muse_builtins.metadata import export_metadata
    metadata = export_metadata(EXAMPLE / 'controller.json')
    descriptor = parse_descriptor_json(metadata.descriptor.model_dump_json())
    assert descriptor.algorithms['mappo'].critic_inputs == 'all_centralized'
    assert descriptor.algorithms['mappo'].actor_observation == 'all_local'


def test_mappo_metadata_and_task_sdk_keep_cpu_import_boundary():
    from flight_helpers import EXAMPLE
    script = '''
import importlib.abc, sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mujoco', 'mjlab', 'warp'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from muse_sdk.experiment import BoundTask
from muse_sdk.runtime_types import TaskUpdateResult
from muse_builtins.metadata import export_metadata
metadata = export_metadata(Path(sys.argv[1]))
assert set(metadata.descriptor.algorithms) == {'constant_velocity', 'ippo', 'mappo'}
assert metadata.descriptor.algorithms['mappo'].critic_inputs == 'all_centralized'
'''
    result = subprocess.run([sys.executable, '-c', script, str(EXAMPLE / 'controller.json')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def team_case(pipeline):
    catalog, source = pipeline
    snapshot = catalog.snapshot()
    request = copy.deepcopy(source)
    package = next(r for r in snapshot.resources if r.resource_id == source['training_package_id'] and r.revision == 1)
    composition = package.metadata['composition']
    composition.update(algorithm='builtin:mappo', algorithm_config={}, critic_inputs={
        'team': {'observation_terms': [
            {'selection_id': 'scout_target', 'term': 'builtin:relative_target', 'role_id': 'scout'},
            {'selection_id': 'guard_target', 'term': 'builtin:relative_target', 'role_id': 'guard'},
        ]}})
    for role in composition['agent_roles'].values():
        role['reward_terms'] = []
    package.metadata['terms']['team_reward'] = dict(term_kind='reward', scope='world',
        config_schema={'type': 'object', 'properties': {}, 'additionalProperties': False},
        input_fields=[], output_spec=tensor(['B']))
    composition['world_reward_terms'] = [{'selection_id': 'team', 'term': 'team_reward'}]
    request.update(algorithm_id='builtin:mappo', algorithm_config={'collection_steps': 4})
    return request, snapshot, package


def resolve(case):
    request, snapshot, _ = case
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors
    return report.resolved_experiment


def add_task(case):
    request, _, package = case
    task = dict(task_id='mission', config_schema={'type': 'object', 'properties': {
        'threshold': {'type': 'integer', 'default': 3, 'minimum': 1}}, 'additionalProperties': False},
        input_fields=[
            field(['physics_outputs', 'episode_decision'], ['B'], dtype='int64'),
            field(['role_positions_world', 'guard'], ['B', 'N_role', 3], units='m', frame='world_z_up'),
            field(['commands', 'target_world', 'scout'], ['B', 1, 3], phase='next_command', units='m', frame='world_z_up'),
            field(['role_masks', 'guard', 'present'], ['B', 'N_role'], dtype='bool'),
        ], published_fields={'stage': tensor(['B'], dtype='int64'), 'features': tensor(['B', 2, 3])},
        role_status_bindings={}, learning_done_roles=['guard'])
    package.metadata['tasks']['mission'] = task
    package.metadata['composition'].update(task='mission', task_config={'threshold': 5})
    request.update(task_id='mission', task_config={'threshold': 7})
    return task


def test_centralized_inputs_are_explicit_and_actor_inputs_stay_local(team_case):
    exp = resolve(team_case)
    assert list(exp.algorithm_inputs.critics) == ['team']
    assert [f.path for f in exp.algorithm_inputs.critics['team']] == [
        ['observations', 'critic_inputs', 'team', 'scout_target'],
        ['observations', 'critic_inputs', 'team', 'guard_target']]
    assert [f.tensor.shape for f in exp.algorithm_inputs.critics['team']] == [[2, 1, 3], [2, 2, 3]]
    assert all(f.path[:3] == ['observations', 'actors', role]
               for role, fields in exp.algorithm_inputs.actors.items() for f in fields)


@pytest.mark.parametrize('disabled', [False, True])
def test_all_centralized_requires_enabled_observations(team_case, disabled):
    request, snapshot, package = team_case
    if disabled:
        request['term_config'] = {'critic_inputs': {'team': {'observation': {
            name: {'enabled': False} for name in ('scout_target', 'guard_target')}}}}
    else:
        package.metadata['composition']['critic_inputs'] = {}
    report = resolve_experiment(request, snapshot)
    assert not report.valid and any('critic' in e['field'] for e in report.errors)


@pytest.mark.parametrize('path,shape,allowed', [
    (['observations', 'actors', 'scout', 'target'], ['B', 1, 3], False),
    (['role_positions_world', 'scout'], ['B', 1, 3], False),
    (['observations', 'critic_inputs', 'team', 'scout_target'], ['B', 1, 3], True),
])
def test_explicit_critic_inputs_use_centralized_namespace(team_case, path, shape, allowed):
    request, snapshot, _ = team_case
    phase = 'post_physics' if path[0] == 'role_positions_world' else 'post_observation'
    snapshot.builtins.descriptor.algorithms['mappo'].critic_inputs = {'team': [InputField.model_validate(
        field(path, shape, phase=phase, units='m', frame='world_z_up'))]}
    report = resolve_experiment(request, snapshot)
    assert report.valid == allowed, report.errors
    if not allowed:
        assert any('critic' in e['field'] for e in report.errors)


def explicit_critic_inputs(case, group='team', observation_group='team'):
    algorithm = case[1].builtins.descriptor.algorithms['mappo']
    algorithm.critic_inputs = {group: [InputField.model_validate(field(
        ['observations', 'critic_inputs', observation_group, name], ['B', 'N_role', 3],
        phase='post_observation', units='m', frame='world_z_up'))
        for name in ('scout_target', 'guard_target')]}
    return algorithm.critic_inputs[group]


@pytest.mark.parametrize('group,observation_group', [
    ('team', 'team'), ('scout', 'team'), ('guard', 'team'),
    ('team', 'scout'), ('team', 'guard'),
])
def test_explicit_critic_roles_follow_selections_not_group_names(team_case, group, observation_group):
    composition = team_case[2].metadata['composition']
    composition['critic_inputs'][observation_group] = composition['critic_inputs'].pop('team')
    explicit_critic_inputs(team_case, group, observation_group)
    exp = resolve(team_case)
    inputs = exp.algorithm_inputs.critics[group]
    assert [item.tensor.shape for item in inputs] == [[2, 1, 3], [2, 2, 3]]
    assert [item.path for item in inputs] == [
        ['observations', 'critic_inputs', observation_group, 'scout_target'],
        ['observations', 'critic_inputs', observation_group, 'guard_target']]


def test_explicit_critic_symbolic_slots_follow_changed_role_binding(team_case):
    request = team_case[0]
    explicit_critic_inputs(team_case)
    assert [item.tensor.shape for item in resolve(team_case).algorithm_inputs.critics['team']] == [
        [2, 1, 3], [2, 2, 3]]
    binding = request['role_config']['guard']
    group = next(g for g in request['scenario']['entity_groups'] if g['group_id'] == binding['group_id'])
    group['count'] = 3
    request['scenario']['placements'][group['placement_id']].append(
        copy.deepcopy(request['scenario']['placements'][group['placement_id']][-1]))
    binding['slots'] = [0, 1, 2]
    request['scenario']['command_targets_world']['guard'].append([0., 0., 2.])
    assert [item.tensor.shape for item in resolve(team_case).algorithm_inputs.critics['team']] == [
        [2, 1, 3], [2, 3, 3]]


@pytest.mark.parametrize('group', ['team', 'scout', 'guard'])
def test_explicit_critic_world_observation_has_no_role_slots(team_case, group):
    request, snapshot, package = team_case
    package.metadata['terms']['world_observation'] = dict(term_kind='observation', scope='world',
        config_schema={'type': 'object', 'properties': {}, 'additionalProperties': False},
        input_fields=[], output_spec=tensor(['B', 2, 3]))
    package.metadata['composition']['critic_inputs']['team']['observation_terms'].append(
        {'selection_id': 'world', 'term': 'world_observation'})
    item = InputField.model_validate(field(['observations', 'critic_inputs', 'team', 'world'],
        ['B', 2, 3], phase='post_observation'))
    snapshot.builtins.descriptor.algorithms['mappo'].critic_inputs = {group: [item]}
    assert resolve_experiment(request, snapshot).valid
    item.tensor.shape = ['B', 'N_role', 3]
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert any('N_role' in error['message'] for error in report.errors), report.errors


@pytest.mark.parametrize('attribute,value', [
    ('shape', ['B', 'N_role', 4]), ('shape', ['B', 1, 3]), ('units', 'km'),
    ('frame', 'body_flu'),
])
def test_explicit_critic_preserves_declared_tensor_checks(team_case, attribute, value):
    items = explicit_critic_inputs(team_case)
    setattr(items[1].tensor, attribute, value)
    report = resolve_experiment(team_case[0], team_case[1])
    assert not report.valid
    assert any('mismatch' in error['message'] for error in report.errors), report.errors


@pytest.mark.parametrize('group', ['team', 'scout', 'guard'])
def test_explicit_critic_role_placeholder_requires_real_role_context(team_case, group):
    composition = team_case[2].metadata['composition']
    composition['critic_inputs'][group] = composition['critic_inputs'].pop('team')
    items = explicit_critic_inputs(team_case, group, group)
    # group 与 role 同名，也不能作为 RoleBinding 的上下文。
    items[0].path[2] = RoleBinding(binding='role_id')
    items[0].tensor.shape = ['B', 1, 3]
    team_case[1].builtins.descriptor.algorithms['mappo'].critic_inputs = {group: items[:1]}
    report = resolve_experiment(team_case[0], team_case[1])
    assert not report.valid
    assert any('binding' in error['message'] for error in report.errors), report.errors


@pytest.mark.parametrize('kind', ['absent', 'disabled', 'role'])
def test_world_only_learner_requires_enabled_world_reward(team_case, kind):
    request, snapshot, package = team_case
    composition = package.metadata['composition']
    if kind == 'disabled':
        request['term_config'] = {'world': {'reward': {'team': {'enabled': False}}}}
    elif kind == 'absent':
        composition['world_reward_terms'] = []
    else:
        composition['agent_roles']['scout']['reward_terms'] = [
            {'selection_id': 'distance', 'term': 'builtin:distance'}]
    report = resolve_experiment(request, snapshot)
    assert not report.valid and any('reward' in e['field'] or 'reward' in e['message'] for e in report.errors)


def test_world_requirement_uses_contract_not_algorithm_name(team_case):
    request, snapshot, package = team_case
    custom = snapshot.builtins.descriptor.algorithms['mappo'].model_dump(mode='json')
    custom['algorithm_id'] = 'custom_team'
    package.metadata['algorithms']['custom_team'] = custom
    package.metadata['composition'].update(algorithm='custom_team', world_reward_terms=[])
    request['algorithm_id'] = 'custom_team'
    assert not resolve_experiment(request, snapshot).valid
    custom['training_schedule'] = None
    assert resolve_experiment(request, snapshot).valid


def test_task_freezes_inputs_config_fields_and_learning_roles(team_case):
    task = add_task(team_case)
    exp = resolve(team_case)
    assert exp.task_id == 'mission' and exp.task_config == {'threshold': 7}
    assert exp.task_spec.learning_done_roles == ['guard']
    assert [f.tensor.shape for f in exp.task_spec.input_fields] == [[2], [2, 2, 3], [2, 1, 3], [2, 2]]
    assert exp.task_spec.published_fields['stage'].shape == [2]
    assert exp.task_spec.published_fields['features'].shape == [2, 2, 3]
    encoded = exp.model_dump_json()
    assert ExperimentDefinition.model_validate_json(encoded).model_dump_json() == encoded
    task['published_fields']['stage']['shape'].append(9)
    task['learning_done_roles'].append('scout')
    team_case[0]['task_config']['threshold'] = 99
    assert exp.model_dump_json() == encoded


def test_task_fields_are_available_to_observation_probe_and_reward(team_case):
    add_task(team_case)
    _, _, package = team_case
    package.metadata['terms']['stage_observation'] = dict(term_kind='observation', scope='world',
        config_schema={'type': 'object', 'properties': {}, 'additionalProperties': False},
        input_fields=[field(['task', 'stage'], ['B'], phase='post_task', dtype='int64')],
        output_spec=tensor(['B', 1]))
    package.metadata['composition']['critic_inputs']['team']['observation_terms'].append(
        {'selection_id': 'stage', 'term': 'stage_observation'})
    package.metadata['terms']['team_reward']['input_fields'] = [
        field(['task', 'features'], ['B', 2, 3], phase='post_task')]
    exp = resolve(team_case)
    stage = next(s for s in exp.selections if s.selection_id == 'stage')
    assert stage.input_fields[0].path == ['task', 'stage']
    assert stage.input_fields[0].tensor.shape == [2]
    reward = next(s for s in exp.selections if s.selection_id == 'team')
    assert reward.input_fields[0].tensor.shape == [2, 2, 3]


def test_task_term_cannot_read_undeclared_published_field(team_case):
    add_task(team_case)
    request, snapshot, package = team_case
    package.metadata['terms']['team_reward']['input_fields'] = [
        field(['task', 'missing'], ['B'], phase='post_task')]
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert any('Unavailable' in e['message'] and 'missing' in e['message'] for e in report.errors)


@pytest.mark.parametrize('mutation,message', [
    (lambda t: t.update(learning_done_roles=['absent']), 'role'),
    (lambda t: t.update(role_status_bindings={'guard': ['alive']}), 'M2.5'),
    (lambda t: t['published_fields']['features'].update(shape=['B', 'N_role', 3]), 'dimension'),
    (lambda t: t['input_fields'].append(field(['task', 'stage'], ['B'], phase='post_task', dtype='int64')), 'Unavailable'),
    (lambda t: t['input_fields'].append(field(['observations', 'actors', 'scout', 'target'], ['B', 1, 3], phase='post_observation', units='m', frame='world_z_up')), 'Task'),
    (lambda t: t['input_fields'].append(field(['act_commands', 'target_world', 'scout'], ['B', 1, 3], phase='action', units='m', frame='world_z_up')), 'Task'),
    (lambda t: t['input_fields'][0].update(phase='post_observation'), 'Task'),
    (lambda t: t['input_fields'][1].update(path=['role_positions_world', {'binding': 'role_id'}]), 'binding'),
    (lambda t: t['input_fields'][1]['tensor'].update(units='km'), 'mismatch'),
])
def test_task_invalid_dependencies_and_bindings_are_rejected(team_case, mutation, message):
    task = add_task(team_case)
    mutation(task)
    report = resolve_experiment(team_case[0], team_case[1])
    assert not report.valid
    assert any(message in e['message'] for e in report.errors), report.errors


@pytest.mark.parametrize('kind', ['mismatch', 'request_config', 'composition_config', 'invalid_config'])
def test_task_identity_and_config_errors(team_case, kind):
    request, snapshot, package = team_case
    if kind in ('mismatch', 'invalid_config'):
        add_task(team_case)
        if kind == 'mismatch':
            request['task_id'] = None
        else:
            request['task_config']['threshold'] = 0
    elif kind == 'request_config':
        request['task_config'] = {'unused': 1}
    else:
        package.metadata['composition']['task_config'] = {'unused': 1}
    report = resolve_experiment(request, snapshot)
    assert not report.valid, kind


def test_task_contract_ignores_batch_but_preserves_learning_semantics(team_case):
    add_task(team_case)
    first = resolve(team_case)
    team_case[0]['topology']['world_count'] = 3
    second = resolve(team_case)
    contract = learning_contract(first)
    assert contract == learning_contract(second)
    assert contract['task_spec']['published_fields']['features']['shape'] == [2, 3]
    assert contract['task_spec']['input_fields'][1]['tensor']['shape'] == [2, 3]
    second.task_spec.learning_done_roles = ['scout']
    assert contract != learning_contract(second)


@pytest.mark.parametrize('shape', [[], [3], [2, 3]])
def test_task_shared_fields_require_declared_world_batch_axis(team_case, shape):
    task = add_task(team_case)
    task['published_fields']['features']['shape'] = shape
    report = resolve_experiment(team_case[0], team_case[1])
    assert not report.valid
    assert any('published_fields' in e['field'] and 'B' in e['message'] for e in report.errors)


def test_task_config_merges_schema_composition_and_request(team_case):
    add_task(team_case)
    request, _, package = team_case
    request['task_config'] = {}
    assert resolve(team_case).task_config == {'threshold': 5}
    package.metadata['composition']['task_config'] = {}
    assert resolve(team_case).task_config == {'threshold': 3}


def test_ippo_local_critic_uses_actor_contract_without_global_declaration(pipeline):
    from types import SimpleNamespace
    import torch
    from muse_builtins.algorithms.ippo import IPPO
    from muse_sdk.runtime_types import Observations
    from test_experiment_pipeline import learning_report

    class TargetOnly(IPPO):
        @classmethod
        def describe(cls):
            return {**super().describe(), 'algorithm_id': 'target_only', 'actor_observation': {
                role: [field(['observations', 'actors', {'binding': 'role_id'}, 'target'],
                             ['B', 'N_role', 3], phase='post_observation', units='m', frame='world_z_up')]
                for role in ('scout', 'guard')}, 'critic_inputs': {}}

    report = learning_report(pipeline, reference='target_only', implementation=TargetOnly)
    assert report.valid, report.errors
    exp = report.resolved_experiment
    assert exp.algorithm_inputs.critics == {}
    algorithm = TargetOnly(exp.algorithm_config)
    algorithm.initialize(exp, SimpleNamespace(device='cpu'), None)
    observations = Observations({'scout': {'target': torch.zeros(2, 1, 3)},
                                 'guard': {'target': torch.zeros(2, 2, 3)}}, {})
    result = algorithm.act(observations, {}, None, False)
    assert result.role_actions['guard'].motion.shape == (2, 2, 4)


def test_task_is_optional_and_runtime_result_has_empty_status(pipeline):
    from muse_sdk.runtime_types import TaskUpdateResult
    catalog, request = pipeline
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    exp = report.resolved_experiment
    assert exp.task_spec is None
    assert learning_contract(exp)['task_spec'] is None
    done = {'guard': object()}
    result = TaskUpdateResult(role_status={}, learning_done=done, diagnostics={'stage': 1})
    assert tuple(result) == ({}, done, {'stage': 1})
