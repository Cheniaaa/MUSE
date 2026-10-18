"""history声明/覆盖、真实包解析和raw/public shape各自的边界。"""
from copy import deepcopy

import pytest

from muse_sdk.config import ConfigError
from muse_sdk.descriptor import build_descriptor, iter_selections, parse_descriptor_json
from muse_server.resolver import resolve_package_selections, resolve_experiment


def history_descriptor(definition, composition, length):
    from muse_builtins.terms import registrations
    definition.terms['target'] = registrations()['relative_target']
    composition['agent_roles']['scout']['observation_terms'] = [
        dict(selection_id='target', term='target', history_length=length)]
    return build_descriptor(definition, {'default': composition}, 'a'*64, default_composition_id='default')


@pytest.mark.parametrize('length', [0, 1, 4])
def test_history_declaration_override_roundtrip(definition, composition, length):
    descriptor = history_descriptor(definition, composition, length)
    restored = parse_descriptor_json(descriptor.model_dump_json())
    assert restored == descriptor
    for _, kind, selections in iter_selections(restored.compositions['default']):
        if kind != 'observation':
            assert all('history_length' not in item.model_dump() for item in selections)
    resolved = resolve_package_selections(restored, {}, composition=restored.compositions['default'])
    target = next(item for item in resolved.selections if item.selection_id == 'target')
    assert target.history_length == length
    override = {'roles': {'scout': {'observation': {'target': {'history_length': 2}}}}}
    selected = next(item for item in resolve_package_selections(restored, override, composition=restored.compositions['default']).selections
                    if item.selection_id == 'target')
    assert selected.history_length == 2
    assert selected.config == target.config
    override['roles']['scout']['observation']['target']['history_length'] = 0
    selected = next(item for item in resolve_package_selections(restored, override, composition=restored.compositions['default']).selections
                    if item.selection_id == 'target')
    assert selected.history_length == 0


@pytest.mark.parametrize('value', [-1, True, 1.5, None])
def test_history_invalid_length_rejected_at_declaration_and_override(definition, composition, value):
    with pytest.raises(ValueError):
        history_descriptor(definition, composition, value)
    descriptor = history_descriptor(definition, composition, 4)
    with pytest.raises((ValueError, ConfigError)):
        resolve_package_selections(descriptor, {'roles': {'scout': {'observation': {
            'target': {'history_length': value}}}}}, composition=descriptor.compositions['default'])


@pytest.mark.parametrize('owner,kind,selection_id', [
    (('roles', 'scout'), 'reward', 'progress'), (('world',), 'termination', 'collision'),
    (('world',), 'cost', 'position')])
def test_nonobservation_explicit_zero_history_is_rejected(definition, composition, owner, kind, selection_id):
    if kind == 'cost':
        from muse_builtins.terms import registrations
        definition.terms['position'] = registrations()['position_bound_violation']
        composition['world_cost_terms'] = [dict(selection_id='position', term='position',
                                               config={'position_bound_limit_m': .01})]
    rows = (composition['agent_roles']['scout']['reward_terms'] if kind == 'reward'
            else composition['world_' + kind + '_terms'])
    rows[0]['history_length'] = 0
    with pytest.raises(ConfigError, match='history|observation'):
        build_descriptor(definition, {'default': composition}, 'a'*64, default_composition_id='default')
    rows[0].pop('history_length')
    descriptor = build_descriptor(definition, {'default': composition}, 'a'*64, default_composition_id='default')
    override = {kind: {selection_id: {'history_length': 0}}}
    for part in reversed(owner):
        override = {part: override}
    with pytest.raises(ConfigError, match='history|observation'):
        resolve_package_selections(descriptor, override, composition=descriptor.compositions['default'])


def test_history_resolved_public_shape_keeps_scope_and_raw_contract(pipeline, tmp_path):
    from mappo_helpers import cooperative_pipeline
    from muse_sdk.experiment import ExperimentDefinition, learning_contract
    catalog, request = cooperative_pipeline(pipeline, tmp_path)
    request['term_config'] = {
        'roles': {'rover': {'observation': {'target': {'history_length': 4}}}},
        'critic_inputs': {'team': {'observation': {
            'rover_target': {'history_length': 3}, 'phase': {'history_length': 2}}}}}
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    exp = report.resolved_experiment
    assert ExperimentDefinition.model_validate_json(exp.model_dump_json()) == exp
    raw = {(*s.owner_path, s.selection_id): s.output_spec.shape for s in exp.selections if s.enabled}
    assert raw['roles', 'rover', 'target'] == [2, 2, 2]
    assert raw['critic_inputs', 'team', 'rover_target'] == [2, 2, 2]
    assert raw['critic_inputs', 'team', 'phase'] == [2, 1]
    actor = {f.path[-1]: f.tensor.shape for f in exp.algorithm_inputs.actors['rover']}
    critic = {f.path[-1]: f.tensor.shape for f in exp.algorithm_inputs.critics['team']}
    assert actor['target'] == [2, 2, 4, 2]
    assert actor['velocity'] == [2, 2, 2]
    assert critic['rover_target'] == [2, 2, 3, 2]
    assert critic['phase'] == [2, 2, 1]
    changed = deepcopy(request)
    changed['topology']['world_count'] = 3
    other = resolve_experiment(changed, catalog.snapshot())
    assert other.valid, other.errors
    assert learning_contract(exp) == learning_contract(other.resolved_experiment)
    changed['term_config']['roles']['rover']['observation']['target']['history_length'] = 1
    other = resolve_experiment(changed, catalog.snapshot())
    assert other.valid, other.errors
    assert learning_contract(exp) != learning_contract(other.resolved_experiment)
