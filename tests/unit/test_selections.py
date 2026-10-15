from copy import deepcopy
from dataclasses import replace
import json

import pytest

from muse_sdk.config import ConfigError
from muse_sdk.descriptor import build_descriptor
from muse_sdk.selection import ResolvedPackageSelection
from muse_server.resolver import resolve_package_selections


def override(**values):
    return {'roles': {'scout': {'reward': {'progress_aux': values}}}}


def test_role_overrides_do_not_leak(descriptor):
    values = {'roles': {'scout': {'reward': {'progress': {'config': {'scale': 3.0}, 'weight': 0.25}}}}}
    original = deepcopy((descriptor, values))
    result = resolve_package_selections(descriptor, values)
    scout, aux, guard = result.selections[:3]
    assert (scout.config['scale'], scout.weight, scout.binding_role_id) == (3.0, 0.25, 'scout')
    assert (guard.config['scale'], guard.weight, guard.binding_role_id) == (1.0, 0.5, 'guard')
    assert not aux.enabled and 'scale' not in aux.config
    assert len(result.required_inputs) == 2
    assert (descriptor, values) == original
    scout.config['scale'] = 999
    assert guard.config['scale'] == 1.0
    assert resolve_package_selections(descriptor, values).selections[0].config['scale'] == 3.0


def test_required_cannot_disable(descriptor):
    request = {'roles': {'scout': {'reward': {'progress': {'enabled': False}}}}}
    with pytest.raises(ConfigError):
        resolve_package_selections(descriptor, request)


def test_enable_requires_config_and_creates_independent_values(descriptor):
    with pytest.raises(ConfigError):
        resolve_package_selections(descriptor, override(enabled=True))
    result = resolve_package_selections(descriptor, override(enabled=True, config={'scale': 4.0}, weight=0.0))
    first, aux = result.selections[:2]
    assert aux.enabled and aux.weight == 0 and aux.config['scale'] == 4.0
    assert len(result.required_inputs) == 3
    aux.config['scale'] = 9
    assert first.config['scale'] == 2.0


@pytest.mark.parametrize('values', [
    {'config': {'scale': 'bad'}}, {'config': {'role_id': 'guard'}}, {'config': None},
    {'enabled': None}, {'enabled': 1}, {'weight': None}, {'weight': float('nan')},
    {'term': 'collision'}, {'scope': 'world'}, {'cost_id': 'x'}, {'time_out': True},
])
def test_invalid_override_even_for_disabled_selection(descriptor, values):
    with pytest.raises((ConfigError, ValueError)):
        resolve_package_selections(descriptor, override(**values))


@pytest.mark.parametrize('payload', [
    {'unknown': {}}, {'roles': {'missing': {}}}, {'roles': {'scout': {'invalid': {}}}},
    {'roles': {'scout': {'reward': {'missing': {}}}}},
    {'roles': []}, {'world': {'termination': {'collision': {'weight': 1.0}}}},
])
def test_unknown_identity_rejected(descriptor, payload):
    with pytest.raises((ConfigError, ValueError)):
        resolve_package_selections(descriptor, payload)


def test_reason_identity_roundtrip_and_disabled_stability(descriptor):
    result = resolve_package_selections(descriptor, {})
    assert result.termination_reasons == {1: ('world', 'termination', 'collision'), 2: ('world', 'termination', 'deadline')}
    raw = result.model_dump_json()
    assert json.loads(raw)['termination_reasons'] == {'1': ['world', 'termination', 'collision'], '2': ['world', 'termination', 'deadline']}
    assert ResolvedPackageSelection.model_validate_json(raw) == result
    enabled = resolve_package_selections(descriptor, {'world': {'termination': {'deadline': {'enabled': True}}}})
    assert enabled.termination_reasons == result.termination_reasons


def test_all_owners_and_cost_uniqueness(definition, composition):
    base = definition.terms['progress']
    world = {**base.output_spec, 'shape': ['B']}
    definition.terms.update({
        'obs': replace(base, term_kind='observation'),
        'world_obs': replace(base, term_kind='observation', scope='world', output_spec=world),
        'cost': replace(base, term_kind='cost', cost_id='energy'),
        'world_cost': replace(base, term_kind='cost', scope='world', output_spec=world, cost_id='energy'),
        'world_reward': replace(base, scope='world', output_spec=world),
    })
    for role in ('scout', 'guard'):
        composition['agent_roles'][role]['observation_terms'] = [{'selection_id': 'position', 'term': 'obs', 'config': {'scale': 1.0}}]
        composition['agent_roles'][role]['cost_terms'] = [{'selection_id': 'energy', 'term': 'cost', 'config': {'scale': 1.0}}]
    composition['critic_inputs'] = {'team': {'observation_terms': [
        {'selection_id': 'scout', 'term': 'obs', 'role_id': 'scout', 'config': {'scale': 1.0}},
        {'selection_id': 'world', 'term': 'world_obs', 'config': {'scale': 1.0}},
    ]}}
    composition['world_cost_terms'] = [{'selection_id': 'energy', 'term': 'world_cost', 'config': {'scale': 1.0}}]
    composition['world_reward_terms'] = [{'selection_id': 'reward', 'term': 'world_reward', 'config': {'scale': 1.0}}]
    descriptor = build_descriptor(definition, composition, 'a' * 64)
    result = resolve_package_selections(descriptor, {})
    critic = [s for s in result.selections if s.owner_path == ('critic_inputs', 'team')]
    assert [s.binding_role_id for s in critic] == ['scout', None]
    costs = [s for s in result.selections if s.term_kind == 'cost']
    assert len(costs) == 3
    rows = composition['agent_roles']['scout']['cost_terms']
    rows.append({'selection_id': 'duplicate', 'term': 'cost', 'config': {'scale': 1.0}, 'default_enabled': False})
    descriptor = build_descriptor(definition, composition, 'a' * 64)
    resolve_package_selections(descriptor, {})
    with pytest.raises(ConfigError):
        resolve_package_selections(descriptor, {'roles': {'scout': {'cost': {'duplicate': {'enabled': True}}}}})


def test_critic_role_binding_is_required(definition, composition):
    definition.terms['obs'] = replace(definition.terms['progress'], term_kind='observation')
    composition['critic_inputs'] = {'team': {'observation_terms': [{'selection_id': 'x', 'term': 'obs'}]}}
    with pytest.raises(ConfigError):
        build_descriptor(definition, composition, 'a' * 64)
