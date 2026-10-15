import copy

import pytest


def test_resolved_cost_layout_preserves_units_and_rejects_malformed_state():
    from pydantic import ValidationError
    from muse_sdk.experiment import CostLayout

    metadata = dict(cost_id='energy', scope='role', units='J', statistic='episode_discounted_sum',
                    boundary='world_or_learning_done', discount=.9, budget=2.,
                    lambda_initial=.1, lambda_learning_rate=.01)
    raw = {'roles': {'scout': {'energy': metadata}}, 'world': {}}
    layout = CostLayout.model_validate(raw)
    assert CostLayout.model_validate_json(layout.model_dump_json()).roles['scout']['energy'].units == 'J'
    for change in (dict(discount='0.9'), dict(scope='world'), dict(boundary='world'),
                   dict(cost_id='other'), dict(statistic='mean'), dict(unknown=True)):
        invalid = copy.deepcopy(raw)
        invalid['roles']['scout']['energy'].update(change)
        with pytest.raises(ValidationError):
            CostLayout.model_validate(invalid)
    with pytest.raises(ValidationError):
        CostLayout.model_validate({'roles': {}, 'world': []})


def test_algorithm_cost_capabilities_are_part_of_descriptor():
    from muse_sdk.descriptor import AlgorithmDescriptor
    assert 'cost_inputs' in AlgorithmDescriptor.model_fields


@pytest.mark.parametrize('dtype', ['float64', 'int32', 'int64', 'bool'])
def test_cost_descriptor_rejects_dtype_not_supported_by_runtime(dtype):
    from pydantic import ValidationError
    from muse_sdk.descriptor import TermDescriptor

    term = dict(term_kind='cost', scope='world', cost_id='energy', config_schema={}, input_fields=[],
                output_spec=dict(shape=['B'], dtype='float32', units='J', frame='none', rate='decision'))
    assert TermDescriptor.model_validate(term).output_spec.dtype == 'float32'
    term['output_spec']['dtype'] = dtype
    with pytest.raises(ValidationError, match='float32'):
        TermDescriptor.model_validate(term)


def _enable_costs(pipeline):
    catalog, request = pipeline
    request = copy.deepcopy(request)
    snapshot = catalog.snapshot()
    snapshot.builtins.descriptor.algorithms['cost_test'] = snapshot.builtins.descriptor.algorithms['ippo'].model_copy(
        update={'algorithm_id': 'cost_test', 'cost_inputs': ['role', 'world']})
    request['algorithm_id'] = 'builtin:cost_test'
    package = next(record for record in snapshot.resources
                   if record.resource_id == request['training_package_id']
                   and record.revision == request['training_package_revision'])
    composition = package.metadata['composition']
    composition['algorithm'] = request['algorithm_id']
    composition['algorithm_config'] = {}
    builtin_terms = snapshot.builtins.descriptor.terms
    role_cost = builtin_terms['distance'].model_copy(update={
        'term_kind': 'cost', 'cost_id': 'energy',
    })
    builtin_terms['distance'] = role_cost
    for role in composition['agent_roles'].values():
        role['cost_terms'] = role.pop('reward_terms')
        for selection in role['cost_terms']:
            selection.pop('weight', None)
    world_cost = role_cost.model_copy(update={
        'scope': 'world',
        'output_spec': role_cost.output_spec.model_copy(update={'shape': ['B']}),
        'input_fields': [],
        'cost_id': 'team_energy',
    })
    builtin_terms['team_energy'] = world_cost
    composition['world_cost_terms'] = [{
        'selection_id': 'team_energy', 'term': 'builtin:team_energy', 'config': {'scale': 1.0},
    }]
    return catalog, request, snapshot


def _configured_request(request):
    result = copy.deepcopy(request)
    result['cost_config'] = {
        'roles': {
            'scout': {'energy': {
                'discount': .98, 'budget': 1.5, 'lambda_initial': .2,
                'lambda_learning_rate': .01,
            }},
            'guard': {'energy': {
                'discount': .97, 'budget': 2.5, 'lambda_initial': .3,
                'lambda_learning_rate': .02,
            }},
        },
        'world': {'team_energy': {
            'discount': .99, 'budget': 3.0, 'lambda_initial': .4,
            'lambda_learning_rate': .03,
        }},
    }
    return result


def test_cost_remains_optional(pipeline):
    from muse_server.resolver import resolve_experiment

    catalog, request = pipeline
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    assert report.resolved_experiment.costs.model_dump() == {'roles': {}, 'world': {}}


def test_enabled_cost_requires_explicit_constraint(pipeline):
    from muse_server.resolver import resolve_experiment

    catalog, request, snapshot = _enable_costs(pipeline)
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert report.resolved_experiment is None


def test_role_and_world_cost_constraints_resolve_and_round_trip(pipeline):
    from muse_server.resolver import resolve_experiment
    from muse_sdk.experiment import ExperimentDefinition

    catalog, request, snapshot = _enable_costs(pipeline)
    request = _configured_request(request)
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors
    costs = report.resolved_experiment.costs.model_dump()
    assert costs['roles']['scout']['energy'] == {
        'cost_id': 'energy', 'scope': 'role', 'discount': .98, 'budget': 1.5,
        'lambda_initial': .2, 'lambda_learning_rate': .01,
        'units': 'm', 'statistic': 'episode_discounted_sum', 'boundary': 'world_or_learning_done',
    }
    assert costs['world']['team_energy']['scope'] == 'world'
    assert costs['world']['team_energy']['boundary'] == 'world'
    encoded = report.resolved_experiment.model_dump_json()
    assert ExperimentDefinition.model_validate_json(encoded).model_dump_json() == encoded


@pytest.mark.parametrize('mutation', [
    lambda item: item.update(discount=1.1),
    lambda item: item.update(discount=-.1),
    lambda item: item.update(budget=-1.0),
    lambda item: item.update(lambda_initial=-.1),
    lambda item: item.update(lambda_learning_rate=0.0),
])
def test_invalid_cost_constraint_is_rejected(pipeline, mutation):
    from muse_server.resolver import resolve_experiment

    catalog, request, snapshot = _enable_costs(pipeline)
    request = _configured_request(request)
    mutation(request['cost_config']['roles']['scout']['energy'])
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert any('cost_config' in error['field'] for error in report.errors), report.errors


def test_cost_config_must_match_enabled_costs(pipeline):
    from muse_server.resolver import resolve_experiment

    catalog, request, snapshot = _enable_costs(pipeline)
    request = _configured_request(request)
    request['cost_config']['world']['unused'] = request['cost_config']['world'].pop('team_energy')
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    assert any('cost_config' in error['field'] for error in report.errors), report.errors


@pytest.mark.parametrize('supported', [[], ['role'], ['world']])
def test_algorithm_rejects_unconsumed_cost_scope(pipeline, supported):
    from muse_server.resolver import resolve_experiment

    _, request, snapshot = _enable_costs(pipeline)
    snapshot.builtins.descriptor.algorithms['cost_test'].cost_inputs = supported
    report = resolve_experiment(_configured_request(request), snapshot)
    assert not report.valid
    assert any('cost' in error['field'] and 'consume' in error['message'] for error in report.errors)


def test_ippo_world_cost_is_rejected_and_role_cost_is_accepted(pipeline):
    from muse_server.resolver import resolve_experiment

    _, request, snapshot = _enable_costs(pipeline)
    request = _configured_request(request)
    request['algorithm_id'] = 'builtin:ippo'
    package = next(record for record in snapshot.resources if record.resource_id == request['training_package_id'])
    package.metadata['composition']['algorithm'] = request['algorithm_id']
    report = resolve_experiment(request, snapshot)
    assert not report.valid
    request['term_config'] = {'world': {'cost': {'team_energy': {'enabled': False}}}}
    request['cost_config']['world'] = {}
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors
