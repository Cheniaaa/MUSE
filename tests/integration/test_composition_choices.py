"""同一不可变包的组合/模型选择改变冻结语义，而不是改写源码包。"""
from copy import deepcopy
import json

import pytest

from flight_helpers import archive
from muse_server.resolver import resolve_experiment


def test_named_compositions_resolve_without_rewriting_package(pipeline, tmp_path):
    from reference_scenario_helpers import ISAC
    catalog, _ = pipeline
    environment = catalog.register_archive('environment', archive(ISAC / 'environment', tmp_path / 'env.zip'))
    package = catalog.register_archive('training_package', archive(ISAC / 'training_package', tmp_path / 'pkg.zip'))
    request = json.loads((ISAC / 'request_train.json').read_text())
    request.update(environment_id=environment.resource_id, environment_revision=environment.revision,
                   training_package_id=package.resource_id, training_package_revision=package.revision,
                   composition_id='mappo_cost_on')
    snapshot = catalog.snapshot()
    mappo = resolve_experiment(request, snapshot)
    assert mappo.valid, mappo.errors
    request.update(composition_id='ippo', algorithm_config={}, cost_config={})
    ippo = resolve_experiment(request, snapshot)
    assert ippo.valid, ippo.errors
    mappo, ippo = mappo.resolved_experiment, ippo.resolved_experiment
    assert ippo.training_package == mappo.training_package
    assert (ippo.algorithm_id, mappo.algorithm_id) == ('builtin:ippo', 'builtin:mappo')
    assert ippo.algorithm_inputs.critics == {} and set(mappo.algorithm_inputs.critics) == {'team'}
    assert [s.owner_path for s in ippo.selections if s.enabled and s.term_kind == 'reward'] == [('roles', 'alpha'), ('roles', 'beta')]
    assert [s.owner_path for s in mappo.selections if s.enabled and s.term_kind == 'reward'] == [('world',)]
    assert set(mappo.costs.world) == {'mission_position_violation'} and not ippo.costs.world
    request['composition_id'] = 'not_declared'
    invalid = resolve_experiment(request, snapshot)
    assert not invalid.valid and invalid.resolved_experiment is None
    assert invalid.errors[0]['field'] == 'composition_id'


def test_model_choice_is_declared_and_revalidated(pipeline):
    from wireless_helpers import wireless_request
    request, snapshot = wireless_request(pipeline)
    package = next(r for r in snapshot.resources if r.kind == 'training_package')
    composition = package.metadata['compositions']['default']
    package.metadata['radio_models']['link_only'] = {
        **deepcopy(package.metadata['radio_models']['oracle']), 'model_id': 'link_only',
        'output_spec': {'output_blocks': ['link']}}
    package.metadata['radio_models']['alternate'] = {
        **deepcopy(package.metadata['radio_models']['oracle']), 'model_id': 'alternate'}
    composition['model_bindings']['radio'] = {'default': 'oracle', 'options': {
        'oracle': {'config': {'scale': 2.}}, 'link_only': {'config': {}},
        'alternate': {'config': {'scale': 7.}}}}
    request['model_choices'] = {'radio': 'oracle'}
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors
    assert report.resolved_experiment.model_bindings['radio'].config['scale'] == 2.
    request['model_choices']['radio'] = 'alternate'
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors
    assert report.resolved_experiment.model_bindings['radio'].reference == 'alternate'
    assert report.resolved_experiment.model_bindings['radio'].config['scale'] == 7.
    request['model_config'] = {'radio': {'scale': 3.}}
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors
    assert report.resolved_experiment.model_bindings['radio'].config['scale'] == 3.
    request['model_choices']['radio'] = 'link_only'
    invalid = resolve_experiment(request, snapshot)
    assert not invalid.valid and any('rate' in error['message'] for error in invalid.errors)
    request['model_choices']['radio'] = 'undeclared'
    invalid = resolve_experiment(request, snapshot)
    assert not invalid.valid and invalid.errors[0]['field'] == 'model_choices.radio'


def test_selecting_model_does_not_activate_consumers(pipeline):
    from wireless_helpers import wireless_request
    request, snapshot = wireless_request(pipeline, service=False)
    request['scenario']['network'] = None
    request['model_choices'] = {'radio': 'oracle'}
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors
    assert report.resolved_experiment.network is None
    assert report.resolved_experiment.model_bindings == {}
    assert 'radio' not in report.resolved_experiment.activation
    request['model_choices']['radio'] = 'missing'
    assert not resolve_experiment(request, snapshot).valid


@pytest.mark.parametrize('old_field,value', [('task_id', None), ('algorithm_id', 'builtin:constant_velocity')])
def test_request_does_not_accept_redundant_old_selection(pipeline, old_field, value):
    catalog, request = pipeline
    request = deepcopy(request)
    request[old_field] = value
    report = resolve_experiment(request, catalog.snapshot())
    assert not report.valid
    assert any(e['field'] == old_field for e in report.errors)


def test_composition_label_does_not_change_learning_contract(pipeline):
    from muse_sdk.experiment import learning_contract
    catalog, request = pipeline
    snapshot = catalog.snapshot()
    package = next(r for r in snapshot.resources if r.kind == 'training_package')
    package.metadata['compositions']['equivalent'] = deepcopy(package.metadata['compositions']['default'])
    first = resolve_experiment(request, snapshot).resolved_experiment
    changed = {**request, 'composition_id': 'equivalent'}
    second = resolve_experiment(changed, snapshot).resolved_experiment
    assert first is not None and second is not None
    assert first.composition_id != second.composition_id
    assert learning_contract(first) == learning_contract(second)
