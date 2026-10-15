"""真实包导出/Catalog/CPU resolve；不构造物理运行时。"""
import json
import subprocess
import sys

import pytest

from muse_sdk.experiment import RunPlan
from muse_server.resolver import resolve_experiment


@pytest.mark.parametrize('model,measurement,fim', [
    ('free_space_path_length_measurement', True, False),
    ('free_space_position_efim_2d', False, True),
    ('free_space_path_length_efim_2d', True, True),
])
def test_three_capabilities_resolve_and_run_plan_roundtrip(tmp_path, model, measurement, fim):
    from sensing_package_helpers import sensing_plan
    _, plan = sensing_plan(tmp_path, model=model)
    exp = plan.experiment
    descriptor = exp.model_bindings['sensing'].descriptor
    assert (descriptor.output_spec.measurement is not None) == measurement
    assert (descriptor.output_spec.fim is not None) == fim
    if measurement:
        assert descriptor.output_spec.measurement.time_semantics == 'interval_sample'
    if fim:
        assert descriptor.output_spec.fim.time_semantics == 'increment'
        cost = exp.costs.world['position_bound_violation']
        assert (cost.discount, cost.budget, cost.lambda_initial, cost.lambda_learning_rate) == (.99, 10., 0., .01)
        assert cost.statistic == 'episode_discounted_sum'
    else:
        assert exp.costs.world == {}
    assert exp.network.epoch_offsets == (0, 7, 14, 20)
    assert exp.network.link_count == 2 and exp.network.sensing_count == 2
    assert exp.algorithm_inputs.actors == {}
    assert [phase['duration_decisions'] for phase in exp.algorithm_config['phases']] == [50]*4
    assert RunPlan.model_validate_json(plan.model_dump_json()) == plan
    from muse_sdk.experiment import learning_contract
    contract = learning_contract(exp)
    assert contract['model_bindings']['sensing']['descriptor']['output_spec'] == descriptor.model_dump(mode='json')['output_spec']
    for field, original in zip(contract['model_bindings']['sensing']['input_fields'],
                               exp.model_bindings['sensing'].input_fields, strict=True):
        expected = original.tensor.shape if original.path[:2] == ['network', 'resources'] else original.tensor.shape[1:]
        assert field['tensor']['shape'] == expected


def test_sensing_without_consumers_is_not_activated(tmp_path):
    from sensing_package_helpers import sensing_catalog
    catalog, request = sensing_catalog(tmp_path)
    request['term_config'] = {'roles': {role: {'observation': {'measurement': {'enabled': False}}}
                                      for role in ('alpha', 'beta')},
                              'world': {'cost': {'position_bound': {'enabled': False}}}}
    request['cost_config'] = {'roles': {}, 'world': {}}
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    assert 'sensing' not in report.resolved_experiment.model_bindings
    assert 'sensing' not in report.resolved_experiment.activation


def test_sensing_only_with_zero_links_and_no_radio(tmp_path):
    from sensing_package_helpers import sensing_catalog
    catalog, request = sensing_catalog(tmp_path, jammer=False)
    request['scenario']['radio_geometry']['links'] = []
    request['scenario']['network']['communication_allocations'] = []
    request['term_config'] = {'world': {'reward': {'rate': {'enabled': False}}}}
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    exp = report.resolved_experiment
    assert exp.network.link_count == 0
    assert set(exp.model_bindings) == {'sensing'}
    assert 'radio' not in exp.activation


def test_server_resolve_import_isolation(tmp_path):
    from sensing_package_helpers import sensing_catalog
    _, request = sensing_catalog(tmp_path)
    (tmp_path / 'request.json').write_text(json.dumps(request))
    code = '''import json,sys
from pathlib import Path
from muse_server.catalog import Catalog
from muse_server.resolver import resolve_experiment
root=Path(sys.argv[1])
catalog=Catalog(root/'catalog',root/'builtin.json')
report=resolve_experiment(json.loads((root/'request.json').read_text()),catalog.snapshot())
assert report.valid,report.errors
assert not {'torch','warp','muse_builtins.models.sensing','muse_builtins.models.analytical_radio'} & sys.modules.keys()
'''
    subprocess.run([sys.executable, '-c', code, str(tmp_path)], check=True)


def test_actor_all_local_only_binds_own_measurement_and_rejects_raw_truth(tmp_path):
    from copy import deepcopy
    from sensing_package_helpers import sensing_catalog
    catalog, request = sensing_catalog(tmp_path)
    snapshot = catalog.snapshot()
    package = next(record for record in snapshot.resources if record.resource_id == request['training_package_id'])
    algorithm = package.metadata['algorithms']['sensing_reference_rule']
    algorithm['actor_observation'] = 'all_local'
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors
    for role, fields in report.resolved_experiment.algorithm_inputs.actors.items():
        assert [field.path for field in fields] == [['observations', 'actors', role, 'measurement']]
        assert fields[0].tensor.shape == [2, 1, 3]
    raw = next(selection.input_fields[0] for selection in report.resolved_experiment.selections
               if selection.enabled and selection.term_kind == 'cost').model_dump(mode='json')
    for actor_field in (raw, dict(path=['role_positions_world', 'alpha'], phase='post_physics',
                                 tensor=dict(shape=['B', 'N_role', 3], dtype='float32', units='m',
                                             frame='world_z_up', rate='decision'))):
        candidate = deepcopy(snapshot)
        record = next(record for record in candidate.resources if record.resource_id == request['training_package_id'])
        record.metadata['algorithms']['sensing_reference_rule']['actor_observation'] = {'alpha': [actor_field]}
        rejected = resolve_experiment(request, candidate)
        assert not rejected.valid
        assert any('Actor must only read its own observations' in error['message'] for error in rejected.errors)


def test_generalized_worlds_targets_bistatic_and_resource_descriptor(tmp_path):
    from sensing_package_helpers import sensing_catalog
    catalog, request = sensing_catalog(tmp_path, worlds=3)
    scenario = request['scenario']
    scenario['entity_groups'][1]['count'] = 4
    scenario['placements']['anchor_start'].append(dict(position=[2., 7., 0.], orientation_wxyz=[1., 0., 0., 0.]))
    for relation in scenario['radio_geometry']['sensing']:
        relation['target']['slots'] = [3, 2]
    scenario['radio_geometry']['sensing'][0]['rx'] = 'beta_rf'
    scenario['radio_geometry']['sensing'][1]['rx'] = 'alpha_rf'
    network = scenario['network']
    network['resources'].insert(0, dict(resource_id='band_c', channel_id=19,
        center_frequency_hz=2.462e9, bandwidth_hz=5e6, sharing='reuse'))
    for limit in network['endpoint_limits'].values():
        limit['resource_ids'].insert(0, 'band_c')
    for allocation in network['communication_allocations']:
        allocation['resource_ids'].insert(0, 'band_c')
    # 此Term明确限每slot唯一单站关系；关闭Term后科学descriptor不限制此拓扑。
    request['term_config'] = {'roles': {role: {'observation': {'measurement': {'enabled': False}}}
                                      for role in request['role_config']}}
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    exp = report.resolved_experiment
    assert exp.topology.world_count == 3
    assert exp.network.sensing_count == 4 and len(exp.network.resources) == 3
    assert [resource.channel_id for resource in exp.network.resources] == [19, 7, -3]
    assert exp.model_bindings['sensing'].descriptor.output_spec.fim.shape == ['B', 'S', 'F', 2, 2]
    from muse_builtins.wireless_terms import LocalPathLengthObservation
    from types import SimpleNamespace
    env = SimpleNamespace(experiment=exp, device='cpu')
    cfg = SimpleNamespace(params=dict(role_id='alpha', sensing_id='alpha_probe', range_scale_m=100.))
    with pytest.raises(ValueError, match='exactly one monostatic'):
        LocalPathLengthObservation(cfg, env)


@pytest.mark.parametrize('model', ['free_space_path_length_measurement', 'free_space_position_efim_2d'])
def test_missing_capability_is_rejected_without_implicit_model(tmp_path, model):
    from sensing_package_helpers import sensing_catalog
    catalog, request = sensing_catalog(tmp_path, model=model)
    request['term_config'] = {}
    request['cost_config'] = {'roles': {}, 'world': {'position_bound_violation': dict(
        discount=.99, budget=10., lambda_initial=0., lambda_learning_rate=.01)}}
    report = resolve_experiment(request, catalog.snapshot())
    assert not report.valid
    assert any('Unavailable public field' in error['message'] for error in report.errors)
