"""真实归档、metadata及Catalog的CPU接线。"""
import json
import math
import subprocess
import sys

import pytest
from muse_sdk.run_protocol import RunRequest

from muse_server.resolver import resolve_experiment
from reference_scenario_helpers import reference_catalog


def test_benchmark_variants_resolve_with_real_catalog(tmp_path):
    from examples.reference_benchmark import build_variants
    from reference_scenario_helpers import REFERENCE_EXAMPLES
    catalog, _ = reference_catalog(tmp_path / 'setup', scenario='isac_cost', algorithm='mappo',
                                    costs=True, history_length=0, worlds=2)
    for index, variant in enumerate(build_variants(REFERENCE_EXAMPLES)):
        request = variant['request']
        for key, kind in [('environment', 'environment'), ('package', 'training_package')]:
            path = tmp_path / f'{index}-{key}.zip'
            path.write_bytes(variant[key])
            resource = catalog.register_archive(kind, path)
            request[kind + '_id'], request[kind + '_revision'] = resource.resource_id, resource.revision
        report = resolve_experiment(request, catalog.snapshot())
        assert report.valid, (variant['name'], report.errors)
        identity = catalog.save_experiment(report.resolved_experiment, source=request)
        initial = catalog.create_run_plan(RunRequest(experiment_id=identity, mode='TRAIN', max_decision_steps=1), device='cpu')
        assert initial.experiment.training_schedule.collection_steps == 32


@pytest.mark.parametrize('budget', [1, 35])
def test_partial_budget_run_uses_unchanged_collection_horizon(tmp_path, budget):
    catalog, request = reference_catalog(tmp_path, scenario='hybrid_history_markov', algorithm='ippo',
                                         costs=False, history_length=4, worlds=2)
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment, source=request)
    plan = catalog.create_run_plan(RunRequest(experiment_id=identity, mode='TRAIN', max_decision_steps=budget), device='cpu')
    assert plan.experiment.training_schedule.collection_steps == 32
    assert plan.max_decision_steps == budget


@pytest.mark.parametrize('scenario', ['isac_cost', 'hybrid_history_markov'])
def test_reference_request_explicitly_freezes_the_training_budget_inputs(scenario):
    from reference_scenario_helpers import REFERENCE_EXAMPLES
    request = json.loads((REFERENCE_EXAMPLES / scenario / 'request_train.json').read_text())
    assert request['topology']['world_count'] == 16 and request['root_seed'] == 7
    expected = dict(
        collection_steps=32, hidden_dims=[64, 64], learning_rate=.0003,
        num_learning_epochs=4, num_mini_batches=4, gamma=.99, lam=.95, clip_param=.2,
        value_loss_coef=1, entropy_coef=.01, max_grad_norm=1)
    assert {key: request['algorithm_config'].get(key) for key in expected} == expected


@pytest.mark.parametrize('algorithm,costs', [('mappo', True), ('mappo', False), ('ippo', False)])
def test_isac_real_catalog_shapes_services_and_scopes(tmp_path, algorithm, costs):
    catalog, request = reference_catalog(tmp_path, scenario='isac_cost', algorithm=algorithm,
                                         costs=costs, history_length=0, worlds=2)
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    exp = report.resolved_experiment
    for fields in exp.algorithm_inputs.actors.values():
        assert sum(math.prod(field.tensor.shape[2:]) for field in fields) == 29
    assert [service.service_id for service in exp.network.services] == ['alpha_upload', 'beta_upload']
    assert [service.initial_bits for service in exp.network.services] == [4e9, 4e9]
    assert [len(service.link_indices) for service in exp.network.services] == [1, 1]
    assert exp.network.epoch_offsets == (0, 7, 14, 20)
    assert bool(exp.costs.world) is costs
    rewards = [selection for selection in exp.selections if selection.enabled and selection.term_kind == 'reward']
    assert [selection.owner_path for selection in rewards] == ([('world',)] if algorithm == 'mappo' else
                                                               [('roles', 'alpha'), ('roles', 'beta')])
    if algorithm == 'mappo':
        critic_fields = exp.algorithm_inputs.critics['team']
        assert sum(math.prod(field.tensor.shape[1:]) for field in critic_fields) == 58
    else:
        assert exp.algorithm_inputs.critics == {}
    from muse_sdk.experiment import ExperimentDefinition, learning_contract
    assert ExperimentDefinition.model_validate_json(exp.model_dump_json()) == exp
    assert learning_contract(exp)['costs'] == exp.costs.model_dump(mode='json')
    assert not next(selection for selection in exp.selections if selection.term_kind == 'termination').time_out
    request['composition_id'] = 'missing'
    rejected = resolve_experiment(request, catalog.snapshot())
    assert not rejected.valid


@pytest.mark.parametrize('length,width', [(1, 30), (4, 66)])
def test_history_catalog_has_local_history_and_empty_costs(tmp_path, length, width):
    catalog, request = reference_catalog(tmp_path, scenario='hybrid_history_markov', algorithm='ippo',
                                         costs=False, history_length=length, worlds=2)
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    exp = report.resolved_experiment
    fields = exp.algorithm_inputs.actors['pilot']
    assert sum(math.prod(field.tensor.shape[2:]) for field in fields) == width
    assert fields[-1].tensor.shape == [2, 1, length, 3, 4]
    assert exp.costs.world == exp.costs.roles == {}
    assert not any(selection.term_kind == 'cost' for selection in exp.selections)
    emitter = exp.network.background_emitters[0]
    assert emitter.behavior == 'markov' and emitter.config.dwell_ticks == 20
    assert [resource.channel_id for resource in exp.network.resources] == [7, -3, 19]
    matrix = emitter.config.transition_matrix
    # 当前resource0无干扰，即条件化到当前state为1/2；不是无条件两步转移。
    for previous, expected in ((1, .04), (2, .2)):
        probability = sum(matrix[previous][state] * matrix[state][0] for state in (1, 2))
        probability /= sum(matrix[previous][state] for state in (1, 2))
        assert probability == pytest.approx(expected)


@pytest.mark.parametrize('scenario,algorithm,costs', [('isac_cost', 'mappo', True),
                                                   ('hybrid_history_markov', 'ippo', False)])
def test_reference_register_describe_resolve_does_not_import_backends(tmp_path, scenario, algorithm, costs):
    catalog, request = reference_catalog(tmp_path, scenario=scenario, algorithm=algorithm,
                                         costs=costs, history_length=4, worlds=2)
    (tmp_path / 'request.json').write_text(json.dumps(request))
    code = '''import json,sys,importlib,yaml
from pathlib import Path
from muse_sdk.descriptor import build_descriptor
from muse_server.catalog import Catalog
from muse_server.resolver import resolve_experiment
root=Path(sys.argv[1])
scenario=sys.argv[2]
module=importlib.import_module('examples.'+scenario+'.training_package')
package_config=yaml.safe_load((Path('examples')/scenario/'training_package/training_package.yaml').read_text())
catalog=Catalog(root/'catalog',root/'builtin.json')
build_descriptor(module.build_training_package(),package_config['compositions'],'a'*64,
    catalog.snapshot().builtins.descriptor,default_composition_id=package_config['default_composition_id'])
report=resolve_experiment(json.loads((root/'request.json').read_text()),catalog.snapshot())
assert report.valid,report.errors
assert not {'torch','warp'} & sys.modules.keys()
'''
    subprocess.run([sys.executable, '-c', code, str(tmp_path), scenario], check=True)


@pytest.mark.parametrize('scenario,algorithm', [('isac_cost', 'mappo'), ('hybrid_history_markov', 'ippo')])
def test_real_layout_resource_order_and_slot_counts_are_not_hardcoded(tmp_path, scenario, algorithm):
    from types import SimpleNamespace as NS
    import torch
    from importlib import import_module

    catalog, request = reference_catalog(tmp_path, scenario=scenario, algorithm=algorithm,
                                         costs=False, history_length=4, worlds=3)
    group = request['scenario']['entity_groups'][0]
    old_count = group['count']
    group['count'] *= 2
    positions = request['scenario']['placements']['uav_start']
    positions.extend([dict(position=[position['position'][0], 1, 1], orientation_wxyz=[1, 0, 0, 0])
                      for position in positions[:old_count]])
    for index, role in enumerate(request['role_config']):
        request['role_config'][role]['slots'].append(index + old_count)
        request['scenario']['command_targets_world'][role] *= 2
        endpoint = request['scenario']['network']['role_controls'][role]
        request['scenario']['radio_geometry']['endpoints'][endpoint]['slots'].append(index + old_count)
    request['scenario']['network']['resources'].reverse()
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    exp = report.resolved_experiment
    env = NS(experiment=exp, num_envs=3, device='cpu')
    cls = getattr(import_module(f'examples.{scenario}.training_package.task'),
                  'IsacTask' if scenario == 'isac_cost' else 'HistoryTask')
    task = cls(exp.task_config, env)
    assert task.endpoints.numel() == old_count * 2
    assert task.failure_amplitude == pytest.approx(2)
    for fields in exp.algorithm_inputs.actors.values():
        assert all(field.tensor.shape[:2] == [3, 2] for field in fields)
    for service in exp.network.services:
        assert len(service.link_indices) == 2 and service.initial_bits == 4e9
    assert task.remaining_fraction.dtype == torch.float32
