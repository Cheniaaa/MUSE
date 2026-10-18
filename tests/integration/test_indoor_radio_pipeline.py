"""示例按真实注册/CPU解析接入builtin，不以候选模型冒充激活。"""
from copy import deepcopy
import json
import subprocess
import sys

import pytest
import yaml

from flight_helpers import archive
from indoor_radio_helpers import indoor_radio_catalog, indoor_radio_plan
from muse_sdk.experiment import RunPlan
from muse_server.resolver import resolve_experiment


def test_example_resolves_builtin_without_services(tmp_path):
    catalog, plan = indoor_radio_plan(tmp_path)
    exp = plan.experiment
    assert set(exp.model_bindings) == {'radio'}
    assert exp.model_bindings['radio'].reference == 'builtin:indoor_office_radio'
    assert {'radio', 'geometry'} <= exp.activation.keys()
    assert exp.network.services == ()
    assert exp.network.epoch_offsets == (0, 7, 14, 20)
    assert [r.channel_id for r in exp.network.resources] == [7, -3]
    assert exp.network.link_count == 2
    assert RunPlan.model_validate_json(plan.model_dump_json()) == plan


@pytest.mark.parametrize('model', ['indoor_office_radio', 'free_space_radio'])
def test_service_variant_and_free_space_comparator_resolve(tmp_path, model):
    _, plan = indoor_radio_plan(tmp_path, model=model, service=True)
    exp = plan.experiment
    assert exp.network.services[0].initial_bits > 0
    assert exp.algorithm_config['world_power_scales'] == [1., .25]
    assert any(s.enabled and s.selection_id == 'done' for s in exp.selections)
    assert exp.model_bindings['radio'].config['receiver_noise_temperature_k'] == 290.
    if model == 'free_space_radio':
        assert exp.model_bindings['radio'].config['minimum_distance_m'] == 1.


@pytest.mark.parametrize('direction', ['downlink', 'mobile_to_mobile'])
def test_example_does_not_limit_topology_or_direction(tmp_path, direction):
    catalog, request = indoor_radio_catalog(tmp_path, worlds=3)
    scenario = request['scenario']
    group = scenario['entity_groups'][0]
    group['count'] = 3
    scenario['placements']['uav_start'].append(dict(position=[0., -8., 1.], orientation_wxyz=[1., 0., 0., 0.]))
    request['role_config']['pilot']['slots'] = [2, 0, 1]
    scenario['command_targets_world']['pilot'] = [[0., -4., 1.], [0., 3., 1.], [0., -1., 1.]]
    geometry, network = scenario['radio_geometry'], scenario['network']
    del geometry['endpoints']['uavs']
    geometry['endpoints']['receiver'] = dict(group_id='uav_group', slots=[1], device_id='ideal_point_radio')
    geometry['endpoints']['senders'] = dict(group_id='uav_group', slots=[2, 0], device_id='ideal_point_radio')
    geometry['links'] = [dict(link_id='reverse', tx='station' if direction == 'downlink' else 'senders', rx='receiver')]
    # 集合不重叠；运动仍归role，RF用独立显式emitter，沿既有唯一归属契约。
    budget = network['endpoint_limits'].pop('uavs')
    network['endpoint_limits'].update(senders=deepcopy(budget), receiver=deepcopy(budget))
    network['role_controls'] = {}
    network['passive_endpoints'] = ['receiver', 'senders' if direction == 'downlink' else 'station']
    count = 1 if direction == 'downlink' else 2
    network['background_emitters'].append(dict(emitter_id='reverse_tx',
        endpoints=['station' if direction == 'downlink' else 'senders'], behavior='fixed',
        config=dict(active=[True]*count, power_w=[.001]*count, channel_id=[7]*count)))
    package = tmp_path / 'package'
    composition = yaml.safe_load((package / 'training_package.yaml').read_text())
    del composition['compositions']['default']['agent_roles']['pilot']['action_interfaces']['radio']
    (package / 'training_package.yaml').write_text(yaml.safe_dump(composition))
    record = catalog.register_archive('training_package', archive(package, tmp_path / 'reverse.zip'))
    request.update(training_package_id=record.resource_id, training_package_revision=record.revision)
    network['resources'].append(dict(resource_id='band_c', channel_id=19, center_frequency_hz=2.462e9,
                                      bandwidth_hz=5e6, sharing='reuse'))
    for limit in network['endpoint_limits'].values():
        limit['resource_ids'].append('band_c')
    network['communication_allocations'] = [dict(link_id='reverse', resource_ids=['band_a', 'band_b', 'band_c'])]
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    exp = report.resolved_experiment
    assert exp.bindings['pilot'].role_slot_to_group_slot == [2, 0, 1]
    assert exp.network.link_count == (1 if direction == 'downlink' else 2)
    assert len(exp.network.resources) == 3


def test_server_resolve_keeps_model_import_isolated(tmp_path):
    catalog, request = indoor_radio_catalog(tmp_path)
    (tmp_path / 'request.json').write_text(json.dumps(request))
    code = '''import json,sys
from pathlib import Path
from muse_server.catalog import Catalog
from muse_server.resolver import resolve_experiment
root=Path(sys.argv[1])
catalog=Catalog(root/'catalog',root/'builtin.json')
report=resolve_experiment(json.loads((root/'request.json').read_text()),catalog.snapshot())
assert report.valid,report.errors
assert not {'torch','warp','muse_builtins.models.analytical_radio'} & sys.modules.keys()
'''
    subprocess.run([sys.executable, '-c', code, str(tmp_path)], check=True)
