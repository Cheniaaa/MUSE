"""真实训练包和资源往返，测试拓扑参数不进入平台默认值。"""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from flight_helpers import EXAMPLE, add_fixed_anchors, archive


PACKAGE = Path(__file__).parent / 'fixtures' / 'wireless_package'


def wireless_catalog(root, *, worlds=2, services=True, sensing=True, background_only=False, sensing_only=False,
                     direction='uplink', joint_markov=False, transmission_sensing=False):
    from muse_server.catalog import Catalog
    metadata = root / 'builtin.json'
    subprocess.run([sys.executable, '-m', 'muse_builtins.metadata', '--output', str(metadata),
                    '--controller', str(EXAMPLE / 'controller.json')], check=True)
    catalog = Catalog(root / 'catalog', metadata)
    env = root / 'environment'
    env.mkdir()
    (env / 'environment.yaml').write_text(yaml.safe_dump(dict(scene='scene.xml', rf_geometry=dict(rf_blocking=['wall']))))
    (env / 'scene.xml').write_text('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="20 20 .1"/>
      <geom name="wall" type="box" size=".1 1 1" pos="8 0 1" contype="0" conaffinity="0"/>
    </worldbody></mujoco>''')
    environment = catalog.register_archive('environment', archive(env, root/'env.zip'))
    package = root / 'package'
    shutil.copytree(PACKAGE, package)
    package_config = yaml.safe_load((package/'training_package.yaml').read_text())
    composition = package_config['compositions']['default']
    composition['algorithm_config']['world_power_scales'] = [1./(i+1)**2 for i in range(worlds)]
    if not services:
        composition['world_termination_terms'] = composition['world_termination_terms'][1:]
    if not sensing:
        del composition['model_bindings']['sensing']
    if background_only or sensing_only:
        services = False
        composition['world_termination_terms'] = [composition['world_termination_terms'][-1]]
        for role in composition['agent_roles'].values():
            del role['action_interfaces']['radio']
        composition['model_bindings'] = {} if background_only else {
            'sensing': {'default': 'synthetic_sensing', 'options': {'synthetic_sensing': {}}}}
    (package/'training_package.yaml').write_text(yaml.safe_dump(package_config, sort_keys=False))
    request = json.loads((EXAMPLE/'request.json').read_text())
    request.update(environment_id=environment.resource_id, environment_revision=environment.revision,
        composition_id='default', algorithm_config={}, topology=dict(world_count=worlds))
    scenario = request['scenario']
    scenario['scene'].update(control_substeps=10, physics_dt=.001)
    # IMU仍沿既有rate/整数tick契约；不同dt只改变冻结period_ticks。
    scenario['entity_groups'][1]['count'] = 1
    scenario['placements']['guard_start'] = [scenario['placements']['guard_start'][0]]
    scenario['command_targets_world']['guard'] = [scenario['command_targets_world']['guard'][0]]
    request['role_config']['guard']['slots'] = [0]
    add_fixed_anchors(request)
    scenario['placements']['anchor_start'] = [dict(position=[4., 0., 1.], orientation_wxyz=[1., 0., 0., 0.]),
                                             dict(position=[4., 3., 1.], orientation_wxyz=[1., 0., 0., 0.])]
    scenario['radio_geometry'] = dict(endpoints={
        'tx': dict(group_id='scout_group', slots=[0], device_id='ideal_point_radio'),
        'peer': dict(group_id='guard_group', slots=[0], device_id='ideal_point_radio'),
        'rx': dict(group_id='anchor_group', slots=[0], device_id='ideal_point_radio'),
        'jammer': dict(group_id='anchor_group', slots=[1], device_id='ideal_point_radio')},
        links=[dict(link_id='up', tx='tx', rx='rx'), dict(link_id='peer_up', tx='peer', rx='rx')],
        sensing=[dict(sensing_id='echo', tx='tx', rx='rx', target=dict(group_id='guard_group', slots=[0]))] if sensing else [])
    scenario['network'] = dict(resources=[dict(resource_id='band_a', channel_id=7, center_frequency_hz=1e9, bandwidth_hz=1e6, sharing='reuse'),
            dict(resource_id='band_b', channel_id=-3, center_frequency_hz=2e9, bandwidth_hz=2e6, sharing='reuse')],
        epoch_max_ticks=4, endpoint_limits={name: dict(resource_ids=['band_a', 'band_b'], max_power_w=2.) for name in ('tx', 'peer', 'rx', 'jammer')},
        role_controls={'scout': 'tx', 'guard': 'peer'}, passive_endpoints=['rx'],
        background_emitters=[dict(emitter_id='noise', endpoints=['jammer'], behavior='fixed',
            config=dict(active=[True], power_w=[.5], channel_id=[7]))],
        communication_allocations=[dict(link_id=name, resource_ids=['band_a', 'band_b']) for name in ('up', 'peer_up')],
        services=[dict(service_id='upload', link_id='up', resource_ids=['band_a', 'band_b'], initial_bits=1.)] if services else [])
    request['model_config'] = {'radio': {'rate_scale': 1000.}}
    if sensing:
        request['model_config']['sensing'] = {'increment_scale': 3.}
    if not services and not background_only and not sensing_only:
        # 显式world评分需求激活link-only模型，不能靠候选binding偷偷执行。
        composition['world_reward_terms'] = [dict(selection_id='radio_valid', term='radio_valid')]
    if sensing:
        composition.setdefault('world_reward_terms', []).append(dict(selection_id='sensing_valid', term='sensing_valid'))
    if background_only or sensing_only:
        scenario['network']['role_controls'] = {}
        scenario['network']['passive_endpoints'] = ['rx', 'peer']
        scenario['network']['background_emitters'].append(dict(emitter_id='source', endpoints=['tx'], behavior='fixed',
            config=dict(active=[True], power_w=[1.], channel_id=[7])))
        scenario['network']['services'] = []
        scenario['network']['communication_allocations'] = []
        scenario['radio_geometry']['links'] = []
        request['model_config'] = {} if background_only else {'sensing': {'increment_scale': 3.}}
        if background_only:
            scenario['radio_geometry']['sensing'] = []
            composition.pop('world_reward_terms', None)
    if direction != 'uplink':
        # 运动仍归role，RF改由显式背景源，证明两类控制所有权独立。
        for role in composition['agent_roles'].values():
            del role['action_interfaces']['radio']
        tx, rx = ('rx', 'tx') if direction == 'downlink' else ('peer', 'tx')
        scenario['radio_geometry']['links'] = [dict(link_id='up', tx=tx, rx=rx)]
        scenario['network']['role_controls'] = {}
        scenario['network']['passive_endpoints'] = [name for name in ('tx', 'peer', 'rx') if name != tx]
        scenario['network']['background_emitters'].append(dict(emitter_id='moving_or_fixed_source', endpoints=[tx],
            behavior='fixed', config=dict(active=[True], power_w=[1.], channel_id=[7])))
        scenario['network']['communication_allocations'] = [dict(link_id='up', resource_ids=['band_a', 'band_b'])]
    if joint_markov:
        for role in composition['agent_roles'].values():
            role['action_interfaces'].pop('radio', None)
        scenario['network']['role_controls'] = {}
        scenario['network']['background_emitters'] = [dict(emitter_id='joint', endpoints=['tx', 'peer', 'jammer'],
            behavior='markov', config=dict(states=[
                dict(active=[True, True, True], power_w=[1., .5, .25], channel_id=[7, -3, 7]),
                dict(active=[True, True, False], power_w=[.5, 1., 0.], channel_id=[-3, 7, -3])],
                transition_matrix=[[0., 1.], [1., 0.]], dwell_ticks=5, initial_state=0))]
    if transmission_sensing:
        composition['model_bindings']['sensing'] = {
            'default': 'transmission_sensing', 'options': {'transmission_sensing': {}}}
    # 按最终composition真实登记，CPU descriptor不创建任何模型。
    (package/'training_package.yaml').write_text(yaml.safe_dump(package_config, sort_keys=False))
    record = catalog.register_archive('training_package', archive(package, root/'pkg-final.zip'))
    request.update(training_package_id=record.resource_id, training_package_revision=record.revision)
    return catalog, request


def wireless_plan(root, **kwargs):
    from muse_server.resolver import resolve_experiment
    catalog, request = wireless_catalog(root, **kwargs)
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment)
    return catalog, catalog.create_run_plan(identity, mode='EVALUATE', device='cuda:0', max_decision_steps=6)
