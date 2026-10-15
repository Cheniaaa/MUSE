"""CPU声明夹具；合成模型descriptor不代替产品积分oracle。"""
from copy import deepcopy

from flight_helpers import add_fixed_anchors


def resource(resource_id='band', channel_id=7, frequency=1e9, sharing='reuse'):
    return dict(resource_id=resource_id, channel_id=channel_id,
        center_frequency_hz=frequency, bandwidth_hz=1e6, sharing=sharing)


def wireless_request(pipeline, *, service=True, radio=True):
    catalog, source = pipeline
    request = deepcopy(source)
    request['scenario']['scene']['control_substeps'] = 10
    request['role_config']['guard']['slots'] = [1, 0]
    add_fixed_anchors(request)
    request['scenario']['radio_geometry'] = dict(endpoints={
        'tx': dict(group_id='guard_group', slots=[1, 0], device_id='ideal_point_radio'),
        'rx': dict(group_id='anchor_group', slots=[0], device_id='ideal_point_radio')},
        links=[dict(link_id='up', tx='tx', rx='rx')])
    request['scenario']['network'] = dict(resources=[resource()], epoch_max_ticks=4,
        endpoint_limits={'tx': dict(resource_ids=['band'], max_power_w=2.),
                         'rx': dict(resource_ids=['band'], max_power_w=2.)},
        role_controls={}, background_emitters=[dict(emitter_id='source', endpoints=['tx'], behavior='fixed',
            config=dict(active=[True, True], power_w=[1., 1.], channel_id=[7, 7]))],
        passive_endpoints=['rx'], communication_allocations=[dict(link_id='up', resource_ids=['band'])],
        services=[dict(service_id='payload', link_id='up', resource_ids=['band'], initial_bits=7.)] if service else [])
    snapshot = catalog.snapshot()
    environment = next(x for x in snapshot.resources if x.resource_id == request['environment_id'])
    environment.metadata['rf_geometry'] = {'rf_blocking': []}
    environment.metadata['geometry_summary'] = {'geoms': {}, 'flex_names': []}
    package = next(x for x in snapshot.resources if x.resource_id == request['training_package_id'])
    if radio:
        package.metadata['radio_models']['oracle'] = dict(model_id='oracle',
            config_schema=dict(type='object', properties={'scale': {'type': 'number', 'default': 1.}}, additionalProperties=False),
            input_fields=[dict(path=['network', 'allocations', 'link_power_w'], phase=phase,
                tensor=dict(shape=['B', 'L', 'F'], dtype='float32', units='W', frame='none', rate='radio_epoch'))
                for phase in ('reset', 'post_physics')], state_scope='world', state_schema={},
            diagnostics_schema={}, assumptions={'synthetic': True}, output_spec={'output_blocks': ['rate']})
        package.metadata['composition']['model_bindings'] = {'radio': 'oracle'}
    return request, snapshot
