"""手算P轴oracle：通信、感知直达与候选干扰不能混淆。"""
import pytest

from muse_sdk.experiment import EntityGroup, EntityModelRef
from muse_sdk.geometry import EnvironmentRF, GeometryAssetSummary, GeometrySelection, RadioDeviceDefinition


def inputs():
    device = RadioDeviceDefinition(can_transmit=True, can_receive=True, body='root',
        position_m=[0., 0., 0.], orientation_wxyz=[1., 0., 0., 0.], assumption='ideal')
    models = {'vehicle': EntityModelRef(resource_id='vehicle', revision=1, sha256='a'*64,
        physics_asset='asset.xml', root_body='root', support_geoms=[], radio_devices={'radio': device})}
    groups = {'g': EntityGroup(group_id='g', model_id='vehicle', model_revision=1, count=5,
        placement_id='initial', controller_id='none', sensor_profile_id='none')}
    return dict(environment_rf=EnvironmentRF(rf_blocking=[]),
        summary=GeometryAssetSummary(geoms={}, flex_names=[]), groups=groups, entity_models=models)


def endpoint(slots, device='radio'):
    return dict(group_id='g', slots=slots, device_id=device)


def resolve(raw, **changes):
    from muse_server.geometry_resolution import resolve_endpoints, resolve_geometry
    selection = GeometrySelection.model_validate(raw)
    args = inputs() | changes
    endpoints = resolve_endpoints(selection, groups=args['groups'], entity_models=args['entity_models'])
    return resolve_geometry(selection, endpoints=endpoints, **args)


def test_desired_paths_are_shared_with_other_consumers_interference():
    resolved = resolve(dict(endpoints={'tx': endpoint([0, 1]), 'rx': endpoint([2]), 'jammer': endpoint([3])},
        links=[dict(link_id='uplink', tx='tx', rx='rx')]))
    layout = resolved.layout
    assert [(p.tx, p.rx) for p in layout.paths] == [(0, 2), (1, 2), (3, 2)]
    assert [p.path_id for p in layout.paths] == ['p0', 'p1', 'p2']
    assert [x.path for x in layout.desired] == [0, 1]
    assert [[(x.tx, x.path) for x in row.sources] for row in layout.interference] == [[(1, 1), (3, 2)], [(0, 0), (3, 2)]]
    assert len(layout.points) == 4


def test_sensing_only_receiver_has_echo_direct_and_external_interference():
    args = inputs()
    args['entity_models']['vehicle'].radio_devices['receiver'] = args['entity_models']['vehicle'].radio_devices['radio'].model_copy(update={'can_transmit': False})
    resolved = resolve(dict(endpoints={'tx': endpoint([0]), 'rx': endpoint([1], 'receiver'), 'jammer': endpoint([2])},
        sensing=[dict(sensing_id='radar', tx='tx', rx='rx', target=dict(group_id='g', slots=[4]))]), **args)
    layout = resolved.layout
    assert [(p.tx, p.rx) for p in layout.paths] == [(0, 3), (3, 1), (0, 1), (2, 1)]
    assert (layout.sensing[0].outbound, layout.sensing[0].inbound, layout.sensing[0].direct) == (0, 1, 2)
    assert [(s.tx, s.path) for s in layout.interference[0].sources] == [(2, 3)]
    assert layout.interference[0].kind == 'sensing'


def test_monostatic_has_no_zero_length_self_direct_and_slot_order_is_preserved():
    resolved = resolve(dict(endpoints={'tx': endpoint([2, 0])}, sensing=[dict(
        sensing_id='radar', tx='tx', rx='tx', target=dict(group_id='g', slots=[1]))]))
    assert [p.slot for p in resolved.layout.points] == [2, 0, 1]
    assert resolved.layout.sensing[0].direct is None
    assert resolved.layout.sensing[-1].direct is None
    assert all(p.tx != p.rx for p in resolved.layout.paths)


@pytest.mark.parametrize('mutation', [
    lambda r: r['endpoints'].update(other=endpoint([0])),
    lambda r: r['endpoints']['tx'].update(slots=[5]),
    lambda r: r['endpoints']['tx'].update(slots=[0, 0]),
    lambda r: r['endpoints']['tx'].update(device_id='unknown'),
    lambda r: r['endpoints']['tx'].update(group_id='unknown'),
    lambda r: r['links'][0].update(rx='unknown'),
    lambda r: r['links'][0].update(rx='tx'),
    lambda r: r['links'].append(dict(link_id='up', tx='tx', rx='rx')),
])
def test_invalid_bindings_are_field_errors(mutation):
    raw = dict(endpoints={'tx': endpoint([0]), 'rx': endpoint([1])}, links=[dict(link_id='up', tx='tx', rx='rx')])
    mutation(raw)
    with pytest.raises(ValueError, match='radio_geometry'):
        resolve(raw)


@pytest.mark.parametrize('can_transmit,can_receive', [(False, True), (True, False)])
def test_direction_capabilities_are_checked(can_transmit, can_receive):
    args = inputs()
    args['entity_models']['vehicle'].radio_devices['radio'].can_transmit = can_transmit
    args['entity_models']['vehicle'].radio_devices['radio'].can_receive = can_receive
    with pytest.raises(ValueError, match='radio_geometry'):
        resolve(dict(endpoints={'tx': endpoint([0]), 'rx': endpoint([1])}, links=[dict(link_id='up', tx='tx', rx='rx')]), **args)


def test_input_field_dimensions_are_derived_and_layout_roundtrips():
    from muse_sdk.geometry import ResolvedGeometry, geometry_input_fields
    resolved = resolve(dict(endpoints={'tx': endpoint([0]), 'rx': endpoint([1])}, links=[dict(link_id='up', tx='tx', rx='rx')]))
    assert ResolvedGeometry.model_validate_json(resolved.model_dump_json()) == resolved
    fields = {f.path[-1]: f.tensor for f in geometry_input_fields(resolved.layout, batch_size=7, phase='reset')}
    assert fields['positions_world'].shape == [7, 2, 3]
    assert fields['distance_m'].shape == [7, 1]
    assert fields['sample_time'].dtype == 'float64'
    assert fields['valid'].dtype == 'bool'


def test_invalid_target_reference_is_rejected():
    with pytest.raises(ValueError, match='radio_geometry'):
        resolve(dict(endpoints={'tx': endpoint([0])}, sensing=[dict(sensing_id='radar', tx='tx', rx='tx', target=dict(group_id='missing', slots=[0]))]))
