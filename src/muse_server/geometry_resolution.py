"""RF语义绑定与固定有向路径展开；只消费CPU资源事实。"""
from itertools import product

from muse_sdk.config import ConfigError
from muse_sdk.geometry import (ConsumerInterference, DesiredGeometry, GeometryPoint,
    InterfererGeometry, PropagationPath, PropagationPathLayout, ResolvedGeometry, SensingGeometry)


def _selected_model(selected, path, groups, entity_models):
    group = groups.get(selected.group_id)
    if group is None:
        raise ConfigError(path + '.group_id', 'Unknown group')
    if len(set(selected.slots)) != len(selected.slots) or any(slot < 0 or slot >= group.count for slot in selected.slots):
        raise ConfigError(path + '.slots', 'Expected distinct in-range slots')
    return entity_models[group.model_id]


def resolve_endpoints(selection, *, groups, entity_models):
    """一次展开唯一设备点；geometry与network复用同一E→Q顺序。"""
    points, endpoints, identities = [], {}, set()
    for name, selected in selection.endpoints.items():
        path = 'scenario.radio_geometry.endpoints.' + name
        model = _selected_model(selected, path, groups, entity_models)
        device = model.radio_devices.get(selected.device_id)
        if device is None:
            raise ConfigError(path + '.device_id', 'Unknown radio device')
        endpoints[name] = []
        for slot in selected.slots:
            identity = (selected.group_id, slot, selected.device_id)
            if identity in identities:
                raise ConfigError(path, 'Duplicate physical radio device')
            identities.add(identity)
            index = len(points)
            endpoints[name].append(index)
            points.append(GeometryPoint(group_id=selected.group_id, slot=slot,
                device_id=selected.device_id, body=device.body, position_m=tuple(device.position_m),
                assumption=device.assumption))
        endpoints[name] = tuple(endpoints[name])
    return tuple(points), endpoints


def resolve_geometry(selection, *, environment_rf, summary, groups, entity_models, endpoints):
    if selection is None or not (selection.links or selection.sensing):
        return None
    field = 'scenario.radio_geometry'
    if environment_rf is None:
        raise ConfigError('environment.rf_geometry', 'RF geometry requires explicit environment semantics')
    if summary is None:
        raise ConfigError('environment.geometry_summary', 'Missing registered native geometry facts')
    blockers = environment_rf.rf_blocking
    if len(set(blockers)) != len(blockers):
        raise ConfigError('environment.rf_geometry.rf_blocking', 'Duplicate blocker')
    for name in blockers:
        fact = summary.geoms.get(name)
        if name in summary.flex_names or fact is None or not fact.is_static or fact.geom_type not in ('box', 'mesh'):
            raise ConfigError('environment.rf_geometry.rf_blocking', f'Unsupported or unknown RF blocker: {name}')
    device_points, endpoints = endpoints
    points = list(device_points)
    capabilities = {i: entity_models[groups[point.group_id].model_id].radio_devices[point.device_id]
                    for i, point in enumerate(device_points)}

    def endpoints_for(name, direction, path):
        if name not in endpoints:
            raise ConfigError(path, f'Unknown endpoint collection: {name}')
        indices = endpoints[name]
        if any(not getattr(capabilities[i], 'can_' + direction) for i in indices):
            raise ConfigError(path, 'Device direction capability mismatch')
        return indices

    paths, path_indices = [], {}

    def add_path(tx, rx):
        pair = (tx, rx)
        if pair not in path_indices:
            index = len(paths)
            path_indices[pair] = index
            paths.append(PropagationPath(path_id=f'p{index}', tx=tx, rx=rx))
        return path_indices[pair]

    desired, sensing, consumers = [], [], []
    seen_links, seen_sensing, targets = set(), set(), {}
    for link in selection.links:
        path = field + '.links.' + link.link_id
        if link.link_id in seen_links:
            raise ConfigError(path, 'Duplicate link ID')
        seen_links.add(link.link_id)
        for tx, rx in product(endpoints_for(link.tx, 'transmit', path), endpoints_for(link.rx, 'receive', path)):
            if tx == rx:
                raise ConfigError(path, 'Self desired link has no propagation segment')
            consumers.append(('desired', len(desired), tx, rx))
            desired.append(DesiredGeometry(link_id=link.link_id, path=add_path(tx, rx)))
    for relation in selection.sensing:
        path = field + '.sensing.' + relation.sensing_id
        if relation.sensing_id in seen_sensing:
            raise ConfigError(path, 'Duplicate sensing ID')
        seen_sensing.add(relation.sensing_id)
        model = _selected_model(relation.target, path + '.target', groups, entity_models)
        target_points = []
        for slot in relation.target.slots:
            identity = (relation.target.group_id, slot)
            if identity not in targets:
                targets[identity] = len(points)
                points.append(GeometryPoint(group_id=identity[0], slot=slot, device_id=None,
                    body=model.root_body, position_m=(0., 0., 0.), assumption='理想实体root目标参考点'))
            target_points.append(targets[identity])
        for tx, target, rx in product(endpoints_for(relation.tx, 'transmit', path), target_points,
                                     endpoints_for(relation.rx, 'receive', path)):
            consumers.append(('sensing', len(sensing), tx, rx))
            sensing.append(SensingGeometry(sensing_id=relation.sensing_id,
                outbound=add_path(tx, target), inbound=add_path(target, rx),
                direct=add_path(tx, rx) if tx != rx else None))
    interference = []
    for kind, index, own_tx, rx in consumers:
        sources = tuple(InterfererGeometry(tx=tx, path=add_path(tx, rx))
            for tx, device in capabilities.items() if device.can_transmit and tx not in (own_tx, rx))
        interference.append(ConsumerInterference(kind=kind, consumer=index, sources=sources))
    if not consumers:
        return None
    return ResolvedGeometry(endpoint_epsilon_m=selection.endpoint_epsilon_m, blockers=tuple(blockers),
        layout=PropagationPathLayout(points=tuple(points), paths=tuple(paths), desired=tuple(desired),
            sensing=tuple(sensing), interference=tuple(interference)))
