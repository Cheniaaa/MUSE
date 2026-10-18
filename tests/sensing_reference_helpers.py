"""独立math科学oracle与真实ModelInstance CPU/CUDA共用夹具。"""
import math
from types import MappingProxyType

from muse_sdk.geometry import (ConsumerInterference, GeometryPoint, InterfererGeometry,
    PropagationPath, PropagationPathLayout, ResolvedGeometry, SensingGeometry)
from muse_sdk.models import ExecutionContext, SensingInputs
from muse_sdk.network import EndpointBudget, NetworkSpec, ResourceSelection, network_model_input_fields


def scalar_sensing_reference(*, config, geometry_layout, network_spec, inputs):
    """仅用math和SDK静态关系；数值输入按已量化tensor转list，不调用产品公式。"""
    views = inputs._asdict() if isinstance(inputs, SensingInputs) else inputs
    views = {name: {key: value.tolist() if hasattr(value, 'tolist') else value
                   for key, value in view.items()} if isinstance(view, dict) or hasattr(view, 'items')
             else view.tolist() if hasattr(view, 'tolist') else view for name, view in views.items()}
    geometry, transmissions = views['geometry'], views['simultaneous_transmissions']
    present, resources = views['endpoint_masks']['present'], views['resources']
    point_to_endpoint = {point: e for e, point in enumerate(network_spec.endpoint_to_point)}
    sources = {item.consumer: item.sources for item in geometry_layout.interference if item.kind == 'sensing'}
    b, s, f = len(present), len(geometry_layout.sensing), len(network_spec.resources)
    result = {name: [[[False if name.endswith('valid') else 0. for _ in range(f)]
                     for _ in range(s)] for _ in range(b)]
              for name in ('h', 'a', 'variance', 'echo', 'noise_psd', 'measurement_valid', 'fim_valid')}
    result['gradient'] = [[[[0., 0.] for _ in range(f)] for _ in range(s)] for _ in range(b)]
    result['fim'] = [[[[[0., 0.], [0., 0.]] for _ in range(f)] for _ in range(s)] for _ in range(b)]
    for world in range(b):
        for index, relation in enumerate(geometry_layout.sensing):
            outbound, inbound = (geometry_layout.paths[path] for path in (relation.outbound, relation.inbound))
            tx, rx = point_to_endpoint[outbound.tx], point_to_endpoint[inbound.rx]
            for resource in range(f):
                power = transmissions['power_w'][world][tx][resource]
                if not (present[world][tx] and present[world][rx]
                        and transmissions['active'][world][tx][resource] and power > 0):
                    result['fim_valid'][world][index][resource] = True
                    continue
                paths = (relation.outbound, relation.inbound)
                if any(geometry['valid'][world][p] and not geometry['los'][world][p] for p in paths):
                    result['fim_valid'][world][index][resource] = True
                    continue
                if any(not geometry['valid'][world][p]
                       or geometry['distance_m'][world][p] < config['minimum_distance_m'] for p in paths):
                    continue
                dt, dr = (geometry['distance_m'][world][p] for p in paths)
                frequency, bandwidth = resources['frequency_hz'][resource], resources['bandwidth_hz'][resource]
                interference, known = 0., True
                for source in sources.get(index, ()):
                    endpoint = point_to_endpoint[source.tx]
                    source_power = transmissions['power_w'][world][endpoint][resource]
                    if not (present[world][endpoint] and transmissions['active'][world][endpoint][resource]
                            and source_power > 0):
                        continue
                    if not geometry['valid'][world][source.path]:
                        known = False
                    elif geometry['los'][world][source.path]:
                        distance = geometry['distance_m'][world][source.path]
                        if distance < config['minimum_distance_m']:
                            known = False
                        else:
                            interference += source_power * (299792458. / (4 * math.pi * distance * frequency))**2
                if not known:
                    continue
                echo = power * config['target_rcs_m2'] * (299792458. / frequency)**2 / ((4 * math.pi)**3 * dt**2 * dr**2)
                noise = 1.380649e-23 * config['receiver_noise_temperature_k'] * 10**(config['receiver_noise_figure_db'] / 10) + interference / bandwidth
                a = 299792458.**2 * noise / (8 * math.pi**2 * (bandwidth**2 / 12) * echo)
                variance = a / views['interval_s'][world]
                q, t, r = (geometry['positions_world'][world][p] for p in (outbound.rx, outbound.tx, inbound.rx))
                gradient = [(q[k] - t[k]) / dt + (q[k] - r[k]) / dr for k in (0, 1)]
                for name, value in (('h', dt + dr), ('echo', echo), ('noise_psd', noise), ('a', a), ('variance', variance), ('gradient', gradient)):
                    result[name][world][index][resource] = value
                result['fim'][world][index][resource] = [[gradient[j] * gradient[k] / variance for k in (0, 1)] for j in (0, 1)]
                result['measurement_valid'][world][index][resource] = True
                result['fim_valid'][world][index][resource] = True
    return result


def sensing_case(*, device='cpu', permutation=False):
    """B2/S4/E4/F3，两个穿插Q的目标，单站/异站及每endpoint单resource发射。"""
    import torch

    endpoint_points = [3, 0, 5, 2]
    points = tuple(GeometryPoint(group_id='nodes' if i in endpoint_points else 'targets', slot=i,
        device_id='ideal_point_radio' if i in endpoint_points else None, body='node',
        position_m=(0., 0., 0.), assumption='独立科学算例点') for i in range(6))
    paths, lookup = [], {}

    def path(tx, rx):
        if (tx, rx) not in lookup:
            lookup[tx, rx] = len(paths)
            paths.append(PropagationPath(path_id=f'p{len(paths)}', tx=tx, rx=rx))
        return lookup[tx, rx]

    relations, interference = [], []
    for s, (tx, target, rx) in enumerate(((3, 1, 3), (0, 1, 0), (3, 4, 5), (5, 4, 0))):
        relations.append(SensingGeometry(sensing_id=f's{s}', outbound=path(tx, target),
            inbound=path(target, rx), direct=None if tx == rx else path(tx, rx)))
        interference.append(ConsumerInterference(kind='sensing', consumer=s, sources=tuple(
            InterfererGeometry(tx=j, path=path(j, rx)) for j in endpoint_points if j not in (tx, rx))))
    resources = [ResourceSelection(resource_id=f'f{i}', channel_id=channel, center_frequency_hz=f,
        bandwidth_hz=bw, sharing='reuse') for i, (channel, f, bw) in enumerate(
            ((7, 2.412e9, 5e6), (-3, 2.437e9, 5e6), (19, .9e9, 2e6)))]
    positions = torch.tensor([[(8, 0, 1), (4, 4, 0), (4, -8, 1), (0, 0, 1), (3, 5, 1), (4, 8, 1)],
                             [(9, 1, 2), (3, 5, 0), (5, -7, 1), (-1, 0, 1), (6, 4, 1), (4, 9, 2)]], dtype=torch.float32, device=device)
    power = torch.zeros((2, 4, 3), dtype=torch.float32, device=device)
    for world, values in enumerate((((0, .001), (1, .001), (2, .0006), (0, .001)),
                                  ((2, .0007), (0, .0013), (1, .0008), (0, .0004)))):
        for endpoint, (f, p) in enumerate(values):
            power[world, endpoint, f] = p
    distance = torch.stack([torch.linalg.vector_norm(positions[:, p.tx].double() - positions[:, p.rx].double(), dim=-1).float() for p in paths], dim=1)
    if permutation:
        e_order, p_order, f_order, s_order = [2, 0, 3, 1], list(reversed(range(len(paths)))), [2, 0, 1], [3, 1, 0, 2]
        p_inverse, s_inverse = ({old: new for new, old in enumerate(order)} for order in (p_order, s_order))
        endpoint_points = [endpoint_points[i] for i in e_order]
        paths = [paths[i] for i in p_order]
        relations = [relations[i].model_copy(update={name: p_inverse[value] if value is not None else None
            for name, value in (('outbound', relations[i].outbound), ('inbound', relations[i].inbound), ('direct', relations[i].direct))}) for i in s_order]
        interference = [item.model_copy(update={'consumer': s_inverse[item.consumer], 'sources': tuple(
            source.model_copy(update={'path': p_inverse[source.path]}) for source in item.sources)}) for item in interference]
        resources = [resources[i] for i in f_order]
        power, distance = power[:, e_order][:, :, f_order], distance[:, p_order]
    geometry = PropagationPathLayout(points=points, paths=tuple(paths), desired=(), sensing=tuple(relations), interference=tuple(interference))
    network = NetworkSpec(endpoints=tuple(points[i] for i in endpoint_points), endpoint_sets={}, endpoint_to_point=tuple(endpoint_points),
        resources=tuple(resources), endpoint_limits=tuple(EndpointBudget(resource_indices=(0, 1, 2), max_power_w=.01) for _ in range(4)),
        role_controls=(), background_emitters=(), passive_endpoints=(), communication_allocations=(), services=(),
        epoch_offsets=(0, 7, 14, 20), physics_dt=.002, link_count=0, sensing_count=4, point_count=6, path_count=len(paths))
    layout = MappingProxyType({'geometry': geometry, 'network': network, 'instance_id': 'sensing_reference',
        'axes': MappingProxyType({'B': 2, 'E': 4, 'S': 4, 'Q': 6, 'P': len(paths), 'F': 3})})
    inputs = SensingInputs({'positions_world': positions, 'distance_m': distance, 'valid': torch.ones_like(distance, dtype=torch.bool),
        'los': torch.ones_like(distance, dtype=torch.bool)}, {'present': torch.ones((2, 4), dtype=torch.bool, device=device)},
        {'frequency_hz': torch.tensor([r.center_frequency_hz for r in resources], dtype=torch.float64, device=device),
         'bandwidth_hz': torch.tensor([r.bandwidth_hz for r in resources], dtype=torch.float64, device=device)}, {},
        {'active': power > 0, 'power_w': power}, None, torch.zeros(2, dtype=torch.float64, device=device),
        torch.full((2,), .014, dtype=torch.float64, device=device))
    return layout, inputs


def sensing_instance(model_id, layout, *, config, device='cpu', rng=None):
    from muse_builtins.models import sensing_registrations
    from muse_sdk.descriptor import build_descriptor
    from muse_sdk.registration import TrainingPackageDefinition
    from muse_runtime.model_execution import ModelInstance
    from muse_runtime.rng import WorldRng

    implementation = sensing_registrations()[model_id]
    descriptor = build_descriptor(TrainingPackageDefinition({}, {}, {}, {}, {model_id: implementation}), {}, 'a' * 64, default_composition_id=None).sensing_models[model_id]
    geometry = ResolvedGeometry(layout=layout['geometry'], endpoint_epsilon_m=1e-4, blockers=())
    fields = tuple(field for phase in ('reset', 'post_physics') for field in network_model_input_fields(
        layout['network'], geometry=geometry, batch_size=layout['axes']['B'], phase=phase))
    return ModelInstance(implementation, descriptor=descriptor, config=config, layout=layout,
        context=ExecutionContext(device), available_inputs=fields, rng=rng or WorldRng(73, layout['axes']['B'], device))
