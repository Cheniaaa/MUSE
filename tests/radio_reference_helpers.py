"""独立math标量oracle及CPU/CUDA共用的非对称冻结布局。"""
import math
from types import MappingProxyType

from muse_sdk.geometry import (ConsumerInterference, DesiredGeometry, GeometryPoint,
    InterfererGeometry, PropagationPath, PropagationPathLayout)
from muse_sdk.models import RadioInputs
from muse_sdk.network import (CommunicationBinding, EndpointBudget, NetworkSpec,
    ResourceSelection)


def scalar_radio_reference(*, law, config, geometry_layout, network_spec, inputs):
    """输入各view为嵌套list；返回{block: {field: [B][L][F]}}，evaluate语义。"""
    views = inputs._asdict() if isinstance(inputs, RadioInputs) else inputs
    geometry, masks = views['geometry'], views['endpoint_masks']
    resources, allocations = views['resources'], views['allocations']
    transmissions = views['simultaneous_transmissions']
    point_to_endpoint = {point: e for e, point in enumerate(network_spec.endpoint_to_point)}
    candidates = {(item.link_index, f): item for item in network_spec.communication_allocations
                  for f in item.resource_indices}
    sources = {item.consumer: item.sources for item in geometry_layout.interference
               if item.kind == 'desired'}
    b, l, f = len(geometry['distance_m']), network_spec.link_count, len(network_spec.resources)
    fields = {'link': ('path_gain_linear', 'received_power_w', 'valid'),
              'interference': ('interference_power_w', 'sinr_linear', 'valid'),
              'rate': ('rate_bps', 'valid')}
    output = {block: {field: [[[False if field == 'valid' else 0. for _ in range(f)]
        for _ in range(l)] for _ in range(b)] for field in names} for block, names in fields.items()}

    def gain(world, path, frequency):
        if not geometry['valid'][world][path]:
            return None
        distance = geometry['distance_m'][world][path]
        if law == 'free_space_radio':
            if distance < config['minimum_distance_m']:
                return None
            return (299792458. / (4. * math.pi * distance * frequency)) ** 2
        if distance < 1. or distance > 150.:
            return None
        ghz = frequency / 1e9
        loss = 32.4 + 17.3 * math.log10(distance) + 20. * math.log10(ghz)
        if not geometry['los'][world][path]:
            loss = max(loss, 38.3 * math.log10(distance) + 17.30 + 24.9 * math.log10(ghz))
        return 10. ** (-loss / 10.)

    for world in range(b):
        for link in range(l):
            for resource in range(f):
                binding = candidates.get((link, resource))
                if binding is None or not (masks['present'][world][binding.tx]
                                           and masks['present'][world][binding.rx]):
                    continue
                frequency = resources['frequency_hz'][resource]
                desired_gain = gain(world, geometry_layout.desired[link].path, frequency)
                if desired_gain is None:
                    continue
                signal = (allocations['link_power_w'][world][link][resource] * desired_gain
                          if allocations['link_active'][world][link][resource] else 0.)
                output['link']['path_gain_linear'][world][link][resource] = desired_gain
                output['link']['received_power_w'][world][link][resource] = signal
                output['link']['valid'][world][link][resource] = True
                receiver_transmitting = (transmissions['active'][world][binding.rx][resource]
                    and transmissions['power_w'][world][binding.rx][resource] > 0.)
                interference, known = 0., not receiver_transmitting
                for source in sources.get(link, ()):
                    endpoint = point_to_endpoint[source.tx]
                    power = transmissions['power_w'][world][endpoint][resource]
                    if (not masks['present'][world][endpoint]
                        or not transmissions['active'][world][endpoint][resource] or power == 0.):
                        continue
                    source_gain = gain(world, source.path, frequency)
                    if source_gain is None:
                        known = False
                    else:
                        interference += power * source_gain
                if not known:
                    continue
                noise = (1.380649e-23 * config['receiver_noise_temperature_k']
                    * resources['bandwidth_hz'][resource]
                    * 10. ** (config['receiver_noise_figure_db'] / 10.))
                sinr = signal / (interference + noise)
                output['interference']['interference_power_w'][world][link][resource] = interference
                output['interference']['sinr_linear'][world][link][resource] = sinr
                output['interference']['valid'][world][link][resource] = True
                output['rate']['rate_bps'][world][link][resource] = (
                    resources['bandwidth_hz'][resource] * math.log1p(sinr) / math.log(2.))
                output['rate']['valid'][world][link][resource] = True
    return output


def radio_case(*, device='cpu', permutation=False):
    """返回(layout, RadioInputs, bound_slots)；B=2/E=4/L=3/P=6/F=3。

    permutation=True按old-index顺序E=[2,0,3,1]、P=[5,2,0,4,1,3]、
    F=[2,0,1]重排相同实体。L0与L2共享P0，候选F不同；point→E非恒等。
    bound_slots故意全True，真实候选只能从communication_allocations读取。
    """
    import torch

    points = tuple(GeometryPoint(group_id='nodes', slot=slot, device_id='ideal_point_radio',
        body='node', position_m=(0., 0., 0.), assumption='科学算例理想点')
        for slot in (8, 2, 31, 5, 19))
    paths = tuple(PropagationPath(path_id=f'p{i}', tx=tx, rx=rx)
                  for i, (tx, rx) in enumerate(((3, 0), (4, 0), (1, 0), (1, 3), (0, 3), (4, 3))))
    endpoint_points = [3, 0, 4, 1]
    resources = [ResourceSelection(resource_id=f'f{i}', channel_id=channel,
        center_frequency_hz=frequency, bandwidth_hz=bandwidth, sharing='reuse')
        for i, (channel, frequency, bandwidth) in enumerate(
            ((7, 2.4e9, 5e6), (-3, 5e9, 2e6), (19, .9e9, 1e6)))]
    bindings = [CommunicationBinding(link_index=0, tx=0, rx=1, path=0, resource_indices=(0, 2)),
                CommunicationBinding(link_index=1, tx=3, rx=0, path=3, resource_indices=(1, 2)),
                CommunicationBinding(link_index=2, tx=0, rx=1, path=0, resource_indices=(1,))]
    desired_paths = [0, 3, 0]
    interference = [ConsumerInterference(kind='desired', consumer=link,
        sources=tuple(InterfererGeometry(tx=tx, path=path) for tx, path in sources))
        for link, sources in ((0, ((4, 1), (1, 2))), (1, ((0, 4), (4, 5))),
                              (2, ((4, 1), (1, 2))))]
    tensor = lambda values, dtype: torch.tensor(values, dtype=dtype, device=device)
    distance = tensor([[10., 17., 7., 13., 11., 23.], [25., 8., 19., 31., 29., 6.]], torch.float32)
    los = tensor([[True, False, True, False, True, True],
                  [False, True, False, True, False, True]], torch.bool)
    power = tensor([[[.001, .0003, 0.], [0., 0., 0.], [.0004, .0002, 0.], [.0002, .0005, .0001]],
                    [[.0007, 0., .0004], [0., .0001, 0.], [.0003, 0., .0008], [0., .0002, .0006]]], torch.float32)
    link_power = tensor([[[.001, 0., 0.], [0., .0005, .0001], [0., .0003, 0.]],
                         [[.0007, 0., .0004], [0., .0002, .0006], [0., 0., 0.]]], torch.float32)
    present = torch.ones((2, 4), dtype=torch.bool, device=device)
    if permutation:
        e_order, p_order, f_order = [2, 0, 3, 1], [5, 2, 0, 4, 1, 3], [2, 0, 1]
        e_inverse = {old: new for new, old in enumerate(e_order)}
        p_inverse = {old: new for new, old in enumerate(p_order)}
        f_inverse = {old: new for new, old in enumerate(f_order)}
        endpoint_points = [endpoint_points[i] for i in e_order]
        paths = tuple(paths[i] for i in p_order)
        resources = [resources[i] for i in f_order]
        desired_paths = [p_inverse[i] for i in desired_paths]
        interference = [item.model_copy(update={'sources': tuple(source.model_copy(
            update={'path': p_inverse[source.path]}) for source in item.sources)}) for item in interference]
        bindings = [item.model_copy(update={'tx': e_inverse[item.tx], 'rx': e_inverse[item.rx],
            'path': p_inverse[item.path], 'resource_indices': tuple(f_inverse[i] for i in item.resource_indices)})
            for item in bindings]
        distance, los = distance[:, p_order], los[:, p_order]
        power, present = power[:, e_order][:, :, f_order], present[:, e_order]
        link_power = link_power[:, :, f_order]
    geometry = PropagationPathLayout(points=points, paths=paths,
        desired=tuple(DesiredGeometry(link_id=f'l{i}', path=path) for i, path in enumerate(desired_paths)),
        sensing=(), interference=tuple(interference))
    network = NetworkSpec(endpoints=tuple(points[i] for i in endpoint_points), endpoint_sets={},
        endpoint_to_point=tuple(endpoint_points), resources=tuple(resources),
        endpoint_limits=tuple(EndpointBudget(resource_indices=(0, 1, 2), max_power_w=.01) for _ in range(4)),
        role_controls=(), background_emitters=(), passive_endpoints=(), communication_allocations=tuple(bindings),
        services=(), epoch_offsets=(0, 1), physics_dt=.002, link_count=3, sensing_count=0, point_count=5, path_count=6)
    layout = MappingProxyType({'geometry': geometry, 'network': network, 'instance_id': 'reference',
        'axes': MappingProxyType({'B': 2, 'E': 4, 'L': 3, 'P': 6, 'Q': 5, 'F': 3})})
    inputs = RadioInputs({'distance_m': distance, 'valid': torch.ones_like(los), 'los': los},
        {'present': present},
        {'frequency_hz': tensor([r.center_frequency_hz for r in resources], torch.float64),
         'bandwidth_hz': tensor([r.bandwidth_hz for r in resources], torch.float64)},
        {'link_active': link_power > 0, 'link_power_w': link_power},
        {'active': power > 0, 'power_w': power},
        torch.zeros(2, dtype=torch.float64, device=device), torch.full((2,), .014, dtype=torch.float64, device=device))
    return layout, inputs, torch.ones((2, 3, 3), dtype=torch.bool, device=device)
