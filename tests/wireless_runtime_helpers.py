"""独立标量调度oracle；系数不代表科学无线模型。"""
from types import SimpleNamespace

import torch

from muse_sdk.config import DataModel, export_config_schema
from muse_sdk.descriptor import RadioDescriptor, SensingDescriptor
from muse_sdk.geometry import GeometryPoint
from muse_sdk.models import ModelResult, RadioEvaluationResult, RadioModel, RateMetrics, SensingModel, SensingResult
from muse_sdk.network import (BoundModel, CommunicationBinding, ControlBinding, EndpointBudget,
    NetworkSpec, ResourceSelection, ServiceBinding, network_model_input_fields)


class EmptyConfig(DataModel):
    pass


def runtime_spec(*, resources=1, links=1, bits=7., sharing='reuse', service=True):
    return NetworkSpec(
        endpoints=tuple(GeometryPoint(group_id='agents', slot=i, device_id='rf', body='root',
            position_m=(0., 0., 0.), assumption='test') for i in range(links+1)),
        endpoint_sets={'tx': tuple(range(links)), 'rx': (links,)},
        endpoint_to_point=tuple(range(links+1)),
        resources=tuple(ResourceSelection(resource_id=f'r{i}', channel_id=[7, -3, 19][i],
            center_frequency_hz=1e9+i*1e8, bandwidth_hz=1e6, sharing=sharing) for i in range(resources)),
        endpoint_limits=tuple(EndpointBudget(resource_indices=tuple(range(resources)), max_power_w=2.) for _ in range(links+1)),
        role_controls=(ControlBinding(role_id='pilot', endpoint_indices=tuple(range(links))),),
        background_emitters=(), passive_endpoints=(links,),
        communication_allocations=tuple(CommunicationBinding(link_index=i, tx=i, rx=links, path=i,
            resource_indices=tuple(range(resources))) for i in range(links)),
        services=(ServiceBinding(service_id='data', link_indices=tuple(range(links)),
            resource_indices=tuple(range(resources)), initial_bits=bits),) if service else (),
        epoch_offsets=(0, 4, 8, 10), physics_dt=.001, link_count=links,
        sensing_count=links, point_count=links+1, path_count=links)


class ConstantRate(RadioModel):
    @classmethod
    def describe(cls):
        return {}

    def reset_state(self, state, reset_mask, inputs, rng):
        self.calls = []
        return self.result(inputs, reset=True)

    def result(self, inputs, reset=False):
        active = inputs.allocations['link_active']
        rate = torch.full(active.shape, 0. if reset else 1000., device=self.context.device)
        valid = torch.zeros_like(active) if reset else active.clone()
        return ModelResult(RadioEvaluationResult(None, None, RateMetrics(rate, valid)), (), {})

    def evaluate(self, inputs, state, rng):
        self.calls.append((inputs.sim_time.clone(), inputs.interval_s.clone()))
        return self.result(inputs)


class MixedSensing(SensingModel):
    @classmethod
    def describe(cls):
        return {}

    def result(self, inputs, reset=False):
        b = self.layout['axes']['B']
        tick = (inputs.sim_time/.001).round().long()
        value = torch.where(tick == 0, 2., 3.).to(torch.float64)
        invalid = (tick == 4) | reset
        fim = value[:, None, None] * torch.eye(2, dtype=torch.float64, device=self.context.device)
        fim[invalid] = float('nan')
        measurement = tick[:, None].float()
        return ModelResult(SensingResult(measurement, fim,
            {'measurement': torch.ones_like(measurement, dtype=torch.bool), 'fim': ~invalid},
            {'measurement': 'snapshot', 'fim': 'increment'}), (), {})

    def reset_state(self, state, reset_mask, inputs, rng):
        return self.result(inputs, True)

    def evaluate(self, inputs, state, rng):
        return self.result(inputs)


def make_runtime(*, spec=None, batch=2, radio=ConstantRate, sensing=None):
    from muse_runtime.network.runtime import WirelessRuntime
    from muse_runtime.rng import WorldRng
    spec = spec or runtime_spec()
    fields = [item for phase in ('reset', 'post_physics') for item in
              network_model_input_fields(spec, geometry=None, batch_size=batch, phase=phase)]
    model_fields = [item.model_copy(update={'tensor': item.tensor.model_copy(update={'shape': [
        'B', *item.tensor.shape[1:]]})}) for item in fields if item.path[:2] == ['network', 'allocations']]
    common = dict(config_schema=export_config_schema(EmptyConfig), state_scope='world',
                  state_schema={}, diagnostics_schema={}, assumptions={'synthetic': True})
    bindings, types = {}, {}
    if radio is not None:
        descriptor = RadioDescriptor(model_id='test_rate', input_fields=model_fields,
            output_spec=dict(output_blocks=['rate']), **common)
        bindings['radio'] = BoundModel(reference='test_rate', config={}, descriptor=descriptor, input_fields=fields)
        types['radio'] = radio
    if sensing is not None:
        descriptor = SensingDescriptor(model_id='test_sensing', input_fields=[], output_spec=dict(
            measurement=dict(shape=['B', 1], dtype='float32', units='tick', frame='none', rate='radio_epoch', time_semantics='snapshot'),
            fim=dict(shape=['B', 2, 2], dtype='float64', units='1', frame='none', rate='radio_epoch', time_semantics='increment')), **common)
        bindings['sensing'] = BoundModel(reference='test_sensing', config={}, descriptor=descriptor, input_fields=fields)
        types['sensing'] = sensing
    result = WirelessRuntime(spec, bindings, types, batch_size=batch, device='cpu', rng=WorldRng(7, batch, 'cpu'),
                             geometry_layout=None, geometry_view=None)
    result.reset(torch.ones(batch, dtype=torch.bool), sample_tick=torch.zeros(batch, dtype=torch.int64))
    return result


def requests(links=1, batch=2, power=1., channel=7):
    from muse_sdk.runtime_types import RadioAction
    return {'pilot': RadioAction(torch.full((batch, links), power),
                                 torch.full((batch, links), channel, dtype=torch.int32))}


def epoch(runtime, left, right, action=None, batch=2):
    runtime.begin_epoch(torch.full((batch,), left, dtype=torch.int64), interval_ticks=right-left,
                        radio_requests=action or requests(batch=batch))
    runtime.end_epoch(torch.full((batch,), right, dtype=torch.int64))
