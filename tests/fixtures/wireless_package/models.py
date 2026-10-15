"""人工系数只验证真实输入依赖和区间调度，不代表传播/CRB。"""
from muse_sdk.config import DataModel
from muse_sdk.models import RadioModel, SensingModel, ModelResult, RadioEvaluationResult, RateMetrics, SensingResult


class RadioConfig(DataModel):
    rate_scale: float


class SensingConfig(DataModel):
    increment_scale: float


def field(path, shape, dtype, units, phase, frame='none'):
    return dict(path=path, phase=phase, tensor=dict(shape=shape, dtype=dtype, units=units,
                                                  frame=frame, rate='radio_epoch'))


def geometry_fields():
    return [field(['network', 'geometry', name], shape, dtype, units, phase)
            for phase in ('reset', 'post_physics') for name, shape, dtype, units in (
                ('distance_m', ['B', 'P'], 'float32', 'm'),
                ('valid', ['B', 'P'], 'bool', '1'),
                ('sample_tick', ['B'], 'int64', 'tick'))]


class SyntheticRadio(RadioModel):
    @classmethod
    def describe(cls):
        fields = geometry_fields() + [field(['network', view, name], shape, dtype, units, phase)
            for phase in ('reset', 'post_physics') for view, name, shape, dtype, units in (
                ('allocations', 'link_active', ['B', 'L', 'F'], 'bool', '1'),
                ('allocations', 'link_power_w', ['B', 'L', 'F'], 'float32', 'W'),
                ('simultaneous_transmissions', 'power_w', ['B', 'E', 'F'], 'float32', 'W'))]
        return dict(model_id='synthetic_radio', config_schema=RadioConfig, input_fields=fields,
            state_scope='world', state_schema={}, diagnostics_schema={},
            assumptions={'synthetic_schedule_oracle': True, 'scientific_propagation': False},
            output_spec={'output_blocks': ['rate']})

    def __init__(self, *args):
        super().__init__(*args)
        self.evaluations = []
        self.resets = []

    def result(self, inputs, reset=False):
        import torch
        paths = [link.path for link in self.layout['geometry'].desired]
        distance = inputs.geometry['distance_m'][:, paths, None]
        valid = inputs.geometry['valid'][:, paths, None] & inputs.allocations['link_active']
        power = inputs.allocations['link_power_w']
        others = inputs.simultaneous_transmissions['power_w'].sum(1)[:, None, :] - power
        rate = self.config['rate_scale']*power/(1+distance)/(1+others)
        if reset:
            rate, valid = torch.zeros_like(rate), torch.zeros_like(valid)
        return ModelResult(RadioEvaluationResult(None, None, RateMetrics(rate, valid)), (), {})

    def reset_state(self, state, reset_mask, inputs, rng):
        self.resets.append(reset_mask.world_reset.clone())
        return self.result(inputs, True)

    def evaluate(self, inputs, state, rng):
        self.evaluations.append((inputs.geometry['sample_tick'].clone(), inputs.interval_s.clone(),
                                 inputs.geometry['distance_m'].clone()))
        return self.result(inputs)


class SyntheticSensing(SensingModel):
    @classmethod
    def describe(cls):
        return dict(model_id='synthetic_sensing', config_schema=SensingConfig, input_fields=geometry_fields(),
            state_scope='world', state_schema={}, diagnostics_schema={}, assumptions={'synthetic_schedule_oracle': True},
            output_spec=dict(
                measurement=dict(shape=['B', 'S', 2], dtype='float32', units='m', frame='none', rate='radio_epoch', time_semantics='snapshot'),
                fim=dict(shape=['B', 'S', 2, 2], dtype='float64', units='1', frame='none', rate='radio_epoch', time_semantics='increment')))

    def result(self, inputs, reset=False):
        import torch
        paths = [[link.outbound, link.inbound] for link in self.layout['geometry'].sensing]
        distances = inputs.geometry['distance_m'][:, paths]
        valid = inputs.geometry['valid'][:, paths]
        fim = (torch.eye(2, dtype=torch.float64, device=self.context.device)[None, None]
               * inputs.interval_s[:, None, None, None]*self.config['increment_scale']).expand(
                   -1, len(paths), -1, -1).clone()
        fim_valid = valid.all(-1) & (not reset)
        return ModelResult(SensingResult(distances, fim, {'measurement': valid, 'fim': fim_valid},
            {'measurement': 'snapshot', 'fim': 'increment'}), (), {})

    def reset_state(self, state, reset_mask, inputs, rng):
        return self.result(inputs, True)

    def evaluate(self, inputs, state, rng):
        return self.result(inputs)


class TransmissionSensing(SensingModel):
    """只消费发射tensor，却必须用静态关系解释S轴。"""
    @classmethod
    def describe(cls):
        return dict(model_id='transmission_sensing', config_schema=SensingConfig,
            input_fields=[field(['network', 'simultaneous_transmissions', 'power_w'],
                ['B', 'E', 'F'], 'float32', 'W', phase) for phase in ('reset', 'post_physics')],
            state_scope='world', state_schema={}, diagnostics_schema={}, assumptions={'synthetic_schedule_oracle': True},
            output_spec=dict(measurement=dict(shape=['B', 'S', 2], dtype='float32', units='W', frame='none',
                rate='radio_epoch', time_semantics='snapshot')))

    def __init__(self, *args):
        super().__init__(*args)
        geometry = self.layout['geometry']
        self.tx = [geometry.paths[relation.outbound].tx for relation in geometry.sensing]

    def result(self, inputs):
        import torch
        assert not inputs.geometry
        power = inputs.simultaneous_transmissions['power_w'][:, self.tx].sum(-1)
        value = power[..., None].expand(-1, -1, 2).clone()
        return ModelResult(SensingResult(value, None, {'measurement': torch.ones_like(value, dtype=torch.bool)},
            {'measurement': 'snapshot'}), (), {})

    def reset_state(self, state, reset_mask, inputs, rng):
        return self.result(inputs)

    def evaluate(self, inputs, state, rng):
        return self.result(inputs)
