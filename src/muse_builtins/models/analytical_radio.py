"""确定性大尺度Radio参考；Torch仅在实例构造和计算时加载。"""
import math

from pydantic import Field

from muse_sdk.config import DataModel
from muse_sdk.models import (ExecutionContext, InterferenceMetrics, LinkMetrics, ModelResult,
    RadioEvaluationResult, RadioModel, RateMetrics)


class ReceiverNoiseConfig(DataModel):
    receiver_noise_temperature_k: float = Field(gt=0)
    receiver_noise_figure_db: float = Field(ge=0)


class FreeSpaceConfig(ReceiverNoiseConfig):
    minimum_distance_m: float = Field(gt=0)


def _description(model_id, config_type, *, indoor):
    fields = [
        ('geometry', 'distance_m', ['B', 'P'], 'float32', 'm'),
        ('geometry', 'valid', ['B', 'P'], 'bool', '1'),
        ('endpoint_masks', 'present', ['B', 'E'], 'bool', '1'),
        ('resources', 'frequency_hz', ['F'], 'float64', 'Hz'),
        ('resources', 'bandwidth_hz', ['F'], 'float64', 'Hz'),
        ('allocations', 'link_active', ['B', 'L', 'F'], 'bool', '1'),
        ('allocations', 'link_power_w', ['B', 'L', 'F'], 'float32', 'W'),
        ('simultaneous_transmissions', 'active', ['B', 'E', 'F'], 'bool', '1'),
        ('simultaneous_transmissions', 'power_w', ['B', 'E', 'F'], 'float32', 'W'),
    ]
    if indoor:
        fields.append(('geometry', 'los', ['B', 'P'], 'bool', '1'))
    return dict(model_id=model_id, config_schema=config_type,
        input_fields=[dict(path=['network', view, name], phase=phase,
            tensor=dict(shape=shape, dtype=dtype, units=units, frame='none', rate='radio_epoch'))
            for phase in ('reset', 'post_physics') for view, name, shape, dtype, units in fields],
        state_scope='world', state_schema={}, diagnostics_schema={},
        output_spec={'output_blocks': ['link', 'interference', 'rate']},
        frequency_hz_schema=({'type': 'number', 'exclusiveMinimum': 5e8, 'exclusiveMaximum': 1e11}
                             if indoor else None),
        assumptions={
            'source': ('3GPP TR38.901 V16.1.0 Table 7.4.1-1, InH-Office' if indoor
                       else 'ITU-R P.525-5 (11/2024) Annex 2.3 equations (4)-(5)'),
            'propagation': ('室内office确定性dB中心路径损耗，geometry LoS选分支；无随机阴影/快衰落/LoS概率'
                            if indoor else '理想各向同性、极化匹配、远场自由空间；忽略有效geometry的LoS分类'),
            'validity': ('1≤d3D≤150m，0.5<f<100GHz；名义终端1m/固定站3m/层高3m，'
                         '不声明连续高度域或室外有效性' if indoor
                         else 'minimum_distance_m为实验声明的计算下界，不证明具体天线满足远场'),
            'receiver': '所有接收端共享T/NF；T是输入热噪声参考温度，NF只计一次',
            'service': '干扰视为噪声的单链路理想容量；无MAC/重传开销，不积分时间或扣业务量',
            'half_duplex': '接收端同resource非零实际发射时interference/rate无效；不支持自干扰抵消',
        })


class _AnalyticalRadio(RadioModel):
    """两个具体law共用的冻结索引、实际RF归约及输出构造。"""

    def __init__(self, config, layout, context: ExecutionContext):
        import torch

        super().__init__(config, layout, context)
        geometry, network, axes = layout['geometry'], layout['network'], layout['axes']
        point_to_endpoint = {point: endpoint for endpoint, point in enumerate(network.endpoint_to_point)}
        self._desired_path = torch.tensor([item.path for item in geometry.desired],
            dtype=torch.long, device=context.device)
        self._tx = torch.zeros(axes['L'], dtype=torch.long, device=context.device)
        self._rx = torch.zeros_like(self._tx)
        self._candidate = torch.zeros((axes['L'], axes['F']), dtype=torch.bool, device=context.device)
        for binding in network.communication_allocations:
            self._tx[binding.link_index], self._rx[binding.link_index] = binding.tx, binding.rx
            self._candidate[binding.link_index, list(binding.resource_indices)] = True
        sources = [(item.consumer, point_to_endpoint[source.tx], source.path)
                   for item in geometry.interference if item.kind == 'desired' for source in item.sources]
        self._source_link, self._source_endpoint, self._source_path = (
            torch.tensor([source[column] for source in sources], dtype=torch.long, device=context.device)
            for column in range(3))
        self._noise_factor = 10. ** (config['receiver_noise_figure_db'] / 10.)

    def _result(self, inputs, *, reset):
        import torch

        gain, path_valid = self._path_gain(inputs)
        present = inputs.endpoint_masks['present']
        link_valid = (self._candidate[None] & path_valid[:, self._desired_path, None]
                      & present[:, self._tx, None] & present[:, self._rx, None])
        desired_gain = gain[:, self._desired_path]
        allocated = inputs.allocations['link_active']
        power = inputs.allocations['link_power_w'].double()
        signal = torch.where(allocated, power * desired_gain, 0.)
        active = (inputs.simultaneous_transmissions['active']
                  & (inputs.simultaneous_transmissions['power_w'] > 0)
                  & present[:, :, None])
        source_active = active[:, self._source_endpoint]
        source_valid = path_valid[:, self._source_path, None]
        source_power = inputs.simultaneous_transmissions['power_w'][:, self._source_endpoint].double()
        contribution = torch.where(source_active & source_valid,
                                   source_power * gain[:, self._source_path], 0.)
        interference = torch.zeros_like(signal).index_add_(1, self._source_link, contribution)
        unknown = torch.zeros_like(signal, dtype=torch.long).index_add_(
            1, self._source_link, (source_active & ~source_valid).long())
        interference_valid = link_valid & (unknown == 0) & ~active[:, self._rx]
        bandwidth = inputs.resources['bandwidth_hz']
        noise = 1.380649e-23 * self.config['receiver_noise_temperature_k'] * bandwidth * self._noise_factor
        # 噪声溢出会被S/(I+N)隐藏成零；在产生点暴露，不把它当成invalid。
        if not torch.isfinite(noise).all():
            raise RuntimeError('receiver noise: nonfinite numeric result')
        sinr = signal / (interference + noise)
        rate = bandwidth * torch.log1p(sinr) / math.log(2.)
        rate_valid = torch.zeros_like(interference_valid) if reset else interference_valid
        return ModelResult(RadioEvaluationResult(
            LinkMetrics(torch.where(link_valid, desired_gain, 0.).float(),
                        torch.where(link_valid, signal, 0.).float(), link_valid),
            InterferenceMetrics(torch.where(interference_valid, interference, 0.).float(),
                                torch.where(interference_valid, sinr, 0.).float(), interference_valid),
            RateMetrics(torch.where(rate_valid, rate, 0.).float(), rate_valid)), (), {})

    def reset_state(self, state, reset_mask, inputs, rng):
        return self._result(inputs, reset=True)

    def evaluate(self, inputs, state, rng):
        return self._result(inputs, reset=False)


class FreeSpaceRadio(_AnalyticalRadio):
    @classmethod
    def describe(cls):
        return _description('free_space_radio', FreeSpaceConfig, indoor=False)

    def _path_gain(self, inputs):
        import torch

        distance = inputs.geometry['distance_m'].double()
        valid = inputs.geometry['valid'] & ~(distance < self.config['minimum_distance_m'])
        safe_distance = torch.where(valid, distance, 1.)
        gain = (299792458. / (4. * math.pi * safe_distance[:, :, None]
                             * inputs.resources['frequency_hz'])) ** 2
        return torch.where(valid[:, :, None], gain, 0.), valid


class IndoorOfficeRadio(_AnalyticalRadio):
    @classmethod
    def describe(cls):
        return _description('indoor_office_radio', ReceiverNoiseConfig, indoor=True)

    def _path_gain(self, inputs):
        import torch

        distance = inputs.geometry['distance_m'].double()
        valid = inputs.geometry['valid'] & ~((distance < 1.) | (distance > 150.))
        log_distance = torch.log10(torch.where(valid, distance, 1.))[:, :, None]
        log_frequency = torch.log10(inputs.resources['frequency_hz'] / 1e9)
        los_loss = 32.4 + 17.3 * log_distance + 20. * log_frequency
        nlos_loss = torch.maximum(los_loss, 38.3 * log_distance + 17.30 + 24.9 * log_frequency)
        loss = torch.where(inputs.geometry['los'][:, :, None], los_loss, nlos_loss)
        return torch.where(valid[:, :, None], 10. ** (-loss / 10.), 0.), valid
