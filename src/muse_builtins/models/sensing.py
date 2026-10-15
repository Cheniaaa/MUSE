"""合成Gaussian路程观测与自由variance nuisance二维位置EFIM。"""
import math

from pydantic import Field

from muse_sdk.models import ModelResult, SensingModel, SensingResult

from .analytical_radio import ReceiverNoiseConfig


class FreeSpaceSensingConfig(ReceiverNoiseConfig):
    target_rcs_m2: float = Field(gt=0)
    minimum_distance_m: float = Field(gt=0)


def _description(model_id, *, measurement, fim):
    fields = [
        ('geometry', 'distance_m', ['B', 'P'], 'float32', 'm', 'none'),
        ('geometry', 'valid', ['B', 'P'], 'bool', '1', 'none'),
        ('geometry', 'los', ['B', 'P'], 'bool', '1', 'none'),
        ('endpoint_masks', 'present', ['B', 'E'], 'bool', '1', 'none'),
        ('resources', 'frequency_hz', ['F'], 'float64', 'Hz', 'none'),
        ('resources', 'bandwidth_hz', ['F'], 'float64', 'Hz', 'none'),
        ('simultaneous_transmissions', 'active', ['B', 'E', 'F'], 'bool', '1', 'none'),
        ('simultaneous_transmissions', 'power_w', ['B', 'E', 'F'], 'float32', 'W', 'none'),
    ]
    if fim:
        fields.append(('geometry', 'positions_world', ['B', 'Q', 3], 'float32', 'm', 'world_z_up'))
    inputs = [dict(path=['network', view, name], phase='post_physics', tensor=dict(
        shape=shape, dtype=dtype, units=units, frame=frame, rate='radio_epoch'))
        for view, name, shape, dtype, units, frame in fields]
    inputs.append(dict(path=['network', 'interval_s'], phase='post_physics', tensor=dict(
        shape=['B'], dtype='float64', units='s', frame='none', rate='physics')))
    outputs = {}
    if measurement:
        outputs['measurement'] = dict(shape=['B', 'S', 'F'], dtype='float64', units='m',
            frame='none', rate='radio_epoch', time_semantics='interval_sample')
    if fim:
        outputs['fim'] = dict(shape=['B', 'S', 'F', 2, 2], dtype='float64', units='m^-2',
            frame='world_z_up', rate='radio_epoch', time_semantics='increment')
    return dict(model_id=model_id, config_schema=FreeSpaceSensingConfig, input_fields=inputs,
        state_scope='world', state_schema={}, diagnostics_schema={}, output_spec=outputs,
        assumptions={
            'observation': 'z~Normal(h,a/T)，h为双站总路程；目标二维位置未知，高度与收发位置已知，decision内目标固定',
            'nuisance': '每观测a是独立自由未知variance nuisance；nominal a按回波/有效PSD标定；EFIM=T/a ggᵀ，不利用variance位置导数',
            'propagation': '理想各向同性极化匹配远场点散射，恒定双站RCS；P_echo=P_tx σ λ²/((4π)³ d_t² d_r²)',
            'waveform': '时间同步；平坦零中心基带谱，β²=B²/12；未知载波相位不提供位置测量',
            'receiver': 'T/NF热噪声只计一次；理想抑制自身泄漏与本次tx direct，其他实际同resource直达功率作为白噪声',
            'separation': '目标关联已知、回波可分离；各关系/resource/epoch有效Gaussian噪声条件独立，无杂波/多径/同步偏差',
            'blocker': '有效非LoS路径不透射；未知必要geometry不能作为零干扰；minimum_distance_m只是计算域下界',
            'interpretation': '合成测距统计参考，不是波形处理器或实测精度；局部无偏位置界不保证MLE存在/可达，完整EFIM不等于最新样本actor可达误差',
        })


class _FreeSpaceSensing(SensingModel):
    _measurement = False
    _fim = False

    def __init__(self, config, layout, context):
        import torch

        super().__init__(config, layout, context)
        geometry, network = layout['geometry'], layout['network']
        point_to_endpoint = {point: e for e, point in enumerate(network.endpoint_to_point)}
        relations, seen = [], set()
        for item in geometry.sensing:
            outbound, inbound = geometry.paths[item.outbound], geometry.paths[item.inbound]
            identity = (outbound.tx, outbound.rx, inbound.rx)
            if identity in seen:
                raise ValueError('Duplicate physical sensing observation (tx, target, rx)')
            seen.add(identity)
            relations.append((item.outbound, item.inbound, point_to_endpoint[outbound.tx],
                point_to_endpoint[inbound.rx], outbound.tx, outbound.rx, inbound.rx))
        self._outbound, self._inbound, self._tx, self._rx, self._tx_point, self._target, self._rx_point = (
            torch.tensor([row[i] for row in relations], dtype=torch.long, device=context.device) for i in range(7))
        sources = [(item.consumer, point_to_endpoint[source.tx], source.path)
            for item in geometry.interference if item.kind == 'sensing' for source in item.sources
            if source.tx not in (relations[item.consumer][4], relations[item.consumer][6])]
        self._source_s, self._source_e, self._source_path = (
            torch.tensor([row[i] for row in sources], dtype=torch.long, device=context.device) for i in range(3))
        self._shape = (layout['axes']['B'], layout['axes']['S'], layout['axes']['F'])
        self._noise_factor = 10. ** (config['receiver_noise_figure_db'] / 10.)

    def _output(self, measurement, fim, measurement_valid, fim_valid):
        validity, semantics = {}, {}
        if self._measurement:
            validity['measurement'], semantics['measurement'] = measurement_valid, 'interval_sample'
        if self._fim:
            validity['fim'], semantics['fim'] = fim_valid, 'increment'
        return ModelResult(SensingResult(measurement, fim, validity, semantics), (), {})

    def reset_state(self, state, reset_mask, inputs, rng):
        import torch

        zero = torch.zeros(self._shape, dtype=torch.float64, device=self.context.device)
        invalid = torch.zeros(self._shape, dtype=torch.bool, device=self.context.device)
        return self._output(zero if self._measurement else None,
            torch.zeros((*self._shape, 2, 2), dtype=torch.float64, device=self.context.device) if self._fim else None,
            invalid, invalid.clone() if self._measurement and self._fim else invalid)

    def evaluate(self, inputs, state, rng):
        import torch

        distance = inputs.geometry['distance_m'].double()
        valid, los = inputs.geometry['valid'], inputs.geometry['los']
        present = inputs.endpoint_masks['present']
        power = inputs.simultaneous_transmissions['power_w'].double()
        active = inputs.simultaneous_transmissions['active'] & (power > 0) & present[:, :, None]
        exists = active[:, self._tx] & present[:, self._rx, None]
        blocked = ((valid[:, self._outbound] & ~los[:, self._outbound])
                   | (valid[:, self._inbound] & ~los[:, self._inbound]))[:, :, None]
        path_known = (valid[:, self._outbound] & valid[:, self._inbound]
            & ~(distance[:, self._outbound] < self.config['minimum_distance_m'])
            & ~(distance[:, self._inbound] < self.config['minimum_distance_m']))[:, :, None]
        visible = exists & ~blocked & path_known
        dt = torch.where(visible, distance[:, self._outbound, None], 1.)
        dr = torch.where(visible, distance[:, self._inbound, None], 1.)
        # 只检查应使用的距离；valid=false或已知没有观测的NaN占位不参与传播。
        if not torch.isfinite(dt).all() or not torch.isfinite(dr).all():
            raise RuntimeError('sensing distance: nonfinite numeric result')
        source_active = active[:, self._source_e] & visible[:, self._source_s]
        source_valid = valid[:, self._source_path, None]
        source_los = los[:, self._source_path, None]
        source_domain = ~(distance[:, self._source_path, None] < self.config['minimum_distance_m'])
        unknown = source_active & (~source_valid | (source_los & ~source_domain))
        unknown_count = torch.zeros(self._shape, dtype=torch.long, device=self.context.device).index_add_(1, self._source_s, unknown.long())
        observed = visible & (unknown_count == 0)
        contributing = source_active & source_valid & source_los & source_domain & observed[:, self._source_s]
        source_distance = torch.where(contributing, distance[:, self._source_path, None], 1.)
        if not torch.isfinite(source_distance).all():
            raise RuntimeError('sensing interference distance: nonfinite numeric result')
        frequency, bandwidth = inputs.resources['frequency_hz'], inputs.resources['bandwidth_hz']
        gain = (299792458. / (4 * math.pi * source_distance * frequency))**2
        contribution = torch.where(contributing, power[:, self._source_e] * gain, 0.)
        interference = torch.zeros(self._shape, dtype=torch.float64, device=self.context.device).index_add_(1, self._source_s, contribution)
        echo = power[:, self._tx] * self.config['target_rcs_m2'] * (299792458. / frequency)**2 / ((4 * math.pi)**3 * dt**2 * dr**2)
        noise = 1.380649e-23 * self.config['receiver_noise_temperature_k'] * self._noise_factor + interference / bandwidth
        beta2 = bandwidth**2 / 12
        # 正量在除法前检查，防止溢出/下溢被后续除法隐藏；只在有观测的cell要求科学量。
        for name, value in (('echo', echo), ('noise PSD', noise), ('RMS bandwidth squared', beta2.expand(self._shape))):
            if not torch.isfinite(value[observed]).all() or (value[observed] <= 0).any():
                raise RuntimeError(f'sensing {name}: expected finite positive numeric result')
        safe_echo = torch.where(observed, echo, 1.)
        a = 299792458.**2 * noise / (8 * math.pi**2 * beta2 * safe_echo)
        variance = a / inputs.interval_s[:, None, None]
        if not torch.isfinite(a[observed]).all() or (a[observed] <= 0).any():
            raise RuntimeError('sensing a: expected finite positive numeric result')
        if not torch.isfinite(variance[observed]).all() or (variance[observed] <= 0).any():
            raise RuntimeError('sensing variance: expected finite positive numeric result')
        variance = torch.where(observed, variance, 1.)
        measurement = None
        if self._measurement:
            noise_sample = torch.stack([torch.randn(self._shape[1:], dtype=torch.float64,
                device=self.context.device, generator=rng.generator(world, self.layout['instance_id'], 'measurement'))
                for world in range(self._shape[0])])
            measurement = torch.where(observed, dt + dr + variance.sqrt() * noise_sample, 0.)
        fim = None
        if self._fim:
            positions = inputs.geometry['positions_world'].double()
            q = positions[:, self._target, None, :2]
            gradient = ((q - positions[:, self._tx_point, None, :2]) / dt[..., None]
                        + (q - positions[:, self._rx_point, None, :2]) / dr[..., None])
            fim = torch.where(observed[..., None, None],
                gradient[..., :, None] * gradient[..., None, :] / variance[..., None, None], 0.)
        fim_valid = ~exists | blocked | (path_known & (unknown_count == 0))
        return self._output(measurement, fim, observed, fim_valid)


class FreeSpacePathLengthMeasurement(_FreeSpaceSensing):
    _measurement = True

    @classmethod
    def describe(cls):
        return _description('free_space_path_length_measurement', measurement=True, fim=False)


class FreeSpacePositionEFIM2D(_FreeSpaceSensing):
    _fim = True

    @classmethod
    def describe(cls):
        return _description('free_space_position_efim_2d', measurement=False, fim=True)


class FreeSpacePathLengthEFIM2D(_FreeSpaceSensing):
    _measurement = True
    _fim = True

    @classmethod
    def describe(cls):
        return _description('free_space_path_length_efim_2d', measurement=True, fim=True)
