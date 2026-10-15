"""原生sensor的group唯一测量缓存；设备独立采样，episode时钟由Env拥有。"""

import torch

from muse_sdk.runtime_types import SensorSample
from .scene import entity_name


class SensorSampler:
    def __init__(self, experiment, scene, device):
        self._device = device
        self._world_count = experiment.topology.world_count
        self._physics_dt = experiment.scenario.scene.physics_dt
        self._periods = {}
        self._sources = {}
        self._range_bounds = {}
        self.outputs: dict[str, dict[str, SensorSample]] = {}
        for group in experiment.scenario.entity_groups:
            group_id = group.group_id
            if 'sensor:' + group_id not in experiment.activation:
                continue
            profile = experiment.sensors[group_id]
            self._periods[group_id] = {}
            self._sources[group_id] = {}
            self.outputs[group_id] = {}
            shape = (self._world_count, group.count)
            for signal, metadata in profile['outputs'].items():
                self._periods[group_id][signal] = metadata['period_ticks']
                if metadata['sensor_type'] == 'rangefinder':
                    self._range_bounds[group_id, signal] = (metadata['min_range_m'], metadata['max_range_m'])
                self._sources[group_id][signal] = [
                    scene.sensors[f'{entity_name(group_id, slot)}/{metadata["sensor_name"]}']
                    for slot in range(group.count)]
                self.outputs[group_id][signal] = SensorSample(
                    torch.zeros((*shape, metadata['width']), dtype=torch.float32, device=device),
                    torch.zeros(shape, dtype=torch.bool, device=device),
                    torch.zeros(shape, dtype=torch.float32, device=device),
                    torch.zeros(shape, dtype=torch.int64, device=device))

    def due(self, ticks):
        return {group: {signal: ticks.remainder(period) == 0 for signal, period in periods.items()}
                for group, periods in self._periods.items()}

    def capture(self, ticks, masks):
        """仅复制调度器指定world；不去重或重新判定采样相位。"""
        for group, signals in masks.items():
            for signal, mask in signals.items():
                if not mask.any():
                    continue
                sources = self._sources[group][signal]
                measurement = torch.stack([source.data[mask] for source in sources], dim=1)
                if not torch.isfinite(measurement).all():
                    raise RuntimeError(f'nonfinite sensor measurement: {group}/{signal}')
                sample = self.outputs[group][signal]
                if (group, signal) in self._range_bounds:
                    minimum, maximum = self._range_bounds[group, signal]
                    valid = ((measurement >= minimum) & (measurement <= maximum)).squeeze(-1)
                    measurement = torch.where(valid.unsqueeze(-1), measurement, maximum)
                    sample.valid[mask] = valid
                else:
                    sample.valid[mask] = True
                sample.data[mask] = measurement
                sample.sample_tick[mask] = ticks[mask, None]
                sample.age[mask] = 0

    def refresh_age(self, ticks):
        for signals in self.outputs.values():
            for sample in signals.values():
                sample.age.copy_((ticks[:, None] - sample.sample_tick) * self._physics_dt)

    def reset(self, env_ids):
        """从Env已forward的状态捕获选中world的新episode初值。"""
        mask = torch.zeros(self._world_count, dtype=torch.bool, device=self._device)
        mask[env_ids] = True
        ticks = torch.zeros(self._world_count, dtype=torch.int64, device=self._device)
        self.capture(ticks, {group: {signal: mask for signal in signals}
                             for group, signals in self.outputs.items()})
