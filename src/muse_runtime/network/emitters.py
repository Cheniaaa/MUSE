"""固定endpoint集合的RF请求过程；只消费宿主episode tick，不推进物理。"""
import torch

from muse_sdk.network import EmitterBinding


class EmitterProcess:
    def __init__(self, binding: EmitterBinding, *, batch_size, device, rng):
        self._behavior = binding.behavior
        self._config = binding.config
        self._instance_id = 'emitter:' + binding.emitter_id
        self._rng = rng
        shape = (batch_size, len(binding.endpoint_indices))
        self.active = torch.zeros(shape, dtype=torch.bool, device=device)
        self.power_w = torch.zeros(shape, dtype=torch.float32, device=device)
        self.channel = torch.zeros(shape, dtype=torch.int32, device=device)

        config = self._config
        if self._behavior == 'markov':
            self._active = torch.tensor([state.active for state in config.states], dtype=torch.bool, device=device)
            self._power = torch.tensor([state.power_w for state in config.states], dtype=torch.float32, device=device)
            self._channel = torch.tensor([state.channel_id for state in config.states], dtype=torch.int32, device=device)
            self._transition = torch.tensor(config.transition_matrix, dtype=torch.float64, device=device)
            if config.initial_distribution is not None:
                self._initial_distribution = torch.tensor(config.initial_distribution, dtype=torch.float64, device=device)
        else:
            self._power = torch.tensor(config.power_w, dtype=torch.float32, device=device)
            if self._behavior == 'sweep':
                self._channel = torch.tensor(config.channel_ids, dtype=torch.int32, device=device)
            else:
                self._channel = torch.tensor(config.channel_id, dtype=torch.int32, device=device)
            if self._behavior == 'periodic':
                self._durations = torch.tensor([config.on_ticks, config.off_ticks], dtype=torch.int64, device=device)
            else:
                self._active = torch.tensor(config.active, dtype=torch.bool, device=device)

        if self._behavior != 'fixed':
            self.state_index = torch.zeros(batch_size, dtype=torch.int64, device=device)
            self.next_switch_tick = torch.zeros_like(self.state_index)

    def _sample(self, worlds, probabilities, purpose):
        # 每world只抽一个联合state；初态与转移使用独立用途流。
        draws = self._rng.uniform(worlds, self._instance_id, purpose, ())
        return (draws[:, None] >= probabilities.cumsum(dim=-1)).sum(dim=-1)

    def _publish(self, worlds):
        if self._behavior == 'markov':
            indices = self.state_index[worlds]
            self.active[worlds] = self._active[indices]
            self.power_w[worlds] = self._power[indices]
            self.channel[worlds] = self._channel[indices]
        elif self._behavior == 'periodic':
            on = self.state_index[worlds, None] == 0
            self.active[worlds] = on
            self.power_w[worlds] = self._power * on
            self.channel[worlds] = self._channel
        else:
            self.active[worlds] = self._active
            self.power_w[worlds] = self._power
            self.channel[worlds] = (self._channel[self.state_index[worlds], None]
                                   if self._behavior == 'sweep' else self._channel)

    @torch.no_grad()
    def reset(self, world_mask):
        worlds = world_mask.nonzero().flatten()
        if not worlds.numel():
            return
        config = self._config
        if self._behavior == 'periodic':
            on = config.phase_ticks < config.on_ticks
            self.state_index[worlds] = 0 if on else 1
            self.next_switch_tick[worlds] = (config.on_ticks if on else config.on_ticks + config.off_ticks) - config.phase_ticks
        elif self._behavior == 'sweep':
            self.state_index[worlds] = config.initial_index
            self.next_switch_tick[worlds] = config.dwell_ticks
        elif self._behavior == 'markov':
            self.state_index[worlds] = (config.initial_state if config.initial_state is not None
                                       else self._sample(worlds, self._initial_distribution, 'init'))
            self.next_switch_tick[worlds] = config.dwell_ticks
        self._publish(worlds)

    @torch.no_grad()
    def advance(self, sample_tick):
        if self._behavior == 'fixed':
            return
        worlds = (sample_tick >= self.next_switch_tick).nonzero().flatten()
        if not worlds.numel():
            return
        config = self._config
        if self._behavior == 'periodic':
            self.state_index[worlds] = 1 - self.state_index[worlds]
            self.next_switch_tick[worlds] += self._durations[self.state_index[worlds]]
        elif self._behavior == 'sweep':
            self.state_index[worlds] = (self.state_index[worlds] + 1) % len(config.channel_ids)
            self.next_switch_tick[worlds] += config.dwell_ticks
        else:
            self.state_index[worlds] = self._sample(worlds, self._transition[self.state_index[worlds]], 'transition')
            self.next_switch_tick[worlds] += config.dwell_ticks
        # resolve已保证相邻epoch至多跨一个名义边界；不按延后的sample_tick累积漂移。
        self._publish(worlds)
