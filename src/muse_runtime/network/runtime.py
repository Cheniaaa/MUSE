"""无线epoch唯一分配与服务提交；物理时钟和几何刷新归MuseEnv。"""
from types import MappingProxyType, SimpleNamespace

import torch

from muse_sdk.models import ExecutionContext, ModelResetMask, RadioInputs, SensingInputs
from muse_sdk.network import ServiceView, SensingFieldView, WirelessSnapshot, network_axes
from muse_runtime.model_execution import ModelInstance
from .emitters import EmitterProcess


class WirelessRuntime:
    @torch.no_grad()
    def __init__(self, spec, model_bindings, model_types, *, batch_size, device, rng, geometry_layout, geometry_view):
        self.spec = spec
        self._batch, self._device = batch_size, device
        b, e, f, l = batch_size, len(spec.endpoints), len(spec.resources), spec.link_count
        self._worlds = torch.ones(b, dtype=torch.bool, device=device)
        self._sample_tick = torch.zeros(b, dtype=torch.int64, device=device)
        self._time = torch.zeros(b, dtype=torch.float64, device=device)
        self._interval = torch.zeros_like(self._time)
        self._power = torch.zeros((b, e), device=device)
        self._channel = torch.tensor([spec.resources[limit.resource_indices[0]].channel_id
            for limit in spec.endpoint_limits], dtype=torch.int32, device=device).expand(b, -1).clone()
        self._active = torch.zeros((b, e), dtype=torch.bool, device=device)
        self._channel_ids = torch.tensor([r.channel_id for r in spec.resources], device=device)
        self._budgets = torch.tensor([x.max_power_w for x in spec.endpoint_limits], device=device)
        self._exclusive = torch.tensor([r.sharing == 'exclusive' for r in spec.resources], dtype=torch.bool, device=device)
        transmissions = SimpleNamespace(active=torch.zeros((b, e, f), dtype=torch.bool, device=device),
            power_w=torch.zeros((b, e, f), device=device))
        allocations = SimpleNamespace()
        if l:
            allocations.link_active = torch.zeros((b, l, f), dtype=torch.bool, device=device)
            allocations.link_power_w = torch.zeros((b, l, f), device=device)
        services = {binding.service_id: ServiceView(
            torch.full((b, len(binding.link_indices)), binding.initial_bits, dtype=torch.float64, device=device),
            torch.zeros((b, len(binding.link_indices)), dtype=torch.float64, device=device),
            torch.zeros((b, len(binding.link_indices)), dtype=torch.bool, device=device),
            torch.zeros(b, dtype=torch.int64, device=device), torch.zeros(b, dtype=torch.int64, device=device))
            for binding in spec.services}
        self.network = SimpleNamespace(endpoint_masks=SimpleNamespace(present=torch.ones((b, e), dtype=torch.bool, device=device)),
            resources=SimpleNamespace(
                frequency_hz=torch.tensor([r.center_frequency_hz for r in spec.resources], dtype=torch.float64, device=device),
                bandwidth_hz=torch.tensor([r.bandwidth_hz for r in spec.resources], dtype=torch.float64, device=device)),
            allocations=allocations, simultaneous_transmissions=transmissions, services=services,
            interval_s=self._interval)
        if geometry_view is not None:
            self.network.geometry = geometry_view
        self.emitters = [EmitterProcess(binding, batch_size=b, device=device, rng=rng)
                         for binding in spec.background_emitters]
        self.models = {}
        self.sensing_outputs = {}
        self._increment_fields = []
        self._interval_sample_fields = []
        axes = MappingProxyType(network_axes(spec, geometry=None, batch_size=b))
        self._input_views = dict(geometry=geometry_view.mapping if geometry_view is not None else MappingProxyType({}),
            **{name: MappingProxyType(vars(getattr(self.network, name))) for name in
               ('endpoint_masks', 'resources', 'allocations', 'simultaneous_transmissions')},
            sim_time=self._time, interval_s=self._interval)
        for kind in ('radio', 'sensing'):
            if kind not in model_bindings:
                continue
            binding = model_bindings[kind]
            model = ModelInstance(model_types[kind], descriptor=binding.descriptor,
                config=MappingProxyType(binding.config),
                layout=MappingProxyType(dict(axes=axes, instance_id='model:'+kind, geometry=geometry_layout, network=spec)),
                context=ExecutionContext(device), available_inputs=binding.input_fields, rng=rng,
                bound_slots=torch.ones((b, l, f), dtype=torch.bool, device=device) if kind == 'radio' else None)
            self.models[kind] = model
            if kind == 'radio':
                self.network.radio = SimpleNamespace(**model.output._asdict(), sample_tick=self._sample_tick,
                    sample_time=model.sim_time, interval_s=model.interval_s)
            else:
                for name, semantics in model.output.time_semantics.items():
                    data, valid = getattr(model.output, name), model.output.field_valid[name]
                    if semantics != 'snapshot':
                        if semantics == 'increment':
                            self._increment_fields.append(name)
                        else:
                            self._interval_sample_fields.append(name)
                        data, valid = torch.zeros_like(data), torch.zeros_like(valid)
                        start, end = torch.zeros_like(self._sample_tick), torch.zeros_like(self._sample_tick)
                    else:
                        start = end = self._sample_tick
                    self.sensing_outputs[name] = SensingFieldView(data, valid, start, end)

    def _inputs(self, kind):
        if kind == 'radio':
            return RadioInputs(**self._input_views)
        return SensingInputs(**self._input_views,
            radio_outputs=self.models['radio'].output if 'radio' in self.models else None)

    def _allocate(self, worlds):
        for binding, emitter in zip(self.spec.background_emitters, self.emitters):
            indices = list(binding.endpoint_indices)
            self._active[:, indices] = emitter.active
            self._power[:, indices] = emitter.power_w
            self._channel[:, indices] = emitter.channel
        active = self._active[worlds] & (self._power[worlds] > 0)
        selected = self._channel[worlds, :, None] == self._channel_ids
        transmitted = active[..., None] & selected
        power = torch.where(transmitted, self._power[worlds, :, None], 0.)
        if (power.sum(-1) > self._budgets).any():
            raise RuntimeError('RF power exceeds endpoint budget')
        if (transmitted.sum(1)[:, self._exclusive] > 1).any():
            raise RuntimeError('RF exclusive resource has simultaneous transmitters')
        output = self.network.simultaneous_transmissions
        output.active[worlds] = transmitted
        output.power_w[worlds] = power
        if self.spec.link_count:
            allocations = self.network.allocations
            allocations.link_active[worlds] = False
            allocations.link_power_w[worlds] = 0
            rows = worlds.nonzero().flatten()
            for binding in self.spec.communication_allocations:
                for resource in binding.resource_indices:
                    allocations.link_active[rows, binding.link_index, resource] = transmitted[:, binding.tx, resource]
                    allocations.link_power_w[rows, binding.link_index, resource] = power[:, binding.tx, resource]

    @torch.no_grad()
    def reset(self, world_mask, *, sample_tick):
        if not world_mask.any():
            return
        self._sample_tick[world_mask] = sample_tick[world_mask]
        self._time[world_mask] = sample_tick[world_mask].to(torch.float64)*self.spec.physics_dt
        self._interval[world_mask] = 0
        self._active[world_mask] = False
        self._power[world_mask] = 0
        for emitter in self.emitters:
            emitter.reset(world_mask)
        self._allocate(world_mask)
        for binding in self.spec.services:
            view = self.network.services[binding.service_id]
            view.remaining_bits[world_mask] = binding.initial_bits
            view.served_bits[world_mask] = 0
            view.valid[world_mask] = False
            view.interval_start_tick[world_mask] = 0
            view.interval_end_tick[world_mask] = 0
        for kind, model in self.models.items():
            model.reset(ModelResetMask(world_mask), self._inputs(kind))
        for name in self._increment_fields:
            view = self.sensing_outputs[name]
            view.data[world_mask] = 0
            view.valid[world_mask] = False
            view.interval_start_tick[world_mask] = 0
            view.interval_end_tick[world_mask] = 0
        for name in self._interval_sample_fields:
            view = self.sensing_outputs[name]
            view.data[world_mask] = 0
            view.valid[world_mask] = False
            view.interval_start_tick[world_mask] = sample_tick[world_mask]
            view.interval_end_tick[world_mask] = sample_tick[world_mask]

    @torch.no_grad()
    def begin_decision(self, start_tick):
        for view in self.network.services.values():
            view.served_bits.zero_()
            view.valid.fill_(True)
            view.interval_start_tick.copy_(start_tick)
            view.interval_end_tick.copy_(start_tick)
        for name in self._increment_fields:
            view = self.sensing_outputs[name]
            view.data.zero_()
            view.valid.fill_(True)
            view.interval_start_tick.copy_(start_tick)
            view.interval_end_tick.copy_(start_tick)

    @torch.no_grad()
    def begin_epoch(self, sample_tick, *, interval_ticks, radio_requests):
        self._sample_tick.copy_(sample_tick)
        self._time.copy_(sample_tick.to(torch.float64)*self.spec.physics_dt)
        self._interval.fill_(interval_ticks*self.spec.physics_dt)
        for binding in self.spec.role_controls:
            request = radio_requests[binding.role_id]
            indices = list(binding.endpoint_indices)
            self._power[:, indices] = request.power_w
            self._channel[:, indices] = request.channel
            self._active[:, indices] = request.power_w > 0
        for emitter in self.emitters:
            emitter.advance(sample_tick)
        self._allocate(self._worlds)
        for kind, model in self.models.items():
            model.evaluate(self._inputs(kind))

    @torch.no_grad()
    def end_epoch(self, end_tick):
        for binding in self.spec.services:
            rate = self.models['radio'].output.rate
            links, resources = list(binding.link_indices), list(binding.resource_indices)
            active = self.network.allocations.link_active[:, links][:, :, resources]
            valid = rate.valid[:, links][:, :, resources]
            values = rate.rate_bps[:, links][:, :, resources]
            budget = (torch.where(active & valid, values.double(), 0.)*self._interval[:, None, None]).sum(-1)
            view = self.network.services[binding.service_id]
            served = torch.minimum(budget, view.remaining_bits)
            total = view.served_bits + served
            if not torch.isfinite(budget).all() or not torch.isfinite(total).all():
                raise RuntimeError('nonfinite service budget/accumulation')
            view.remaining_bits.sub_(served)
            view.served_bits.copy_(total)
            view.valid.logical_and_((~active | valid).all(-1))
            view.interval_end_tick.copy_(end_tick)
        for name in self._increment_fields:
            model = self.models['sensing']
            valid = model.output.field_valid[name]
            data = getattr(model.output, name)
            view = self.sensing_outputs[name]
            selected = valid[..., None, None] if name == 'fim' else valid
            total = view.data + torch.where(selected, data, 0.)
            if not torch.isfinite(total).all():
                raise RuntimeError('nonfinite sensing increment accumulation')
            view.data.copy_(total)
            view.valid.logical_and_(valid)
            view.interval_end_tick.copy_(end_tick)

        for name in self._interval_sample_fields:
            output = self.models['sensing'].output
            valid = output.field_valid[name]
            view = self.sensing_outputs[name]
            view.data.copy_(torch.where(valid, getattr(output, name), 0.))
            view.valid.copy_(valid)
            view.interval_start_tick.copy_(self._sample_tick)
            view.interval_end_tick.copy_(end_tick)

    def snapshot(self):
        """借用固定缓存；Env在auto reset前统一clone。"""
        radio = self.models.get('radio')
        return WirelessSnapshot(radio.output if radio is not None else None,
            self._sample_tick if radio is not None else None,
            radio.interval_s if radio is not None else None,
            self.network.services, self.sensing_outputs)
