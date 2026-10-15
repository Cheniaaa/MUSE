"""单模型计算边界与world缓存提交；不拥有物理或无线调度。"""
from collections.abc import Mapping
from types import MappingProxyType

import torch

from muse_sdk.config import ConfigError
from muse_sdk.descriptor import RadioDescriptor
from muse_sdk.models import (MODEL_INPUT_VIEWS, RADIO_BLOCK_TYPES, RADIO_FIELDS, ModelResetMask,
    ModelResult, RadioEvaluationResult, RadioInputs, SensingInputs, SensingResult)


class ModelInstance:
    @torch.no_grad()
    def __init__(self, model_type, *, descriptor, config, layout, context,
                 available_inputs, rng, bound_slots=None):
        self._name = layout['instance_id']
        self._axes = layout['axes']
        if not isinstance(self._name, str) or not self._name:
            raise ConfigError('layout.instance_id', 'Expected a stable nonempty binding identity')
        if any(type(value) is not int or value <= 0 for value in self._axes.values()):
            raise ConfigError('layout.axes', 'Expected positive integer axis sizes')
        if 'B' not in self._axes:
            raise ConfigError('layout.axes.B', 'Missing world axis')
        self._batch = self._axes['B']
        self._device = torch.device(context.device)
        self._descriptor = descriptor
        self._radio = isinstance(descriptor, RadioDescriptor)
        self._rng = rng
        self._checked_phases = set()
        self._state_specs = self._resolve_specs(descriptor.state_schema)
        self._diagnostic_specs = self._resolve_specs(descriptor.diagnostics_schema)
        self._bound_slots = bound_slots
        if self._radio:
            self._radio_shape = self._shape(['B', 'L', 'F'])
            self._tensor(bound_slots, self._radio_shape, 'bool', 'bound_slots')
            self._sensing_specs = {}
        else:
            if bound_slots is not None:
                raise ConfigError(self._name + '.bound_slots', 'Only radio models use bound_slots')
            self._sensing_specs = self._resolve_specs({name: getattr(descriptor.output_spec, name)
                for name in ('measurement', 'fim') if getattr(descriptor.output_spec, name) is not None})

        available = {}
        for item in available_inputs:
            key = (tuple(item.path), item.phase)
            if key in available:
                raise ConfigError(self._name + '.inputs', 'Duplicate host field')
            available[key] = item.tensor
        self._fields = {'reset': [], 'post_physics': []}
        for item in descriptor.input_fields:
            shape = self._shape(item.tensor.shape)
            expected = item.tensor.model_copy(update={'shape': list(shape)})
            if available.get((tuple(item.path), item.phase)) != expected:
                raise ConfigError(self._name + '.' + '.'.join(item.path), 'Missing or mismatched host field')
            self._fields[item.phase].append((tuple(item.path), shape, item.tensor.dtype))

        self._model = model_type(config, layout, context)
        state = self._model.initial_state((self._batch,))
        all_worlds = torch.ones(self._batch, dtype=torch.bool, device=self._device)
        self._check_state(state, all_worlds)
        self.state = {name: value.clone() for name, value in state.items()} if self._state_specs else ()
        self.diagnostics = {name: torch.zeros(shape, dtype=getattr(torch, dtype), device=self._device)
                            for name, (shape, dtype) in self._diagnostic_specs.items()}
        # 占位缓存仅供shape probe；episode初态由宿主唯一显式reset建立。
        if self._radio:
            self.output = RadioEvaluationResult(*(RADIO_BLOCK_TYPES[name](*(
                torch.zeros(self._radio_shape, dtype=torch.bool if field == 'valid' else torch.float32,
                            device=self._device) for field in fields))
                if name in descriptor.output_spec.output_blocks else None
                for name, fields in RADIO_FIELDS.items()))
        else:
            values = {name: torch.zeros(shape, dtype=getattr(torch, dtype), device=self._device)
                      for name, (shape, dtype) in self._sensing_specs.items()}
            self.output = SensingResult(values.get('measurement'), values.get('fim'),
                {name: torch.zeros(shape[:-2] if name == 'fim' else shape, dtype=torch.bool,
                                   device=self._device) for name, (shape, dtype) in self._sensing_specs.items()},
                {name: getattr(descriptor.output_spec, name).time_semantics for name in self._sensing_specs})
        self.sim_time = torch.zeros(self._batch, dtype=torch.float64, device=self._device)
        self.interval_s = torch.zeros_like(self.sim_time)

    def _shape(self, shape):
        result = []
        for axis in shape:
            if isinstance(axis, str):
                if axis not in self._axes:
                    raise ConfigError(self._name + '.layout.axes', f'Unknown axis {axis}')
                axis = self._axes[axis]
            if type(axis) is not int or axis <= 0:
                raise ConfigError(self._name + '.shape', 'Expected positive dimensions')
            result.append(axis)
        return tuple(result)

    def _resolve_specs(self, specs):
        return {name: (self._shape(spec.shape), spec.dtype) for name, spec in specs.items()}

    def _fail(self, field, message):
        raise RuntimeError(f'{self._name}.{field}: {message}')

    def _tensor(self, value, shape, dtype, field):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != getattr(torch, dtype):
            self._fail(field, 'tensor shape/dtype mismatch')
        if value.device != self._device:
            self._fail(field, 'tensor device mismatch')

    def _inputs(self, inputs, phase, worlds):
        input_type = RadioInputs if self._radio else SensingInputs
        if not isinstance(inputs, input_type):
            self._fail('inputs', f'Expected {input_type.__name__}')
        cold = phase not in self._checked_phases
        views = {name: {} for name in MODEL_INPUT_VIEWS}
        radio_output = None
        for path, shape, dtype in self._fields[phase]:
            try:
                if path == ('physics_outputs', 'sim_time'):
                    value = inputs.sim_time
                elif path == ('network', 'interval_s'):
                    value = inputs.interval_s
                elif path[:2] == ('network', 'radio'):
                    if not isinstance(inputs.radio_outputs, RadioEvaluationResult):
                        self._fail('inputs.radio_outputs', 'Missing radio result')
                    block = getattr(inputs.radio_outputs, path[2])
                    if not isinstance(block, RADIO_BLOCK_TYPES[path[2]]):
                        self._fail('inputs.' + '.'.join(path), 'Missing declared radio block')
                    value = getattr(block, path[3])
                    radio_output = inputs.radio_outputs
                else:
                    value = getattr(inputs, path[1])[path[2]]
                    views[path[1]][path[2]] = value
            except KeyError as exc:
                self._fail('inputs.' + '.'.join(path), f'Missing declared field: {exc}')
            if cold:
                self._tensor(value, shape, dtype, 'inputs.' + '.'.join(path))
        if cold:
            for name in ('sim_time', 'interval_s'):
                self._tensor(getattr(inputs, name), (self._batch,), 'float64', 'inputs.' + name)
        time, interval = inputs.sim_time[worlds], inputs.interval_s[worlds]
        if not torch.isfinite(time).all() or (time < 0).any():
            self._fail('inputs.sim_time', 'Expected finite nonnegative time')
        wrong_interval = (interval != 0).any() if phase == 'reset' else (interval <= 0).any()
        if not torch.isfinite(interval).all() or wrong_interval:
            self._fail('inputs.interval_s', 'Expected zero reset interval or positive evaluate interval')
        self._checked_phases.add(phase)
        values = {name: MappingProxyType(value) for name, value in views.items()}
        values.update(sim_time=inputs.sim_time, interval_s=inputs.interval_s)
        if not self._radio:
            values['radio_outputs'] = radio_output
        return input_type(**values)

    def _check_mapping(self, value, specs, worlds, field):
        if not isinstance(value, Mapping) or value.keys() != specs.keys():
            self._fail(field, 'Expected fixed tensor fields')
        for name, (shape, dtype) in specs.items():
            item = value[name]
            self._tensor(item, shape, dtype, field + '.' + name)
            if item.is_floating_point() and not torch.isfinite(item[worlds]).all():
                self._fail(field + '.' + name, 'nonfinite value')

    def _check_state(self, state, worlds):
        if self._state_specs:
            self._check_mapping(state, self._state_specs, worlds, 'state')
        elif not isinstance(state, tuple) or len(state) != 0:
            self._fail('state', 'Stateless models return ()')

    def _check_result(self, result, worlds, *, reset):
        if not isinstance(result, ModelResult):
            self._fail('result', 'Expected ModelResult')
        self._check_state(result.next_state, worlds)
        if self._radio:
            self._check_radio(result.output, worlds, reset)
        else:
            self._check_sensing(result.output, worlds, reset)
        self._check_mapping(result.diagnostics, self._diagnostic_specs, worlds, 'diagnostics')

    def _check_radio(self, output, worlds, reset):
        if not isinstance(output, RadioEvaluationResult):
            self._fail('output', 'Expected RadioEvaluationResult')
        for name, fields in RADIO_FIELDS.items():
            block = getattr(output, name)
            if name not in self._descriptor.output_spec.output_blocks:
                if block is not None:
                    self._fail('output.' + name, 'Unsupported block must be None')
                continue
            if not isinstance(block, RADIO_BLOCK_TYPES[name]):
                self._fail('output.' + name, 'Missing or incorrect SDK block')
            self._tensor(block.valid, self._radio_shape, 'bool', 'output.' + name + '.valid')
            valid = block.valid[worlds]
            if (valid & ~self._bound_slots[worlds]).any():
                self._fail('output.' + name + '.valid', 'Unbound slot marked valid')
            for field in fields:
                if field == 'valid':
                    continue
                value = getattr(block, field)
                self._tensor(value, self._radio_shape, 'float32', 'output.' + name + '.' + field)
                numeric = value[worlds][valid]
                if not torch.isfinite(numeric).all() or (numeric < 0).any():
                    self._fail('output.' + name + '.' + field, 'Valid values must be finite and nonnegative')
            if reset and name == 'rate' and (valid.any() or (block.rate_bps[worlds] != 0).any()):
                self._fail('output.rate', 'Reset rate must be zero and invalid')

    def _check_sensing(self, output, worlds, reset):
        if not isinstance(output, SensingResult):
            self._fail('output', 'Expected SensingResult')
        if (not isinstance(output.field_valid, Mapping) or output.field_valid.keys() != self._sensing_specs.keys()
                or not isinstance(output.time_semantics, Mapping) or output.time_semantics.keys() != self._sensing_specs.keys()):
            self._fail('output', 'Sensing metadata fields differ from declaration')
        for name in ('measurement', 'fim'):
            value = getattr(output, name)
            if name not in self._sensing_specs:
                if value is not None:
                    self._fail('output.' + name, 'Undeclared field must be None')
                continue
            shape, dtype = self._sensing_specs[name]
            self._tensor(value, shape, dtype, 'output.' + name)
            valid = output.field_valid[name]
            self._tensor(valid, shape[:-2] if name == 'fim' else shape, 'bool', 'output.field_valid.' + name)
            semantics = getattr(self._descriptor.output_spec, name).time_semantics
            if output.time_semantics[name] != semantics:
                self._fail('output.time_semantics.' + name, 'Time semantics changed')
            if not torch.isfinite(value[worlds][valid[worlds]]).all():
                self._fail('output.' + name, 'nonfinite valid measurement')
            if reset and semantics in ('increment', 'interval_sample') and valid[worlds].any():
                self._fail('output.' + name, f'Reset cannot publish a valid measurement {semantics}')

    def _leaves(self, result):
        # 只遍历已校验的有限DTO，顺序来自固定schema，不依赖模型返回dict的插入顺序。
        values = [result.next_state[name] for name in self._state_specs]
        if self._radio:
            for name in RADIO_FIELDS:
                block = getattr(result.output, name)
                if block is not None:
                    values.extend(block)
        else:
            for name in self._sensing_specs:
                values.extend((getattr(result.output, name), result.output.field_valid[name]))
        values.extend(result.diagnostics[name] for name in self._diagnostic_specs)
        return values

    def _current(self):
        return ModelResult(self.output, self.state, self.diagnostics)

    def _commit(self, result, worlds, inputs):
        # 调用方可借用缓存的bool切片；提交期间选中集合必须保持不变。
        worlds = worlds.clone()
        destinations = [*self._leaves(self._current()), self.sim_time, self.interval_s]
        # 先取全部所选切片，允许合法next_state交叉借用旧state而不被先写字段污染。
        sources = [value[worlds] for value in (*self._leaves(result), inputs.sim_time, inputs.interval_s)]
        for destination, source in zip(destinations, sources):
            destination[worlds] = source
        return self._current()

    @torch.no_grad()
    def evaluate(self, inputs):
        worlds = torch.ones(self._batch, dtype=torch.bool, device=self._device)
        inputs = self._inputs(inputs, 'post_physics', worlds)
        result = self._model.evaluate(inputs, self.state, self._rng)
        self._check_result(result, worlds, reset=False)
        return self._commit(result, worlds, inputs)

    @torch.no_grad()
    def reset(self, reset_mask: ModelResetMask, inputs):
        self._tensor(reset_mask.world_reset, (self._batch,), 'bool', 'reset_mask')
        worlds = reset_mask.world_reset
        if not worlds.any():
            return self._current()
        inputs = self._inputs(inputs, 'reset', worlds)
        result = self._model.reset_state(self.state, reset_mask, inputs, self._rng)
        self._check_result(result, worlds, reset=True)
        return self._commit(result, worlds, inputs)
