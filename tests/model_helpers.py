"""模型契约合成算例；数值只验证数据流，不是无线科学模型。"""
from copy import deepcopy
from types import MappingProxyType, SimpleNamespace

import torch

from muse_sdk.descriptor import InputField, RadioDescriptor, SensingDescriptor
from muse_sdk.models import (ExecutionContext, InterferenceMetrics, LinkMetrics, ModelResult,
    RadioEvaluationResult, RadioInputs, RadioModel, RateMetrics, SensingInputs, SensingModel, SensingResult)
from muse_runtime.rng import WorldRng


def field(path, shape, dtype='float32', units='m', phase='reset'):
    return dict(path=path, phase=phase,
        tensor=dict(shape=shape, dtype=dtype, units=units, frame='none', rate='radio_epoch'))


def common_case(implementation, descriptor, inputs, device, bound_slots=None):
    axes = {'B': 2, 'L': 3, 'F': 5, 'M': 4, 'K': 2}
    layout = MappingProxyType({'axes': MappingProxyType(axes), 'instance_id': 'model_a'})
    available = tuple(item.model_copy(update={'tensor': item.tensor.model_copy(update={
        'shape': [axes[x] if isinstance(x, str) else x for x in item.tensor.shape]})})
        for item in descriptor.input_fields)
    return SimpleNamespace(implementation=implementation,
        kwargs=dict(model_type=implementation, descriptor=descriptor, config=MappingProxyType({}),
            layout=layout, context=ExecutionContext(device), available_inputs=available,
            rng=WorldRng(7, 2, device), bound_slots=bound_slots),
        reset_inputs=inputs,
        step_inputs=inputs._replace(sim_time=torch.tensor([.2, .4], dtype=torch.float64, device=device),
            interval_s=torch.full((2,), .1, dtype=torch.float64, device=device)))


def make_radio_case(blocks=('rate',), device='cpu'):
    requirements = [field(['network', 'geometry', 'distance'], ['B', 'L', 'F']),
                    field(['network', 'endpoint_masks', 'bound'], ['B', 'L', 'F'], 'bool', '1')]
    requirements += [{**item, 'phase': 'post_physics'} for item in requirements]
    meta = dict(model_id='synthetic_radio', config_schema={}, input_fields=requirements,
        state_scope='world', state_schema={}, diagnostics_schema={}, assumptions={'synthetic': True},
        output_spec={'output_blocks': list(blocks)})
    descriptor = RadioDescriptor.model_validate(meta)

    class SyntheticRadio(RadioModel):
        @classmethod
        def describe(cls):
            from muse_sdk.config import DataModel
            return {**deepcopy(meta), 'config_schema': DataModel}

        def result(self, inputs, *, reset):
            distance = inputs.geometry['distance']
            valid = inputs.endpoint_masks['bound'].clone()
            return RadioEvaluationResult(
                LinkMetrics(torch.ones_like(distance), distance.clone(), valid.clone()) if 'link' in blocks else None,
                InterferenceMetrics(torch.full_like(distance, 2), torch.full_like(distance, 3), valid.clone())
                    if 'interference' in blocks else None,
                RateMetrics(torch.zeros_like(distance) if reset else distance * 10,
                    torch.zeros_like(valid) if reset else valid.clone()) if 'rate' in blocks else None)

        def reset_state(self, state, reset_mask, inputs, rng):
            return ModelResult(self.result(inputs, reset=True), (), {})

        def evaluate(self, inputs, state, rng):
            return ModelResult(self.result(inputs, reset=False), (), {})

    bound = torch.ones(2, 3, 5, dtype=torch.bool, device=device)
    bound[1, 2, 4] = False
    inputs = RadioInputs({'distance': torch.arange(1, 31, dtype=torch.float32, device=device).reshape(2, 3, 5)},
        {'bound': bound}, {}, {}, {}, torch.zeros(2, dtype=torch.float64, device=device),
        torch.zeros(2, dtype=torch.float64, device=device))
    return common_case(SyntheticRadio, descriptor, inputs, device, bound)


def make_sensing_case(fields=('measurement',), device='cpu', semantics=None):
    semantics = semantics or {'measurement': 'snapshot', 'fim': 'increment'}
    specs = {'measurement': dict(shape=['B', 'M', 'K'], dtype='float32', units='m', frame='none', rate='radio_epoch'),
             'fim': dict(shape=['B', 'M', 'K', 'K'], dtype='float64', units='1/m^2', frame='none', rate='radio_epoch')}
    requirements = [dict(path=['network', 'geometry', name], phase=phase, tensor=specs[name])
                    for name in fields for phase in ('reset', 'post_physics')]
    meta = dict(model_id='synthetic_sensing', config_schema={}, input_fields=requirements,
        state_scope='world', state_schema={}, diagnostics_schema={}, assumptions={'synthetic': True},
        output_spec={name: {**specs[name], 'time_semantics': semantics[name]} for name in fields})
    descriptor = SensingDescriptor.model_validate(meta)

    class SyntheticSensing(SensingModel):
        @classmethod
        def describe(cls):
            from muse_sdk.config import DataModel
            return {**deepcopy(meta), 'config_schema': DataModel}

        def result(self, inputs, reset):
            values = {name: inputs.geometry[name].clone() for name in fields}
            valid = {name: torch.full(values[name].shape[:-2] if name == 'fim' else values[name].shape,
                not (reset and semantics[name] in ('increment', 'interval_sample')), dtype=torch.bool, device=self.context.device)
                for name in fields}
            return ModelResult(SensingResult(values.get('measurement'), values.get('fim'), valid,
                {name: semantics[name] for name in fields}), (), {})

        def reset_state(self, state, reset_mask, inputs, rng):
            return self.result(inputs, True)

        def evaluate(self, inputs, state, rng):
            return self.result(inputs, False)

    values = {'measurement': torch.full((2, 4, 2), -1., device=device),
              'fim': torch.eye(2, dtype=torch.float64, device=device).expand(2, 4, 2, 2).clone()}
    inputs = SensingInputs({k: v for k, v in values.items() if k in fields}, {}, {}, {}, {}, None,
        torch.zeros(2, dtype=torch.float64, device=device), torch.zeros(2, dtype=torch.float64, device=device))
    return common_case(SyntheticSensing, descriptor, inputs, device)


def create_instance(case):
    from muse_runtime.model_execution import ModelInstance
    return ModelInstance(**case.kwargs)


def make_stateful_case(device='cpu', instance_id='radio_a', seed=7, randomized=False):
    case = make_radio_case(('link', 'rate'), device)
    descriptor = case.kwargs['descriptor'].model_copy(update={
        'state_schema': {'memory': InputField.model_validate(field(['network', 'geometry', 'memory'],
            ['B', 'L', 'F'])).tensor},
        'diagnostics_schema': {'calls': InputField.model_validate(field(['network', 'geometry', 'calls'],
            ['B'], 'int64', '1')).tensor}})
    class Stateful(case.implementation):
        def initial_state(self, batch_shape):
            assert batch_shape == (2,)
            return {'memory': torch.zeros(2, 3, 5, device=self.context.device)}

        def reset_state(self, state, reset_mask, inputs, rng):
            memory = state['memory'].clone()
            ids = reset_mask.world_reset.nonzero().flatten()
            memory[ids] = (rng.uniform(ids, self.layout['instance_id'], 'memory', (3, 5))
                           if randomized else 1.)
            return ModelResult(self.result(inputs, reset=True), {'memory': memory},
                {'calls': torch.zeros(2, dtype=torch.int64, device=self.context.device)})

        def evaluate(self, inputs, state, rng):
            memory = state['memory'] + (rng.uniform(range(2), self.layout['instance_id'], 'memory', (3, 5))
                                       if randomized else 2.)
            # calls为解析memory oracle派生的诊断，不作为模型隐藏state。
            calls = ((memory[:, 0, 0] - 1) / 2).to(torch.int64)
            return ModelResult(self.result(inputs, reset=False), {'memory': memory}, {'calls': calls})
    case.implementation = Stateful
    case.kwargs.update(model_type=Stateful, descriptor=descriptor,
        layout=MappingProxyType({**case.kwargs['layout'], 'instance_id': instance_id}),
        rng=WorldRng(seed, 2, device))
    return case
