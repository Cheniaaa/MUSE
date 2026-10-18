"""真实包登记不构造模型；仅显式消费需求激活无线运行。"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import MappingProxyType

import pytest
import yaml

from flight_helpers import EXAMPLE, archive
from muse_sdk.descriptor import parse_descriptor_json


RADIO_COMBINATIONS = (
    ('link',), ('interference',), ('rate',), ('link', 'interference'),
    ('link', 'rate'), ('interference', 'rate'), ('link', 'interference', 'rate'),
)
SENSING_COMBINATIONS = (('measurement',), ('fim',), ('measurement', 'fim'))

# 合成模型只验证契约；Torch仅在真正执行的方法内导入。
MODEL_SOURCE = '''
from muse_sdk.config import DataModel
from muse_sdk.models import RadioModel, SensingModel, ModelResult, RadioEvaluationResult, RateMetrics


class Config(DataModel):
    pass


class SentinelRadio(RadioModel):
    def __init__(self, config, layout, context):
        raise AssertionError('registration constructed a radio model')

    @classmethod
    def describe(cls):
        return dict(model_id=cls.model_id, config_schema=Config, input_fields=[],
            state_scope='world', state_schema={}, assumptions={'synthetic': True},
            diagnostics_schema={'calls': dict(shape=['B'], dtype='int64', units='1',
                                             frame='none', rate='radio_epoch')},
            output_spec={'output_blocks': list(cls.blocks)})

    def evaluate(self, inputs, state, rng):
        raise AssertionError('registration evaluated a radio model')

    def reset_state(self, state, reset_mask, inputs, rng):
        raise AssertionError('registration reset a radio model')


class SentinelSensing(SensingModel):
    def __init__(self, config, layout, context):
        raise AssertionError('registration constructed a sensing model')

    @classmethod
    def describe(cls):
        specs = {
            'measurement': dict(shape=['B', 'M', 'K'], dtype='float32', units='m',
                frame='none', rate='radio_epoch', time_semantics='snapshot'),
            'fim': dict(shape=['B', 'M', 'K', 'K'], dtype='float64', units='1/m^2',
                frame='none', rate='radio_epoch', time_semantics='increment'),
        }
        return dict(model_id=cls.model_id, config_schema=Config, input_fields=[],
            state_scope='world', state_schema={}, diagnostics_schema={},
            assumptions={'synthetic': True}, output_spec={name: specs[name] for name in cls.fields})

    def evaluate(self, inputs, state, rng):
        raise AssertionError('registration evaluated a sensing model')

    def reset_state(self, state, reset_mask, inputs, rng):
        raise AssertionError('registration reset a sensing model')


class CpuRate(SentinelRadio):
    model_id = 'cpu_rate'
    blocks = ('rate',)

    def __init__(self, config, layout, context):
        RadioModel.__init__(self, config, layout, context)

    def reset_state(self, state, reset_mask, inputs, rng):
        import torch
        axes = self.layout['axes']
        shape = tuple(axes[name] for name in ('B', 'L', 'F'))
        output = RadioEvaluationResult(None, None, RateMetrics(
            torch.zeros(shape, dtype=torch.float32, device=self.context.device),
            torch.zeros(shape, dtype=torch.bool, device=self.context.device)))
        return ModelResult(output, state, {'calls': torch.zeros(
            axes['B'], dtype=torch.int64, device=self.context.device)})

    def evaluate(self, inputs, state, rng):
        import torch
        axes = self.layout['axes']
        shape = tuple(axes[name] for name in ('B', 'L', 'F'))
        output = RadioEvaluationResult(None, None, RateMetrics(
            torch.zeros(shape, dtype=torch.float32, device=self.context.device),
            torch.ones(shape, dtype=torch.bool, device=self.context.device)))
        return ModelResult(output, state, {'calls': torch.ones(
            axes['B'], dtype=torch.int64, device=self.context.device)})


radio_models = {'cpu_rate': CpuRate}
sensing_models = {}
'''


@pytest.fixture
def model_package(fixture_root, tmp_path):
    package = shutil.copytree(fixture_root, tmp_path / 'package',
                              ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    declarations = []
    for blocks in RADIO_COMBINATIONS:
        name = 'radio_' + '_'.join(blocks)
        declarations.append(
            f"radio_models[{name!r}] = type({name!r}, (SentinelRadio,), "
            f"{{'model_id': {name!r}, 'blocks': {blocks!r}}})\n")
    for fields in SENSING_COMBINATIONS:
        name = 'sensing_' + '_'.join(fields)
        declarations.append(
            f"sensing_models[{name!r}] = type({name!r}, (SentinelSensing,), "
            f"{{'model_id': {name!r}, 'fields': {fields!r}}})\n")
    (package / 'models.py').write_text(MODEL_SOURCE + ''.join(declarations), encoding='utf-8')
    with (package / '__init__.py').open('a', encoding='utf-8') as stream:
        stream.write('''
from .models import radio_models, sensing_models
_build_base = build_training_package

def build_training_package():
    definition = _build_base()
    definition.radio_models.update(radio_models)
    definition.sensing_models.update(sensing_models)
    return definition
''')
    return package


def test_one_shot_describes_models_without_constructing_them(model_package, pipeline, tmp_path):
    catalog, _ = pipeline
    record = catalog.register_archive('training_package', archive(model_package, tmp_path / 'models.zip'))
    validation = Path(record.path).parent / 'validation'
    report = json.loads((validation / 'validation-report.json').read_text())
    assert report['valid'] and report['errors'] == []
    assert report['resolved_experiment'] is None
    descriptor = parse_descriptor_json((validation / 'descriptor.json').read_text(), catalog.builtins.descriptor)
    restored = parse_descriptor_json(descriptor.model_dump_json(), catalog.builtins.descriptor)
    assert restored.model_dump(mode='json') == record.metadata
    assert restored.package_hash == record.sha256 == report['package_hash']
    assert set(restored.radio_models) == {'cpu_rate'} | {
        'radio_' + '_'.join(blocks) for blocks in RADIO_COMBINATIONS}
    for blocks in RADIO_COMBINATIONS:
        declared = restored.radio_models['radio_' + '_'.join(blocks)]
        assert declared.output_spec.output_blocks == list(blocks)
        assert declared.diagnostics_schema['calls'].shape == ['B']
        assert declared.diagnostics_schema['calls'].dtype == 'int64'
        assert declared.assumptions == {'synthetic': True}
    assert set(restored.sensing_models) == {
        'sensing_' + '_'.join(fields) for fields in SENSING_COMBINATIONS}
    for fields in SENSING_COMBINATIONS:
        declared = restored.sensing_models['sensing_' + '_'.join(fields)]
        measurement, fim = declared.output_spec.measurement, declared.output_spec.fim
        if 'measurement' in fields:
            assert measurement.shape == ['B', 'M', 'K']
            assert (measurement.dtype, measurement.units, measurement.time_semantics) == ('float32', 'm', 'snapshot')
        else:
            assert measurement is None
        if 'fim' in fields:
            assert fim.shape == ['B', 'M', 'K', 'K']
            assert (fim.dtype, fim.units, fim.time_semantics) == ('float64', '1/m^2', 'increment')
        else:
            assert fim is None
    persisted = next(r for r in catalog.snapshot().resources if r.resource_id == record.resource_id)
    assert persisted.metadata == record.metadata


@pytest.mark.parametrize('kind', ['radio', 'sensing'])
@pytest.mark.parametrize('mode', ['wrong_kind', 'plain', 'abstract'])
def test_one_shot_rejects_wrong_or_abstract_model(model_package, tmp_path, kind, mode):
    base = 'SentinelRadio' if kind == 'radio' else 'SentinelSensing'
    if mode == 'wrong_kind':
        opposite = 'SentinelSensing' if kind == 'radio' else 'SentinelRadio'
        broken = f'class Broken({opposite}):\n    pass\n'
    elif mode == 'plain':
        broken = f'class Broken:\n    describe = {base}.describe\n'
    else:
        abc = 'RadioModel' if kind == 'radio' else 'SensingModel'
        broken = f'class Broken({base}):\n    evaluate = {abc}.evaluate\n'
    with (model_package / 'models.py').open('a', encoding='utf-8') as stream:
        stream.write('\n' + broken + f"{kind}_models['broken'] = Broken\n")
    output = tmp_path / 'validation'
    result = subprocess.run([
        sys.executable, '-m', 'muse_runtime.validation', '--package', str(model_package),
        '--output-dir', str(output),
    ], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    terminal, = [json.loads(line) for line in result.stdout.splitlines()]
    report = json.loads((output / 'validation-report.json').read_text())
    assert terminal['valid'] is report['valid'] is False
    assert report['resolved_experiment'] is None
    assert report['errors'] == [{
        'field': kind + '.broken',
        'message': 'Model must be a concrete ' + ('RadioModel' if kind == 'radio' else 'SensingModel'),
    }]
    assert not (output / 'descriptor.json').exists()
    assert 'Traceback' in (output / 'user.log').read_text()


def test_registered_model_candidate_does_not_activate_wireless(model_package, pipeline, tmp_path):
    from muse_server.resolver import resolve_experiment

    catalog, source = pipeline
    # 仅替换临时包的组合，复用已验证的基础飞行Scenario与算法。
    composition = yaml.safe_load((EXAMPLE / 'training_package/training_package.yaml').read_text())
    composition_path = model_package / 'training_package.yaml'
    composition_path.write_text(yaml.safe_dump(composition), encoding='utf-8')
    no_binding = catalog.register_archive('training_package', archive(model_package, tmp_path / 'flight.zip'))
    request = {**source, 'training_package_id': no_binding.resource_id,
               'training_package_revision': no_binding.revision}
    control = resolve_experiment(request, catalog.snapshot())
    assert control.valid, control.errors
    assert control.resolved_experiment.model_bindings == {}
    assert 'cpu_rate' in no_binding.metadata['radio_models']

    binding = {'default': 'cpu_rate', 'options': {'cpu_rate': {}}}
    composition['compositions']['default']['model_bindings'] = {'radio': binding}
    composition_path.write_text(yaml.safe_dump(composition), encoding='utf-8')
    bound = catalog.register_archive('training_package', archive(model_package, tmp_path / 'bound-flight.zip'))
    assert bound.metadata['compositions']['default']['model_bindings'] == {'radio': {
        'default': 'cpu_rate', 'options': {'cpu_rate': {'config': {}}}}}
    request.update(training_package_id=bound.resource_id, training_package_revision=bound.revision)
    resolved = resolve_experiment(request, catalog.snapshot())
    assert resolved.valid, resolved.errors
    assert resolved.resolved_experiment.network is None
    assert resolved.resolved_experiment.model_bindings == {}
    assert 'radio' not in resolved.resolved_experiment.activation


def test_registered_package_model_executes_via_model_instance(model_package, pipeline, tmp_path):
    import torch
    from muse_runtime.model_execution import ModelInstance
    from muse_runtime.package_loader import load_package
    from muse_runtime.rng import WorldRng
    from muse_sdk.models import ExecutionContext, ModelResetMask, RadioInputs

    catalog, _ = pipeline
    record = catalog.register_archive('training_package', archive(model_package, tmp_path / 'runtime.zip'))
    descriptor = parse_descriptor_json(json.dumps(record.metadata), catalog.builtins.descriptor)
    empty = MappingProxyType({})
    reset_inputs = RadioInputs(empty, empty, empty, empty, empty,
                                torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64))
    step_inputs = reset_inputs._replace(sim_time=torch.tensor([.1, .2], dtype=torch.float64),
                                        interval_s=torch.full((2,), .1, dtype=torch.float64))
    with load_package(Path(record.path)) as loaded:
        instance = ModelInstance(loaded.definition.radio_models['cpu_rate'],
            descriptor=descriptor.radio_models['cpu_rate'], config=empty,
            layout=MappingProxyType({'axes': MappingProxyType({'B': 2, 'L': 3, 'F': 5}),
                                     'instance_id': 'package_radio'}),
            context=ExecutionContext('cpu'), available_inputs=(),
            rng=WorldRng(7, 2, 'cpu'), bound_slots=torch.ones((2, 3, 5), dtype=torch.bool))
        assert instance.state == ()
        assert instance.output.link is instance.output.interference is None
        assert instance.output.rate.rate_bps.shape == (2, 3, 5)
        assert not instance.output.rate.valid.any()
        assert not instance.output.rate.rate_bps.any()
        assert not instance.diagnostics['calls'].any()
        instance.reset(ModelResetMask(torch.ones(2, dtype=torch.bool)), reset_inputs)
        result = instance.evaluate(step_inputs)
        assert result.output.rate.valid.all()
        assert not result.output.rate.rate_bps.any()  # 有效零值不同于invalid和None。
        torch.testing.assert_close(result.diagnostics['calls'], torch.ones(2, dtype=torch.int64))
        torch.testing.assert_close(instance.sim_time, step_inputs.sim_time)
        reset = instance.reset(ModelResetMask(torch.tensor([True, False])), reset_inputs)
        assert not reset.output.rate.valid[0].any()
        assert reset.output.rate.valid[1].all()
        torch.testing.assert_close(reset.diagnostics['calls'], torch.tensor([0, 1]))
        torch.testing.assert_close(instance.sim_time, torch.tensor([0., .2], dtype=torch.float64))
