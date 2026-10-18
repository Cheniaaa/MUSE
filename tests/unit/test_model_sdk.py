"""模型登记与声明边界，不构造执行模型。"""
import json
import subprocess
import sys

import pytest
from jsonschema.exceptions import SchemaError

from muse_sdk.config import DataModel
from muse_sdk.descriptor import RadioDescriptor, SensingDescriptor, build_descriptor, parse_descriptor_json
from muse_sdk.registration import TrainingPackageDefinition


class Config(DataModel):
    pass


def tensor(shape, dtype='float32', units='m', frame='none', rate='radio_epoch'):
    return dict(shape=shape, dtype=dtype, units=units, frame=frame, rate=rate)


def metadata(kind='radio'):
    return dict(model_id='sample', config_schema={}, input_fields=[], state_scope='world',
        state_schema={}, assumptions={}, output_spec=(
            {'output_blocks': ['link']} if kind == 'radio' else
            {'measurement': {**tensor(['B', 'M']), 'time_semantics': 'snapshot'}}))


def declared(kind, meta, *, construct_raises=True):
    from muse_sdk.models import RadioModel, SensingModel
    base = RadioModel if kind == 'radio' else SensingModel
    class Declared(base):
        def __init__(self, *args):
            if construct_raises:
                raise AssertionError('registration must not construct models')
            super().__init__(*args)

        @classmethod
        def describe(cls):
            return {**meta, 'config_schema': Config}

        def evaluate(self, inputs, state, rng):
            raise AssertionError('registration must not evaluate models')

        def reset_state(self, state, reset_mask, inputs, rng):
            raise AssertionError('registration must not reset models')
    return Declared


def test_sdk_import_does_not_load_execution_stack():
    result = subprocess.run([sys.executable, '-c', '''
import sys
from muse_sdk.models import RadioModel, SensingModel, RadioEvaluationResult
blocked = {'torch', 'warp', 'mujoco', 'muse_runtime', 'fastapi'}
assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules)
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('kind', ['radio', 'sensing'])
def test_registration_preserves_model_contract_without_execution(kind):
    impl = declared(kind, metadata(kind))
    definition = TrainingPackageDefinition({}, {}, {},
        {'sample': impl} if kind == 'radio' else {}, {'sample': impl} if kind == 'sensing' else {})
    report = build_descriptor(definition, {'default': {'model_bindings': {
        kind: {'default': 'sample', 'options': {'sample': {}}}}}}, 'a'*64, default_composition_id='default')
    restored = parse_descriptor_json(report.model_dump_json())
    assert getattr(restored, kind + '_models')['sample'].diagnostics_schema == {}
    binding = restored.compositions['default'].model_bindings[kind]
    assert binding.default == 'sample' and set(binding.options) == {'sample'}


@pytest.mark.parametrize('schema', [None, {
    'type': 'number', 'exclusiveMinimum': 5e8, 'exclusiveMaximum': 1e11,
}])
def test_radio_frequency_schema_roundtrip(schema):
    impl = declared('radio', {**metadata(), 'frequency_hz_schema': schema})
    report = build_descriptor(TrainingPackageDefinition({}, {}, {}, {'sample': impl}, {}),
        {'default': {'model_bindings': {'radio': {'default': 'sample', 'options': {'sample': {}}}}}},
        'a'*64, default_composition_id='default')
    assert report.radio_models['sample'].frequency_hz_schema == schema
    restored = parse_descriptor_json(report.model_dump_json())
    assert restored.radio_models['sample'].frequency_hz_schema == schema
    assert RadioDescriptor.model_validate(metadata()).frequency_hz_schema is None


@pytest.mark.parametrize('schema,message', [
    ({'type': 'string'}, 'number'),
    ({'type': 'object'}, 'number'),
    ({'type': 'integer'}, 'number'),
    ({'minimum': 5e8}, 'number'),
    ({'type': ['number', 'null']}, 'number'),
    ({'type': 'number', 'exclusiveMinimum': 'invalid'}, 'exclusiveMinimum'),
    ({'type': 'number', 'if': {'minimum': 5e8}}, 'Unsupported schema keyword: if'),
    ({'type': 'number', 'execute': 'callback'}, 'Unsupported schema keyword: execute'),
])
@pytest.mark.parametrize('entry', ['dto', 'export', 'json'])
def test_radio_frequency_schema_rejects_invalid_declarations(schema, message, entry):
    meta = {**metadata(), 'frequency_hz_schema': schema}
    if entry == 'json':
        report = build_descriptor(TrainingPackageDefinition({}, {}, {},
            {'sample': declared('radio', metadata())}, {}), {}, 'a'*64, default_composition_id=None)
        payload = report.model_dump()
        payload['radio_models']['sample']['frequency_hz_schema'] = schema
        with pytest.raises((ValueError, SchemaError), match=message):
            parse_descriptor_json(json.dumps(payload))
    elif entry == 'export':
        with pytest.raises((ValueError, SchemaError), match=message):
            build_descriptor(TrainingPackageDefinition({}, {}, {},
                {'sample': declared('radio', meta)}, {}), {}, 'a'*64, default_composition_id=None)
    else:
        with pytest.raises((ValueError, SchemaError), match=message):
            RadioDescriptor.model_validate(meta)


@pytest.mark.parametrize('mode', ['wrong_kind', 'plain', 'abstract'])
def test_model_registration_requires_concrete_kind(mode):
    from muse_sdk.models import RadioModel
    impl = declared('sensing', metadata())
    if mode == 'plain':
        class Plain:
            describe = impl.describe
        impl = Plain
    elif mode == 'abstract':
        class Abstract(RadioModel):
            describe = impl.describe
        impl = Abstract
    with pytest.raises(ValueError, match='radio'):
        build_descriptor(TrainingPackageDefinition({}, {}, {}, {'sample': impl}, {}), {}, 'a'*64, default_composition_id=None)


@pytest.mark.parametrize('path', [
    ['task', 'value'], ['observations', 'value'], ['network', 'radio', 'rate', 'rate_bps'],
    ['network', 'geometry', {'binding': 'group_id'}], ['network', 'unknown', 'x'],
])
def test_radio_model_rejects_non_model_inputs(path):
    meta = metadata()
    meta['input_fields'] = [dict(path=path, phase='post_physics', tensor=tensor(['B', 'L', 'F']))]
    with pytest.raises(ValueError):
        RadioDescriptor.model_validate(meta)


def test_input_phase_identity_and_conflicting_specs():
    meta = metadata()
    field = dict(path=['network', 'geometry', 'distance'], phase='reset', tensor=tensor(['B', 'L']))
    meta['input_fields'] = [field, {**field, 'phase': 'post_physics'}]
    assert len(RadioDescriptor.model_validate(meta).input_fields) == 2
    for bad in (field, {**field, 'phase': 'action'},
                {**field, 'phase': 'post_physics', 'tensor': tensor(['B', 'F'])}):
        with pytest.raises(ValueError):
            RadioDescriptor.model_validate({**meta, 'input_fields': [field, bad]})


@pytest.mark.parametrize('section', ['state_schema', 'diagnostics_schema'])
def test_model_state_and_diagnostics_require_world_axis(section):
    meta = metadata()
    meta[section] = {'memory': tensor(['L', 'F'])}
    with pytest.raises(ValueError):
        RadioDescriptor.model_validate(meta)


@pytest.mark.parametrize('field,shape,dtype', [
    ('fim', ['B', 'M', 2, 3], 'float32'), ('fim', ['B', 0, 0], 'float32'),
    ('measurement', ['M'], 'float32'), ('measurement', ['B', 'M'], 'int64'),
])
def test_sensing_declared_numeric_shape(field, shape, dtype):
    meta = metadata('sensing')
    meta['output_spec'] = {field: {**tensor(shape, dtype), 'time_semantics': 'snapshot'}}
    with pytest.raises(ValueError):
        SensingDescriptor.model_validate(meta)


@pytest.mark.parametrize('path,spec', [
    (['physics_outputs', 'sim_time'], tensor(['B'], 'float32', 's', rate='physics')),
    (['network', 'interval_s'], tensor(['B'], 'float64', 'ms', rate='physics')),
    (['network', 'radio', 'rate', 'rate_bps'], tensor(['B', 'L', 'F'], units='W')),
    (['network', 'radio', 'link', 'unknown'], tensor(['B', 'L', 'F'])),
])
def test_fixed_input_contract_cannot_relabel_units_or_dtype(path, spec):
    meta = metadata('sensing')
    meta['input_fields'] = [dict(path=path, phase='reset', tensor=spec)]
    with pytest.raises(ValueError):
        SensingDescriptor.model_validate(meta)


@pytest.mark.parametrize('blocks', [[], ['link', 'link'], ['capacity']])
def test_invalid_radio_capabilities_rejected(blocks):
    with pytest.raises(ValueError):
        RadioDescriptor.model_validate({**metadata(), 'output_spec': {'output_blocks': blocks}})


def test_interval_sample_descriptor_roundtrip_without_execution():
    meta = metadata('sensing')
    meta['output_spec']['measurement']['time_semantics'] = 'interval_sample'
    report = build_descriptor(TrainingPackageDefinition({}, {}, {}, {},
        {'sample': declared('sensing', meta)}), {'default': {'model_bindings': {
            'sensing': {'default': 'sample', 'options': {'sample': {}}}}}}, 'a'*64, default_composition_id='default')
    restored = parse_descriptor_json(report.model_dump_json())
    assert restored.sensing_models['sample'].output_spec.measurement.time_semantics == 'interval_sample'


def test_fim_cannot_declare_interval_sample():
    meta = metadata('sensing')
    meta['output_spec'] = {'fim': {**tensor(['B', 2, 2]), 'time_semantics': 'interval_sample'}}
    with pytest.raises(ValueError, match='FIM.*interval_sample'):
        SensingDescriptor.model_validate(meta)
