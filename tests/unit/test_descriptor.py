from copy import deepcopy
from dataclasses import replace
import json

import pytest

from muse_sdk.config import ConfigError
from muse_sdk.descriptor import PackageDescriptor, build_descriptor, validate_composition


def test_descriptor_data_roundtrip_and_candidates(descriptor):
    data = descriptor.model_dump(mode='json')
    encoded = json.dumps(data, allow_nan=False)
    assert PackageDescriptor.model_validate_json(encoded) == descriptor
    validate_composition(PackageDescriptor.model_validate_json(encoded))
    assert 'implementation' not in data['terms']['progress']
    choices = data['compositions']['default']['agent_roles']['scout']['reward_terms']
    assert [s['selection_id'] for s in choices] == ['progress', 'progress_aux']
    assert choices[0]['required'] is True
    assert choices[1]['default_enabled'] is False
    assert data['tasks'] == {}
    assert set(data['algorithms']) == {'cpu_rule', 'alternate'}


def test_descriptor_retains_multiple_named_compositions(definition, composition):
    alternate = deepcopy(composition)
    alternate['algorithm'] = 'alternate'
    result = build_descriptor(definition, {'default': composition, 'alternate': alternate}, 'a' * 64,
                              default_composition_id='default')
    assert result.sdk_version == '0.2'
    assert result.default_composition_id == 'default'
    assert result.compositions['default'].algorithm == 'cpu_rule'
    assert result.compositions['alternate'].algorithm == 'alternate'


def test_descriptor_rejects_missing_default_and_bad_option(definition, composition):
    with pytest.raises((ConfigError, ValueError)):
        build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='missing')
    composition['model_bindings'] = {'radio': {'default': 'not_an_option', 'options': {}}}
    with pytest.raises((ConfigError, ValueError)):
        build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='default')


@pytest.mark.parametrize('change', [
    {'term': 'unknown'}, {'term': 'builtin:progress'}, {'required': True, 'default_enabled': False},
    {'config': {'role_id': 'guard'}}, {'scope': 'world'}, {'weight': float('inf')},
])
def test_bad_selection_rejected(definition, composition, change):
    composition['agent_roles']['scout']['reward_terms'][0].update(change)
    with pytest.raises((ConfigError, ValueError)):
        build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='default')


def test_duplicate_selection_and_wrong_kind(definition, composition):
    rows = composition['agent_roles']['scout']['reward_terms']
    rows.append(deepcopy(rows[0]))
    with pytest.raises(ConfigError):
        build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='default')
    rows.pop()
    rows[0]['term'] = 'collision'
    with pytest.raises(ConfigError):
        build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='default')


def test_nonreward_weight_and_bad_scope(definition, composition):
    composition['world_termination_terms'][0]['weight'] = 2.0
    with pytest.raises(ConfigError):
        build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='default')
    composition['world_termination_terms'][0].pop('weight')
    term = definition.terms['progress']
    definition.terms['progress'] = replace(term, scope='world', output_spec={**term.output_spec, 'shape': ['B']})
    with pytest.raises(ConfigError):
        build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='default')


@pytest.mark.parametrize('field,value', [('algorithm', 'missing'), ('task', 'missing')])
def test_unregistered_composition_reference(definition, composition, field, value):
    composition[field] = value
    with pytest.raises(ConfigError):
        build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='default')


def test_describe_id_must_match_registration(definition, composition):
    definition.algorithms['wrong'] = definition.algorithms['cpu_rule']
    with pytest.raises(ConfigError):
        build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='default')


def test_field_path_must_be_specific_and_phase_known(definition, composition):
    term = definition.terms['progress']
    for path, phase in [(['network'], 'post_physics'), (['task', 'value'], 'future')]:
        definition.terms['progress'] = replace(term, input_fields=({'path': path, 'phase': phase, 'tensor': term.output_spec},))
        with pytest.raises(ValueError):
            build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='default')


def test_schema_and_supplied_disabled_values_checked(definition, composition):
    composition['agent_roles']['scout']['reward_terms'][1]['config'] = {'scale': 'bad'}
    with pytest.raises(ConfigError):
        build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='default')


def test_unknown_composition_field_is_not_silently_ignored(definition, composition):
    composition['agorithm'] = 'cpu_rule'
    with pytest.raises(ValueError):
        build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='default')


def test_unconsumed_model_config_values_is_not_an_accepted_option(definition, composition):
    composition['model_config_values'] = {'radio': {'ignored_parameter': 1}}
    with pytest.raises(ValueError):
        build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='default')


def test_descriptor_ingestion_validates_schema_before_use(descriptor):
    from muse_sdk.descriptor import parse_descriptor_json
    data = descriptor.model_dump(mode='json')
    data['terms']['progress']['config_schema'] = {'allOf': [{'$ref': 'file:///missing-schema'}]}
    with pytest.raises(ConfigError):
        parse_descriptor_json(json.dumps(data))


def test_algorithm_describe_requires_full_static_contract(definition, composition):
    config_model = definition.terms['collision'].config_model
    class Incomplete:
        @classmethod
        def describe(cls):
            return {'algorithm_id': 'cpu_rule', 'config_schema': config_model}
    definition.algorithms['cpu_rule'] = Incomplete
    with pytest.raises(ValueError):
        build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='default')


@pytest.mark.parametrize('path', [['unknown_root', ''], ['task', ''], ['task', 'value()'], ['task', {'binding': 'unknown'}]])
def test_input_path_syntax_is_validated(definition, composition, path):
    term = definition.terms['progress']
    definition.terms['progress'] = replace(term, input_fields=({'path': path, 'phase': 'post_physics', 'tensor': term.output_spec},))
    with pytest.raises(ValueError):
        build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='default')


@pytest.mark.parametrize('kind,identity,fields', [
    ('tasks', 'task_id', {'input_fields': [], 'published_fields': {}, 'role_status_bindings': {}, 'learning_done_roles': []}),
    ('radio_models', 'model_id', {'input_fields': [], 'output_spec': {'output_blocks': ['link']}, 'state_scope': 'world', 'state_schema': {}, 'assumptions': {}}),
    ('sensing_models', 'model_id', {'input_fields': [], 'output_spec': {'measurement': {'shape': ['B', 'M'], 'dtype': 'float32', 'units': 'm', 'frame': 'world', 'rate': 'decision', 'time_semantics': 'snapshot'}}, 'state_scope': 'world', 'state_schema': {}, 'assumptions': {}}),
])
def test_other_describe_kinds_have_required_static_fields(definition, composition, kind, identity, fields):
    from muse_sdk.models import RadioModel, SensingModel
    config = definition.terms['collision'].config_model
    meta = {identity: 'example', 'config_schema': config, **fields}
    base = {'radio_models': RadioModel, 'sensing_models': SensingModel}.get(kind, object)
    class Declared(base):
        @classmethod
        def describe(cls):
            return dict(meta)
    if kind != 'tasks':
        # 该fixture只验证登记，不执行模型；提供具体ABC方法使声明错误真实可达。
        class Concrete(Declared):
            def evaluate(self, inputs, state, rng):
                raise AssertionError('not executed during registration')

            def reset_state(self, state, reset_mask, inputs, rng):
                raise AssertionError('not executed during registration')
        Declared = Concrete
    getattr(definition, kind)['example'] = Declared
    descriptor = build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='default')
    assert 'example' in getattr(descriptor, kind)
    meta.pop('input_fields')
    with pytest.raises(ValueError):
        build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='default')


@pytest.mark.parametrize('field', ['missing', 'motion'])
def test_schedule_binding_rejects_missing_or_noninteger_config_field(field):
    from muse_builtins.rules import ConstantVelocity
    from muse_sdk.registration import TrainingPackageDefinition

    class InvalidSchedule(ConstantVelocity):
        @classmethod
        def describe(cls):
            return {**super().describe(), 'training_schedule': {'collection_steps': field}}

    with pytest.raises(ValueError, match='integer config field'):
        build_descriptor(TrainingPackageDefinition({}, {}, {'constant_velocity': InvalidSchedule}, {}, {}), {}, 'a' * 64, default_composition_id=None)
