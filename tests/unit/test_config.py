"""配置边界：错误合并、隐藏Python行为和宽松输入都会破坏实验复现。"""
from copy import deepcopy
from typing import Annotated

import pytest
from pydantic import (AfterValidator, BaseModel, ConfigDict, Field,
                      WithJsonSchema, computed_field, field_serializer, field_validator, model_validator)

from muse_sdk.config import ConfigError, export_config_schema, resolve_config, validate_config_schema


class Gains(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    kp: float = Field(default=1.0, gt=0)
    ki: float = 0.0


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    gains: Gains = Gains()
    channels: list[int] = [0, 1]
    label: str | None = 'baseline'
    budget: float = Field(gt=0)


def test_layers_preserve_null_and_replace_list():
    schema = export_config_schema(Settings)
    lower = {'gains': {'kp': 2.0}, 'budget': 4.0}
    upper = {'gains': {'ki': 0.5}, 'channels': [2], 'label': None}
    original = deepcopy((schema, lower, upper))
    result = resolve_config(schema, lower, upper, require_complete=True)
    assert result == {'gains': {'kp': 2.0, 'ki': 0.5}, 'channels': [2],
                      'label': None, 'budget': 4.0}
    assert (schema, lower, upper) == original
    result['channels'].append(9)
    assert upper['channels'] == [2]


@pytest.mark.parametrize('value', [True, -1.0, float('nan'), float('inf'), '4'])
def test_invalid_budget(value):
    with pytest.raises(ConfigError):
        resolve_config(export_config_schema(Settings), {}, {'budget': value}, require_complete=True)


def test_partial_config_checks_present_values():
    schema = export_config_schema(Settings)
    assert 'budget' not in resolve_config(schema, {}, {}, require_complete=False)
    for values in ({'budget': 'bad'}, {'gains': {'kp': -1}}, {'unknown': 1}):
        with pytest.raises(ConfigError):
            resolve_config(schema, values, {}, require_complete=False)
    with pytest.raises(ConfigError):
        resolve_config(schema, {}, {}, require_complete=True)


@pytest.mark.parametrize('override', [{'gains': {'missing': 1}}, {'gains': None},
                                     {'channels': [True]}, {'channels': [1.2]}])
def test_invalid_nested_values(override):
    with pytest.raises(ConfigError):
        resolve_config(export_config_schema(Settings), {'budget': 3.0}, override, require_complete=True)


def test_empty_list_and_json_integral_number():
    schema = export_config_schema(Settings)
    assert resolve_config(schema, {'budget': 1}, {'channels': []}, require_complete=True)['channels'] == []
    assert resolve_config(schema, {'budget': 1}, {'channels': [1.0]}, require_complete=True)['channels'] == [1.0]


def test_nested_required_and_nullable_objects():
    class Nested(BaseModel):
        model_config = ConfigDict(extra='forbid')
        required: float
        gains: Gains = Gains()

    class Container(BaseModel):
        model_config = ConfigDict(extra='forbid')
        nested: Nested | None
        items: list[Gains] = []
        table: dict[str, Gains] = {}

    schema = export_config_schema(Container)
    assert 'nested' not in resolve_config(schema, {}, {}, require_complete=False)
    value = resolve_config(schema, {'nested': {}, 'items': [{}], 'table': {'a': {}}}, {}, require_complete=False)
    assert value['nested'] == {'gains': {'kp': 1.0, 'ki': 0.0}}
    assert value['items'] == [{'kp': 1.0, 'ki': 0.0}]
    assert value['table'] == {'a': {'kp': 1.0, 'ki': 0.0}}
    with pytest.raises(ConfigError):
        resolve_config(schema, {'nested': {}}, {}, require_complete=True)
    with pytest.raises(ConfigError):
        resolve_config(schema, {'nested': {'required': 'bad'}}, {}, require_complete=False)
    assert resolve_config(schema, {'nested': None}, {}, require_complete=True)['nested'] is None


def test_external_schema_ref_rejected_without_fetch():
    with pytest.raises(ConfigError):
        validate_config_schema({'type': 'object', 'properties': {'x': {'$ref': 'https://invalid.test/schema'}}})


def test_python_validator_not_hidden_by_json_schema():
    class Hidden(Settings):
        @field_validator('budget')
        @classmethod
        def reject_seven(cls, value):
            if value == 7:
                raise ValueError('seven forbidden')
            return value

    with pytest.raises(ConfigError, match='function'):
        export_config_schema(Hidden)


def test_hidden_model_validator_rejected():
    class Hidden(Settings):
        @model_validator(mode='after')
        def check(self):
            return self
    with pytest.raises(ConfigError):
        export_config_schema(Hidden)


@pytest.mark.parametrize('field', [Field(default_factory=list), Field(alias='other'),
                                  Field(validation_alias='other'), Field(serialization_alias='other')])
def test_factory_and_alias_rejected(field):
    class Hidden(BaseModel):
        values: list[int] = field
    with pytest.raises(ConfigError):
        export_config_schema(Hidden)


def test_serializer_computed_and_custom_type_rejected():
    class Serialized(Settings):
        @field_serializer('budget')
        def serialize_budget(self, value):
            return str(value)
    class Computed(Settings):
        @computed_field
        @property
        def doubled(self) -> float:
            return self.budget * 2
    class Custom(Settings):
        special: Annotated[int, AfterValidator(lambda v: v + 1)]
    for model in (Serialized, Computed, Custom):
        with pytest.raises(ConfigError):
            export_config_schema(model)


def test_object_union_rejected_but_scalar_union_allowed():
    class Other(BaseModel):
        model_config = ConfigDict(extra='forbid')
        name: str = 'other'
    class Ambiguous(BaseModel):
        value: Gains | Other
    with pytest.raises(ConfigError):
        export_config_schema(Ambiguous)
    class Scalar(BaseModel):
        model_config = ConfigDict(extra='forbid')
        value: int | str
    assert resolve_config(export_config_schema(Scalar), {'value': 'yes'}, {}, require_complete=True) == {'value': 'yes'}


def test_nested_hidden_behavior_rejected():
    class Hidden(BaseModel):
        value: list[int] = Field(default_factory=list)
    class Parent(BaseModel):
        nested: Hidden
    with pytest.raises(ConfigError):
        export_config_schema(Parent)


@pytest.mark.parametrize('schema', [
    {'allOf': [{'$ref': 'file:///nonexistent-muse-schema'}]},
    {'oneOf': [{'$ref': 'https://invalid.test/schema'}]},
    {'if': {'type': 'object'}, 'then': {'$ref': 'file:///nonexistent-muse-schema'}},
    {'properties': {'x': {'allOf': [{'$ref': 'https://invalid.test/schema'}]}}},
])
def test_external_ref_in_unhandled_keyword_rejected_before_validator(schema, monkeypatch):
    import urllib.request
    def reject_fetch(*args, **kwargs):
        raise AssertionError('External reference reached the I/O layer')
    monkeypatch.setattr(urllib.request, 'urlopen', reject_fetch)
    with pytest.raises(ConfigError):
        validate_config_schema(schema)


def test_type_field_and_nested_type_field_are_ordinary_names():
    class TypeConfig(BaseModel):
        model_config = ConfigDict(extra='forbid')
        type: str
    class Parent(BaseModel):
        model_config = ConfigDict(extra='forbid')
        nested: TypeConfig
    assert resolve_config(export_config_schema(TypeConfig), {'type': 'velocity'}, {}, require_complete=True) == {'type': 'velocity'}
    assert resolve_config(export_config_schema(Parent), {'nested': {'type': 'velocity'}}, {}, require_complete=True) == {'nested': {'type': 'velocity'}}


def test_schema_hooks_post_init_and_custom_init_cannot_hide_behavior():
    class Mismatch(BaseModel):
        model_config = ConfigDict(extra='forbid')
        value: Annotated[int, WithJsonSchema({'type': 'string'})]
    class PostInit(BaseModel):
        model_config = ConfigDict(extra='forbid')
        value: int
        def model_post_init(self, context):
            raise ValueError('hidden')
    class CustomInit(BaseModel):
        model_config = ConfigDict(extra='forbid')
        value: int
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    class SchemaCallback(BaseModel):
        model_config = ConfigDict(extra='forbid', json_schema_extra=lambda s: s.update(type='string'))
        value: int
    class AlteredField(BaseModel):
        model_config = ConfigDict(extra='forbid')
        value: int = Field(json_schema_extra={'type': 'string'})
    for model in (Mismatch, PostInit, CustomInit, SchemaCallback, AlteredField):
        with pytest.raises(ConfigError):
            export_config_schema(model)


def test_declarative_enum_pattern_and_nested_constraints():
    from enum import Enum
    class Mode(str, Enum):
        fixed = 'fixed'
        moving = 'moving'
    class Declared(BaseModel):
        model_config = ConfigDict(extra='forbid')
        mode: Mode = Mode.fixed
        name: str = Field(pattern=r'^[a-z]+$')
        values: list[Annotated[float, Field(gt=0)]]
    schema = export_config_schema(Declared)
    result = resolve_config(schema, {'name': 'scout', 'values': [1.0]}, {}, require_complete=True)
    assert result == {'mode': 'fixed', 'name': 'scout', 'values': [1.0]}
    for values in ({'mode': 'unknown'}, {'name': 'BAD'}, {'values': [-1.0]}):
        with pytest.raises(ConfigError):
            resolve_config(schema, result, values, require_complete=True)
