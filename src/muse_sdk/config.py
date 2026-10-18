"""配置导出和层合并；schema值校验由jsonschema负责。"""
from copy import deepcopy
from enum import Enum
import math
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from annotated_types import Ge, Gt, Le, Lt, MaxLen, MinLen, MultipleOf
from jsonschema import Draft202012Validator, validators
from pydantic import BaseModel, ConfigDict
from pydantic.fields import FieldInfo


class DataModel(BaseModel):
    """JSON边界共用的严格数据模型。"""
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class ConfigError(ValueError):
    def __init__(self, field: str, message: str):
        self.errors = [{'field': field, 'message': message}]
        super().__init__(f'{field}: {message}')


def check_json(value: Any, field: str = '$') -> None:
    """在反序列化/导出边界拒绝非JSON值和非有限值。"""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for i, item in enumerate(value):
            check_json(item, f'{field}[{i}]')
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for key, item in value.items():
            check_json(item, f'{field}.{key}')
        return
    raise ConfigError(field, 'Expected finite JSON data')


def _check_annotations(value: Any, path: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or any(k not in ('title', 'description', 'examples', 'unit', 'units', 'frame', 'rate') and not k.startswith('x-') for k in value):
        raise ConfigError(path, 'Only declarative metadata annotations are supported')
    check_json(value, path)


def _check_field(field: FieldInfo, path: str) -> None:
    if field.default_factory is not None or any(getattr(field, name) is not None for name in ('alias', 'validation_alias', 'serialization_alias')):
        raise ConfigError(path, 'Dynamic defaults and aliases cannot be exported')
    _check_annotations(field.json_schema_extra, path)
    for item in field.metadata:
        if isinstance(item, (Gt, Ge, Lt, Le, MultipleOf, MinLen, MaxLen)):
            continue
        # Field(pattern=...)提供声明式元数据；不依赖其内部类名。
        if getattr(item, '__dict__', {}).keys() == {'pattern'} and isinstance(item.pattern, str):
            continue
        raise ConfigError(path, 'Only declarative Field constraints are supported')


def _check_declarations(model: type[BaseModel], seen: set[type]) -> None:
    if model in seen:
        return
    seen.add(model)
    if model.model_config.get('extra') != 'forbid':
        raise ConfigError(model.__name__, 'Config models must declare extra="forbid"')
    for hook in ('__init__', 'model_post_init', '__get_pydantic_core_schema__', '__get_pydantic_json_schema__'):
        method, base = getattr(model, hook), getattr(BaseModel, hook)
        if getattr(method, '__func__', method) is not getattr(base, '__func__', base):
            raise ConfigError(model.__name__, f'Custom {hook} cannot be exported')
    # Pydantic BaseModel记录的decorator metadata；不遍历Rust/core schema或内部生成器。
    decorators = model.__pydantic_decorators__
    if any(getattr(decorators, key) for key in ('validators', 'root_validators', 'field_validators', 'model_validators', 'field_serializers', 'model_serializers', 'computed_fields')):
        raise ConfigError(model.__name__, 'Python validator/serializer functions cannot be exported')
    if any(model.model_config.get(k) for k in ('str_to_lower', 'str_strip_whitespace', 'str_to_upper', 'coerce_numbers_to_str')):
        raise ConfigError(model.__name__, 'String transformations cannot be exported')
    _check_annotations(model.model_config.get('json_schema_extra'), model.__name__)
    for name, field in model.model_fields.items():
        path = f'{model.__name__}.{name}'
        _check_field(field, path)
        pending = [field.annotation]
        while pending:
            annotation = pending.pop()
            origin, args = get_origin(annotation), get_args(annotation)
            if origin is Annotated:
                _check_field(FieldInfo.from_annotation(annotation), path)
                pending.append(args[0])
            elif origin in (list, dict, Union, UnionType):
                pending.extend(args)
            elif origin is Literal or annotation in (str, int, float, bool, type(None)):
                continue
            elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                _check_declarations(annotation, seen)
            elif isinstance(annotation, type) and issubclass(annotation, Enum):
                check_json([member.value for member in annotation], path)
            else:
                raise ConfigError(path, 'Only declared JSON field types are supported')


def _deref(node: dict, root: dict, chain: tuple[str, ...] = ()) -> dict:
    if '$ref' not in node:
        return node
    ref = node['$ref']
    if not ref.startswith('#/$defs/') or ref in chain:
        raise ConfigError('$schema', 'Only acyclic local $defs references are supported')
    target: Any = root
    try:
        for part in ref[2:].split('/'):
            target = target[part.replace('~1', '/').replace('~0', '~')]
    except (KeyError, TypeError) as exc:
        raise ConfigError('$schema', f'Unresolved reference {ref}') from exc
    return {**_deref(target, root, (*chain, ref)), **{k: v for k, v in node.items() if k != '$ref'}}


def _schema_nodes(node: dict):
    for group in ('properties', '$defs', 'patternProperties'):
        yield from node.get(group, {}).values()
    for key in ('items', 'additionalProperties'):
        if isinstance(node.get(key), dict):
            yield node[key]
    yield from node.get('anyOf', [])


def validate_config_schema(schema: dict) -> None:
    check_json(schema, '$schema')
    Draft202012Validator.check_schema(schema)

    def visit(node, refs=()):
        # 该子集不含条件/组合执行图；拒绝未知schema关键字而非让它绕过本地ref检查。
        allowed = {'$schema', '$defs', '$ref', 'type', 'title', 'description', 'examples',
                   'default', 'properties', 'required', 'additionalProperties', 'items',
                   'anyOf', 'enum', 'const', 'minimum', 'maximum', 'exclusiveMinimum',
                   'exclusiveMaximum', 'multipleOf', 'minLength', 'maxLength', 'pattern',
                   'minItems', 'maxItems', 'minProperties', 'maxProperties',
                   'unit', 'units', 'frame', 'rate'}
        for key in node:
            if key not in allowed and not key.startswith('x-'):
                raise ConfigError('$schema', f'Unsupported schema keyword: {key}')
        ref = node.get('$ref')
        if ref:
            if ref in refs:
                raise ConfigError('$schema', 'Recursive config models are not supported')
            visit(_deref(node, schema), (*refs, ref))
            return
        if 'anyOf' in node:
            branches = [_deref(b, schema) for b in node['anyOf']]
            nonnull = [b for b in branches if b.get('type') != 'null']
            if len(nonnull) > 1 and any(b.get('type') not in ('string', 'number', 'integer', 'boolean', 'null') or 'default' in b for b in nonnull):
                raise ConfigError('$schema', 'Ambiguous union defaults/object branches are unsupported')
        for child in _schema_nodes(node):
            visit(child, refs)
    visit(schema)


def export_config_schema(model_type: type[BaseModel]) -> dict:
    _check_declarations(model_type, set())
    schema = model_type.model_json_schema(mode='validation')
    validate_config_schema(schema)
    return schema


def _value_origins(value, source, path=''):
    if isinstance(value, dict) and value:
        return {pointer: origin for key, item in value.items()
                for pointer, origin in _value_origins(item, source,
                    path + '/' + key.replace('~', '~0').replace('/', '~1')).items()}
    return {path: source} if path else {}


def _merge(lower: dict, upper: dict, origins: dict, source: str, path='') -> dict:
    result = deepcopy(lower)
    for key, value in upper.items():
        pointer = path + '/' + key.replace('~', '~0').replace('/', '~1')
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value, origins, source, pointer)
        else:
            result[key] = deepcopy(value)
            for previous in list(origins):
                if previous == pointer or previous.startswith(pointer + '/'):
                    del origins[previous]
            origins.update(_value_origins(value, source, pointer))
    return result


def _defaults(node: dict, value: Any, root: dict) -> Any:
    node = _deref(node, root)
    if 'anyOf' in node and value is not None:
        branches = [b for b in node['anyOf'] if _deref(b, root).get('type') != 'null']
        if len(branches) == 1:
            return _defaults(branches[0], value, root)
    if isinstance(value, dict):
        result = deepcopy(value)
        properties = node.get('properties', {})
        for key, prop in properties.items():
            resolved = _deref(prop, root)
            if key not in result and 'default' in resolved:
                result[key] = deepcopy(resolved['default'])
            if key in result:
                result[key] = _defaults(prop, result[key], root)
        extra = node.get('additionalProperties')
        if isinstance(extra, dict):
            for key in result.keys() - properties.keys():
                result[key] = _defaults(extra, result[key], root)
        return result
    if isinstance(value, list) and isinstance(node.get('items'), dict):
        return [_defaults(node['items'], item, root) for item in value]
    return deepcopy(value)


# 只取消required关键字；其他schema语义全部由同一个成熟validator执行。
_PartialValidator = validators.extend(Draft202012Validator, {'required': lambda v, r, i, s: iter(())})


def resolve_config(schema: dict, yaml_values: dict, overrides: dict, *, require_complete: bool) -> dict:
    """消费导出/反序列化边界已验证的schema；这里只合并并校验参数值。"""
    values, _ = resolve_config_layers(schema, [('package', yaml_values), ('override', overrides)],
                                     require_complete=require_complete)
    return values


def resolve_config_layers(schema: dict, layers: list[tuple[str, dict]], *,
                          require_complete: bool) -> tuple[dict, dict[str, str]]:
    """同次解析返回有效值和来源；数组为整体覆盖，来源不按索引继承。"""
    for name, value in layers:
        check_json(value, name)
        if not isinstance(value, dict):
            raise ConfigError(name, 'Expected object')
    # 默认先形成低优先级层，避免显式null后又复活下层对象。
    merged = _defaults(schema, {}, schema)
    origins = _value_origins(merged, 'schema')
    for source, values in layers:
        merged = _merge(merged, values, origins, source)
    merged = _defaults(schema, merged, schema)
    # 新插入对象中的默认字段也由同一_defaults填入，不替换显式null。
    origins = {path: origins.get(path, 'schema') for path in _value_origins(merged, 'schema')}
    validator = Draft202012Validator if require_complete else _PartialValidator
    errors = list(validator(schema).iter_errors(merged))
    if errors:
        exc = ConfigError('$', 'Invalid configuration')
        exc.errors = [{'field': '.'.join(map(str, e.absolute_path)) or '$', 'message': e.message} for e in errors]
        exc.effective, exc.origins = merged, origins
        raise exc
    return merged, origins
