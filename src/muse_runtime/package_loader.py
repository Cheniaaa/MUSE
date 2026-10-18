"""可信Python包加载；隔离模块命名，不提供代码sandbox。"""
from dataclasses import dataclass, field
import hashlib
import importlib.util
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import stat
import sys
from uuid import uuid4

import yaml

from muse_sdk.config import ConfigError
from muse_sdk.registration import TrainingPackageDefinition


def _content_paths(root: Path, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    paths = []
    for entry in root.iterdir():
        relative = (*prefix, entry.name)
        mode = entry.lstat().st_mode
        if stat.S_ISDIR(mode):
            paths.extend(_content_paths(entry, relative))
        elif stat.S_ISREG(mode):
            if '__pycache__' not in relative and not entry.name.endswith('.pyc'):
                paths.append(relative)
        else:
            raise ConfigError('/'.join(relative), 'Symlinks and nonregular files are not allowed')
    return paths


def hash_package(package_dir: Path) -> str:
    """按POSIX路径排序；每个路径/文件内容前放8字节大端长度，再做SHA256。

    输入是调用方准备好的可信、不可变目录；不处理并发替换文件的对抗场景。
    拒绝包根/包内链接和特殊文件，但不限制工作区祖先路径。
    """
    root = Path(os.path.abspath(package_dir))
    digest = hashlib.sha256()
    try:
        if not stat.S_ISDIR(root.lstat().st_mode):
            raise ConfigError('package', 'Expected a directory, not a symlink or file')
        for parts in sorted(_content_paths(root), key=lambda p: '/'.join(p)):
            path = root.joinpath(*parts)
            name = '/'.join(parts).encode('utf-8')
            digest.update(len(name).to_bytes(8, 'big'))
            digest.update(name)
            digest.update(path.stat().st_size.to_bytes(8, 'big'))
            with path.open('rb') as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    except OSError as exc:
        raise ConfigError('package', str(exc)) from exc
    return digest.hexdigest()


class _UniqueSafeLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        # 在SafeLoader合并mapping前检查原始键，避免last-write-wins。
        keys = set()
        for key_node, _ in node.value:
            if key_node.tag == 'tag:yaml.org,2002:merge':
                key = '<<'
            else:
                key = self.construct_object(key_node, deep=deep)
            try:
                if key in keys:
                    raise ConfigError('training_package.yaml', f'Duplicate YAML key: {key!r}')
                keys.add(key)
            except TypeError as exc:
                raise ConfigError('training_package.yaml', 'YAML mapping keys must be scalar') from exc
        return super().construct_mapping(node, deep=deep)


def _cleanup(namespace: str) -> None:
    for name in tuple(sys.modules):
        if name == namespace or name.startswith(namespace + '.'):
            del sys.modules[name]


class _SourceLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # 内容hash排除pyc，因此包内模块必须只执行本次源码字节。
        return self.source_to_code(self.get_data(self.path), self.path)


class _PackageFinder(importlib.abc.MetaPathFinder):
    def __init__(self, namespace: str):
        self.namespace = namespace

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.namespace + '.'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            if not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                raise ConfigError(fullname, 'Package modules must be Python source files')
            spec.loader = _SourceLoader(fullname, spec.origin)
        return spec


@dataclass
class LoadedPackage:
    definition: TrainingPackageDefinition
    compositions: dict
    default_composition_id: str
    package_hash: str
    root: Path
    _namespace: str = field(repr=False)
    _finder: _PackageFinder = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __enter__(self) -> 'LoadedPackage':
        if self._closed:
            raise RuntimeError('LoadedPackage is already closed')
        return self

    def close(self) -> None:
        if self._closed:
            return
        _cleanup(self._namespace)
        sys.meta_path.remove(self._finder)
        self._closed = True

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def load_package(package_dir: Path) -> LoadedPackage:
    """加载并注册；调用方必须通过context manager或close结束模块生命周期。"""
    root = Path(os.path.abspath(package_dir))
    package_hash = hash_package(root)
    for name in ('__init__.py', 'training_package.yaml'):
        if not (root / name).is_file():
            raise ConfigError(name, 'Required package file is missing')
    try:
        with (root / 'training_package.yaml').open(encoding='utf-8') as stream:
            declaration = yaml.load(stream, Loader=_UniqueSafeLoader)
    except (yaml.YAMLError, UnicodeError) as exc:
        raise ConfigError('training_package.yaml', str(exc)) from exc
    if not isinstance(declaration, dict):
        raise ConfigError('training_package.yaml', 'Expected a mapping')
    if set(declaration) != {'compositions', 'default_composition_id'}:
        raise ConfigError('training_package.yaml', 'Expected compositions and default_composition_id')

    namespace = '_muse_package_' + uuid4().hex
    finder = _PackageFinder(namespace)
    sys.meta_path.insert(0, finder)
    try:
        spec = importlib.util.spec_from_file_location(
            namespace, root / '__init__.py', submodule_search_locations=[str(root)],
            loader=_SourceLoader(namespace, str(root / '__init__.py')),
        )
        if spec is None or spec.loader is None:
            raise ConfigError('__init__.py', 'Cannot create package import spec')
        module = importlib.util.module_from_spec(spec)
        sys.modules[namespace] = module
        spec.loader.exec_module(module)
        builder = getattr(module, 'build_training_package', None)
        if not callable(builder):
            raise ConfigError('build_training_package', 'Required registration function is missing')
        definition = builder()
        if not isinstance(definition, TrainingPackageDefinition):
            raise ConfigError('build_training_package', 'Expected TrainingPackageDefinition')
        return LoadedPackage(definition, declaration['compositions'], declaration['default_composition_id'],
                             package_hash, root, namespace, finder)
    except BaseException:
        _cleanup(namespace)
        sys.meta_path.remove(finder)
        raise
