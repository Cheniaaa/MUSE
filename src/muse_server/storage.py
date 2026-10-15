"""上传入口的archive检查与不可变文件发布。"""
import json
import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile
import xml.etree.ElementTree as ET

import yaml

from muse_sdk.config import ConfigError, check_json


def verify_checkpoint_file(artifact, run_id: str, directory: Path) -> Path:
    """发布/引用边界共用：核验最终 checkpoint 身份、归属和完整性。"""
    prefix = f'{run_id}:checkpoint:'
    token = artifact.id.removeprefix(prefix)
    if (artifact.run != run_id or not artifact.id.startswith(prefix)
            or re.fullmatch(r'[0-9a-f]{32}', token) is None):
        raise ValueError('Invalid checkpoint run/ID identity')
    if artifact.path != f'checkpoints/{token}.pt':
        raise ValueError('Invalid checkpoint path; expected final relative checkpoints file')
    try:
        root = Path(directory).resolve()
        path = (root / artifact.path).resolve()
        if not path.is_relative_to(root / 'checkpoints'):
            raise ValueError('Checkpoint path outside allocated checkpoints directory')
        if not path.is_file() or path.stat().st_size != artifact.size:
            raise ValueError('Checkpoint file missing or size mismatch')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != artifact.sha256:
            raise ValueError('Checkpoint sha256 mismatch')
    except OSError as exc:
        raise ValueError(f'Checkpoint file validation failed: {exc}') from exc
    return path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(str(key), 'Duplicate JSON key')
        result[key] = value
    return result


class UniqueLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = '<<' if key_node.tag == 'tag:yaml.org,2002:merge' else self.construct_object(key_node, deep=deep)
            if not isinstance(key, (str, int, float, bool, type(None))) or key in keys:
                raise ConfigError('archive', 'Duplicate or non-scalar YAML key')
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def read_document(path: Path):
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() == '.json':
        value = json.loads(text, object_pairs_hook=unique_object)
    else:
        value = yaml.load(text, Loader=UniqueLoader)
    check_json(value, str(path))
    return value


def extract_archive(archive: Path, destination: Path) -> None:
    """只允许普通文件/目录，逐成员写入，不调用extractall。"""
    destination.mkdir()
    seen = set()

    def target(name, is_directory):
        path = PurePosixPath(name)
        if (not name or '\\' in name or ':' in name or path.is_absolute()
                or any(part in ('..', '.') for part in name.rstrip('/').split('/'))
                or not path.parts):
            raise ConfigError('archive', f'Unsafe path: {name}')
        normalized = path.as_posix()
        if normalized in seen:
            raise ConfigError('archive', f'Duplicate member: {name}')
        seen.add(normalized)
        output = destination.joinpath(*path.parts)
        if is_directory:
            output.mkdir(parents=True, exist_ok=True)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
        return output

    try:
        if not zipfile.is_zipfile(archive):
            raise ConfigError('archive', 'Expected ZIP archive')
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                mode = item.external_attr >> 16
                kind = stat.S_IFMT(mode)
                if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ConfigError('archive', 'Links and special files are not allowed')
                output = target(item.filename, item.is_dir())
                if not item.is_dir():
                    with source.open(item) as src, output.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
        for path in destination.rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.json', '.yaml', '.yml'):
                read_document(path)
    except (OSError, zipfile.BadZipFile, yaml.YAMLError) as exc:
        raise ConfigError('archive', str(exc)) from exc


def write_json_new(path: Path, content: str) -> None:
    import os
    with path.open('x', encoding='utf-8') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def validate_bundled_assets(root: Path) -> None:
    """只检查资源路径归属；物理语法/feature编译留给GPU准备阶段。"""
    for path in root.rglob('*'):
        if path.suffix.lower() not in ('.xml', '.mjcf'):
            continue
        try:
            document = ET.parse(path)
        except ET.ParseError as exc:
            raise ConfigError('environment.scene', str(exc)) from exc
        for element in document.iter():
            for name in ('file', 'meshdir', 'texturedir', 'assetdir'):
                value = element.get(name)
                if value is None:
                    continue
                reference = PurePosixPath(value)
                if (reference.is_absolute() or '\\' in value or ':' in value
                        or not (path.parent / value).resolve().is_relative_to(root)):
                    raise ConfigError('environment.scene', 'All assets must use bundled relative paths')
