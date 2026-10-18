"""可信、不可变资源目录的内容摘要。"""
import hashlib
import os
from pathlib import Path
import stat

from muse_sdk.config import ConfigError


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
