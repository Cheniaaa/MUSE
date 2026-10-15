"""一次性CPU包验证CLI；成功仅代表descriptor校验，不是完整Experiment。"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

from muse_sdk.config import ConfigError, DataModel
from muse_sdk.descriptor import build_descriptor, parse_descriptor_json
from muse_sdk.experiment import BuiltinMetadata

from .package_loader import load_package


class ValidationReport(DataModel):
    valid: bool
    package_hash: str | None
    resolved_experiment: None = None
    errors: list[dict[str, str]]
    warnings: list[dict[str, str]]


def _atomic_write(path: Path, content: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """仅供one-shot进程调用；标准流重定向保持到解释器退出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--builtin-metadata', type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir.absolute()
    try:
        output.mkdir()
    except OSError as exc:
        print(f'Cannot create new output directory: {exc}', file=sys.stderr)
        return 1

    protocol_fd = os.dup(1)
    package_hash = None
    descriptor_path = output / 'descriptor.json'
    try:
        # 先保存协议fd，再重定向Python及底层native写入；不恢复到终端，
        # 让解释器退出时的缓冲flush/atexit输出也进入user.log。
        sys.stdout.flush()
        sys.stderr.flush()
        with (output / 'user.log').open('xb') as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
        try:
            builtins = None
            if args.builtin_metadata:
                metadata = BuiltinMetadata.model_validate_json(args.builtin_metadata.read_text())
                builtins = parse_descriptor_json(metadata.descriptor.model_dump_json())
            with load_package(args.package) as loaded:
                package_hash = loaded.package_hash
                descriptor = build_descriptor(loaded.definition, loaded.composition, package_hash, builtins)
                descriptor_json = descriptor.model_dump_json(indent=2)
            report = ValidationReport(valid=True, package_hash=package_hash, errors=[], warnings=[])
            _atomic_write(descriptor_path, descriptor_json)
            _atomic_write(output / 'validation-report.json', report.model_dump_json(indent=2))
        except BaseException as exc:
            # 用户的SystemExit也不能伪装成验证成功；保留完整traceback。
            traceback.print_exc()
            descriptor_path.unlink(missing_ok=True)
            errors = exc.errors if isinstance(exc, ConfigError) else [
                {'field': 'package', 'message': f'{type(exc).__name__}: {exc}'},
            ]
            report = ValidationReport(valid=False, package_hash=package_hash, errors=errors, warnings=[])
            _atomic_write(output / 'validation-report.json', report.model_dump_json(indent=2))
        sys.stdout.flush()
        sys.stderr.flush()
        # 协议只发布一条terminal结果，用户打印不参与此通道。
        terminal = {'valid': report.valid, 'package_hash': report.package_hash}
        with os.fdopen(os.dup(protocol_fd), 'w', encoding='utf-8') as protocol:
            protocol.write(json.dumps(terminal, allow_nan=False) + '\n')
        return 0 if report.valid else 1
    except BaseException:
        # 产物/协议I/O失败也必须明确非零并留下详情，不能静默返回。
        traceback.print_exc()
        return 1
    finally:
        os.close(protocol_fd)


if __name__ == '__main__':
    raise SystemExit(main())
