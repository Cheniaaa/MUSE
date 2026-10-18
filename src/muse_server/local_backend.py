"""本机后端的前台入口及按Linux进程身份核对的status/stop。"""
import argparse
import json
import os
from pathlib import Path
import select
import signal
import sys

from muse_sdk.config import ConfigError

from .supervisor import process_identity


def read_status(data_dir: Path) -> dict:
    try:
        instance = json.loads((Path(data_dir) / 'server-instance.json').read_text())
    except FileNotFoundError:
        return {'state': 'stopped'}
    identity = process_identity(instance['pid'])
    state = 'running' if identity is not None and identity == instance['process_start'] else 'stale'
    return {**instance, 'state': state}


def stop_backend(data_dir: Path, instance_id: str) -> dict:
    path = Path(data_dir) / 'server-instance.json'
    try:
        instance = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError('instance_id', 'No backend instance is recorded') from exc
    if instance['instance_id'] != instance_id:
        raise ConfigError('instance_id', 'Backend instance does not match the requested instance')
    try:
        pidfd = os.pidfd_open(instance['pid'])
    except ProcessLookupError as exc:
        raise ConfigError('process_start', 'Recorded backend process has exited') from exc
    try:
        # 先固定进程对象，再核对boot/start；PID复用不能把信号发给新进程。
        identity = process_identity(instance['pid'])
        if identity is None or identity != instance['process_start']:
            raise ConfigError('process_start', 'Recorded backend process identity does not match')
        try:
            signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        except ProcessLookupError as exc:
            raise ConfigError('process_start', 'Backend exited before the stop signal') from exc
        exited, _, _ = select.select([pidfd], [], [], instance['shutdown_timeout'])
        if not exited:
            return {**instance, 'state': 'unconfirmed', 'reason': 'Backend shutdown timed out'}
        # 仅进程退出不足以证明Worker/Viser清理完成；身份文件应由Server正常关闭删除。
        if path.exists():
            return {**instance, 'state': 'unconfirmed', 'reason': 'Backend instance metadata was not cleared'}
        return {**instance, 'state': 'stopped'}
    finally:
        os.close(pidfd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    # start参数和帮助由既有app入口唯一维护，前台调用，不另建Server owner。
    commands.add_parser('start', add_help=False)
    status = commands.add_parser('status')
    status.add_argument('--data-dir', type=Path, required=True)
    stop = commands.add_parser('stop')
    stop.add_argument('--data-dir', type=Path, required=True)
    stop.add_argument('--instance-id', required=True)
    args, remaining = parser.parse_known_args(argv)
    if args.command == 'start':
        from .app import main as start_backend
        start_backend(remaining)
        return 0
    if remaining:
        parser.error('unrecognized arguments: ' + ' '.join(remaining))
    try:
        result = read_status(args.data_dir) if args.command == 'status' else stop_backend(args.data_dir, args.instance_id)
    except ConfigError as exc:
        print(json.dumps({'errors': exc.errors}), file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 1 if result['state'] == 'unconfirmed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
