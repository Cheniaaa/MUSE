"""按字节 cursor 读取 Run 的持久数据，不持有 Supervisor 锁或跨请求缓存。"""

import base64
import json
import os
from pathlib import Path
from typing import Literal


def read_metric_page(directory: Path, *, offset: int, limit: int, terminal: bool) -> dict:
    """保留完整事件；最后半行只观察，不提交其 cursor。"""
    records = []
    next_offset = offset
    with (directory / 'metrics.jsonl').open('rb') as stream:
        stream.seek(offset)
        for _ in range(limit):
            line = stream.readline()
            if not line.endswith(b'\n'):
                break
            records.append(json.loads(line))
            next_offset = stream.tell()
        # eof 表示已观察到当前物理尾部；不代表 Run 终态或半行已消费。
        eof = stream.tell() >= os.fstat(stream.fileno()).st_size
    return dict(records=records, next_offset=next_offset, eof=eof,
                terminal=terminal, availability='available')


def read_log_page(directory: Path, *, channel: Literal['worker', 'viser'],
                  offset: int, limit: int) -> dict:
    """仅暴露固定日志 channel，UTF-8 跨块解码由消费端负责。"""
    relative = {'worker': 'logs/worker.log', 'viser': 'logs/viser.log'}[channel]
    with (directory / relative).open('rb') as stream:
        stream.seek(offset)
        data = stream.read(limit)
        next_offset = stream.tell()
        eof = next_offset >= os.fstat(stream.fileno()).st_size
    return dict(data_base64=base64.b64encode(data).decode('ascii'),
                next_offset=next_offset, eof=eof, availability='available')
