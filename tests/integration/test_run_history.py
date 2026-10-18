import base64
import codecs
import json
from pathlib import Path
import threading

import pytest

from muse_server.run_data import read_log_page, read_metric_page

from test_runs_api import register_experiment
from test_supervisor import command, wait_for


@pytest.fixture
def history_client(tmp_path):
    from fastapi.testclient import TestClient
    from muse_server.app import create_app
    with TestClient(create_app(tmp_path / 'catalog', worker_command=command('cooperative'))) as client:
        yield client


def test_run_filters_and_artifacts_match_source(history_client):
    client = history_client
    identity, _ = register_experiment(client)
    active = client.post('/runs', json=dict(experiment_id=identity, mode='EVALUATE', max_decision_steps=8)).json()
    run_id = active['run_id']
    wait_for(lambda: client.get('/runs/' + run_id).json()['state'] == 'RUNNING')
    queued = client.post('/runs', json=dict(experiment_id=identity, mode='INTERACTIVE', max_decision_steps=3)).json()
    page = client.get('/runs', params={'limit': 1}).json()
    assert page['items'][0]['run_id'] == run_id and page['next_offset'] == 1
    assert page['items'][0]['mode'] == 'EVALUATE' and page['items'][0]['max_decision_steps'] == 8
    assert page['items'][0]['checkpoint'] is None
    filtered = client.get('/runs', params={'experiment_id': identity, 'mode': 'INTERACTIVE', 'state': 'QUEUED'}).json()
    assert [r['run_id'] for r in filtered['items']] == [queued['run_id']]
    assert filtered['next_offset'] is None
    assert client.get('/runs', params={'experiment_id': 'missing'}).json()['items'] == []
    assert client.get('/runs', params={'offset': -1}).status_code == 422
    assert client.get('/runs', params={'limit': 201}).status_code == 422
    assert client.get('/runs', params={'mode': 'typo'}).status_code == 422
    for channel in ('worker', 'viser'):
        pending = client.get(f"/runs/{queued['run_id']}/logs", params={'channel': channel}).json()
        assert pending['availability'] == 'not_started' and pending['data_base64'] == ''
    client.post(f"/runs/{queued['run_id']}/cancel")
    client.post('/runs/' + run_id + '/cancel')
    wait_for(lambda: client.get('/runs/' + run_id).json()['state'] == 'CANCELLED')
    logs = client.get('/runs/' + run_id + '/logs', params={'channel': 'worker'}).json()
    artifact = next(a for a in client.get('/runs/' + run_id + '/artifacts').json() if a['kind'] == 'log')
    assert base64.b64decode(logs['data_base64']) == client.get(f"/runs/{run_id}/artifacts/{artifact['id']}").content
    assert client.get('/runs/' + run_id + '/logs', params={'channel': '../server'}).status_code == 422
    assert client.get('/runs/' + run_id + '/metrics', params={'limit': 2001}).status_code == 422
    metrics = client.get('/runs/' + run_id + '/metrics').json()
    assert metrics['terminal'] is True and metrics['availability'] == 'available'
    path = Path(client.app.state.supervisor.get_run(run_id)['artifact_directory']) / 'metrics.jsonl'
    path.unlink()
    with pytest.raises(FileNotFoundError):
        client.get('/runs/' + run_id + '/metrics')


def test_slow_logs_do_not_block_cancel(history_client, monkeypatch):
    import asyncio
    from starlette.websockets import WebSocket
    client = history_client
    identity, _ = register_experiment(client)
    run_id = client.post('/runs', json=dict(experiment_id=identity, mode='EVALUATE', max_decision_steps=8)).json()['run_id']
    wait_for(lambda: client.get('/runs/' + run_id).json()['state'] == 'RUNNING')
    blocked, release = threading.Event(), threading.Event()
    original = WebSocket.send_json

    async def slow_send(socket, data, *args, **kwargs):
        if socket.url.path.endswith('/logs'):
            blocked.set()
            await asyncio.to_thread(release.wait)
        return await original(socket, data, *args, **kwargs)

    monkeypatch.setattr(WebSocket, 'send_json', slow_send)
    try:
        with client.websocket_connect(f'/streams/runs/{run_id}/logs?channel=worker'):
            assert blocked.wait(5)
            cancelled = client.post('/runs/' + run_id + '/cancel')
            assert cancelled.status_code == 200
            wait_for(lambda: client.get('/runs/' + run_id).json()['state'] == 'CANCELLED')
            assert client.app.state.supervisor.error is None
            release.set()
    finally:
        release.set()


def test_log_stream_starts_at_tail_and_reports_gap(history_client, monkeypatch):
    import asyncio
    from starlette.websockets import WebSocket
    client = history_client
    identity, _ = register_experiment(client)
    run_id = client.post('/runs', json=dict(experiment_id=identity, mode='EVALUATE', max_decision_steps=8)).json()['run_id']
    wait_for(lambda: client.get('/runs/' + run_id).json()['state'] == 'RUNNING')
    path = Path(client.app.state.supervisor.get_run(run_id)['artifact_directory']) / 'logs/worker.log'
    before = path.read_bytes()
    blocked, release = threading.Event(), threading.Event()
    original = WebSocket.send_json

    async def block_initial_send(socket, data, *args, **kwargs):
        result = await original(socket, data, *args, **kwargs)
        if socket.url.path.endswith('/logs') and not blocked.is_set():
            blocked.set()
            await asyncio.to_thread(release.wait)
        return result

    monkeypatch.setattr(WebSocket, 'send_json', block_initial_send)
    try:
        with client.websocket_connect(f'/streams/runs/{run_id}/logs?channel=worker') as ws:
            initial = ws.receive_json()
            assert initial['data_base64'] == '' and initial['next_offset'] == len(before)
            assert blocked.wait(5)
            raw = b'x' * 300000 + b'visible-tail'
            with path.open('ab') as writer:
                writer.write(raw)
            release.set()
            page = ws.receive_json()
            start = len(before) + len(raw) - 262144
            assert page['gap'] == {'previous_offset': len(before), 'next_offset': start}
            assert page['next_offset'] == start + 65536
            assert len(base64.b64decode(page['data_base64'])) == 65536
            # 有缺口仍可从HTTP按原cursor取回历史字节，不能用latest值补造。
            history = client.get(f'/runs/{run_id}/logs', params={'offset': len(before), 'limit': 262144}).json()
            assert base64.b64decode(history['data_base64']) == raw[:262144]
    finally:
        release.set()


def test_started_viewer_missing_log_is_not_reported_as_not_started(tmp_path):
    from muse_server.app import _run_log_page
    from muse_server.supervisor import Supervisor
    from test_supervisor import make_plan
    from test_display_session import DISPLAY_CHILD, ready
    import sys
    catalog, plan = make_plan(tmp_path)
    worker = lambda p: [sys.executable, '-c', DISPLAY_CHILD,
                        str(Path(p.artifact_directory) / 'resolved_run_plan.json')]
    with Supervisor(catalog, worker_command=worker, viewer_port=0) as supervisor:
        supervisor.submit(plan)
        wait_for(lambda: supervisor.get_run(plan.run_id)['state'] == 'RUNNING')
        assert _run_log_page(supervisor, plan.run_id, 'viser', 0, 10)['availability'] == 'not_started'
        supervisor.patch_display(plan.run_id, {'enabled': True})
        ready(supervisor, plan.run_id)
        supervisor.cancel(plan.run_id)
        wait_for(lambda: supervisor.get_run(plan.run_id)['state'] == 'CANCELLED')
        assert supervisor.get_run(plan.run_id)['viser_log_started'] is True
        assert _run_log_page(supervisor, plan.run_id, 'viser', 0, 10)['availability'] == 'available'
        assert any(a['kind'] == 'viser_log' for a in supervisor.artifacts(plan.run_id))
        Path(plan.artifact_directory, 'logs/viser.log').unlink()
        with pytest.raises(FileNotFoundError):
            _run_log_page(supervisor, plan.run_id, 'viser', 0, 10)


def test_metric_partial_line_cursor(tmp_path):
    # 半行故意截在中文 UTF-8 字符内部，读取器不能提前解码或推进。
    first_line = b'{"event_seq":1,"payload":{"loss":1.5}}\n'
    second_line = b'{"event_seq":2,"payload":{"loss":1.0}}\n'
    third_line = '{"event_seq":3,"payload":{"label":"中文"}}\n'.encode()
    split = third_line.index('中'.encode()) + 1
    path = tmp_path / 'metrics.jsonl'
    path.write_bytes(first_line + second_line + third_line[:split])

    first = read_metric_page(tmp_path, offset=0, limit=500, terminal=False)
    assert first == {
        'records': [
            {'event_seq': 1, 'payload': {'loss': 1.5}},
            {'event_seq': 2, 'payload': {'loss': 1.0}},
        ],
        'next_offset': len(first_line + second_line),
        'eof': True, 'terminal': False, 'availability': 'available',
    }
    pending = read_metric_page(tmp_path, offset=first['next_offset'], limit=500, terminal=False)
    assert pending['records'] == []
    assert pending['next_offset'] == first['next_offset']
    assert pending['eof'] is True

    with path.open('ab') as writer:
        writer.write(third_line[split:])
    second = read_metric_page(tmp_path, offset=first['next_offset'], limit=500, terminal=True)
    assert second == {
        'records': [{'event_seq': 3, 'payload': {'label': '中文'}}],
        'next_offset': path.stat().st_size,
        'eof': True, 'terminal': True, 'availability': 'available',
    }


def test_metric_limit_offset_and_full_event(tmp_path):
    # 非法前缀不属于请求页；从指定字节边界直接读取后续完整事件。
    prefix = b'not JSON\n'
    event = {
        'event_seq': 42, 'run_id': 'run-42', 'kind': 'metric',
        'sim_time': 1.25, 'wall_time': 1800000000.5,
        'payload': {'source': 'learner', 'steps': 128, 'updates': 4,
                    'train': {'loss': 0.25}, 'label': '中文'},
    }
    line = (json.dumps(event, ensure_ascii=False) + '\n').encode()
    last = b'{"event_seq":43,"payload":{"train":{"loss":0.125}}}\n'
    path = tmp_path / 'metrics.jsonl'
    path.write_bytes(prefix + line + last)

    page = read_metric_page(tmp_path, offset=len(prefix), limit=1, terminal=True)
    assert page['records'] == [event]
    assert page['next_offset'] == len(prefix + line)
    assert page['eof'] is False
    final = read_metric_page(tmp_path, offset=page['next_offset'], limit=1, terminal=True)
    assert final['records'] == [{'event_seq': 43, 'payload': {'train': {'loss': 0.125}}}]
    assert final['eof'] is True
    assert final['next_offset'] == path.stat().st_size
    empty = read_metric_page(tmp_path, offset=final['next_offset'], limit=1, terminal=True)
    assert empty['records'] == []
    assert empty['next_offset'] == final['next_offset']
    assert empty['eof'] is True


@pytest.mark.parametrize('terminal', [False, True])
def test_metric_empty_file(tmp_path, terminal):
    (tmp_path / 'metrics.jsonl').touch()
    assert read_metric_page(tmp_path, offset=0, limit=500, terminal=terminal) == {
        'records': [], 'next_offset': 0, 'eof': True,
        'terminal': terminal, 'availability': 'available',
    }


def test_metric_more_than_display_window_pages_and_append(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    with path.open('w', encoding='utf-8') as writer:
        for seq in range(1, 10038):
            writer.write(json.dumps({'event_seq': seq, 'payload': {'value': seq}}) + '\n')

    offset = 0
    sequences = []
    for _ in range(21):
        page = read_metric_page(tmp_path, offset=offset, limit=500, terminal=False)
        assert len(page['records']) <= 500
        assert page['next_offset'] > offset
        sequences.extend(record['event_seq'] for record in page['records'])
        offset = page['next_offset']
    assert sequences == list(range(1, 10038))
    assert page['eof'] is True
    assert offset == path.stat().st_size

    with path.open('ab') as writer:
        writer.write(b'{"event_seq":10038,"payload":{"value":10038}}\n')
        writer.flush()
        # 文件仍可写时，下一页必须看到新增事件而非旧缓存。
        appended = read_metric_page(tmp_path, offset=offset, limit=500, terminal=False)
    assert appended['records'] == [{'event_seq': 10038, 'payload': {'value': 10038}}]
    assert appended['next_offset'] == path.stat().st_size


@pytest.mark.parametrize('invalid', [b'{broken}\n', b'\n', b'{"event_seq":2}\n{broken}\n'])
def test_metric_corrupt_complete_json_is_explicit(tmp_path, invalid):
    (tmp_path / 'metrics.jsonl').write_bytes(invalid)
    with pytest.raises(json.JSONDecodeError):
        read_metric_page(tmp_path, offset=0, limit=500, terminal=True)


@pytest.mark.parametrize('channel', ['worker', 'viser'])
def test_log_utf8_chunks(tmp_path, channel):
    (tmp_path / 'logs').mkdir()
    raw = b'A\xe4\xb8\xad\xe6\x96\x87\xff\n'
    path = tmp_path / 'logs' / f'{channel}.log'
    path.write_bytes(raw)
    offset = 0
    chunks = []
    for expected_offset in (2, 4, 6, 8, 9):
        page = read_log_page(tmp_path, channel=channel, offset=offset, limit=2)
        chunk = base64.b64decode(page['data_base64'], validate=True)
        chunks.append(chunk)
        assert len(chunk) <= 2
        assert page['next_offset'] == expected_offset
        assert page['eof'] is (expected_offset == 9)
        assert page['availability'] == 'available'
        offset = page['next_offset']
    assert b''.join(chunks) == raw

    decoder = codecs.getincrementaldecoder('utf-8')()
    assert decoder.decode(chunks[0]) == 'A'
    assert decoder.decode(chunks[1]) == '中'
    assert decoder.decode(chunks[2]) == ''
    with pytest.raises(UnicodeDecodeError):
        decoder.decode(chunks[3])
    # 非法字节由消费端显式诊断，服务端不替换或丢弃原文。
    assert raw.decode('utf-8', errors='replace') == 'A中文\ufffd\n'

    end = read_log_page(tmp_path, channel=channel, offset=offset, limit=2)
    assert end == {
        'data_base64': '', 'next_offset': len(raw),
        'eof': True, 'availability': 'available',
    }
    with path.open('ab') as writer:
        writer.write(b'tail')
    appended = read_log_page(tmp_path, channel=channel, offset=offset, limit=65536)
    assert base64.b64decode(appended['data_base64']) == b'tail'
    assert appended['next_offset'] == len(raw) + 4


@pytest.mark.parametrize('channel', ['worker', 'viser'])
def test_log_empty_file(tmp_path, channel):
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'logs' / f'{channel}.log').touch()
    assert read_log_page(tmp_path, channel=channel, offset=0, limit=65536) == {
        'data_base64': '', 'next_offset': 0,
        'eof': True, 'availability': 'available',
    }


@pytest.mark.parametrize('channel', ['worker', 'viser'])
def test_log_large_page_is_bounded(tmp_path, channel):
    (tmp_path / 'logs').mkdir()
    path = tmp_path / 'logs' / f'{channel}.log'
    raw = bytes(range(256)) * 2049
    path.write_bytes(raw)
    first = read_log_page(tmp_path, channel=channel, offset=0, limit=262144)
    assert base64.b64decode(first['data_base64']) == raw[:262144]
    assert first['next_offset'] == 262144
    assert first['eof'] is False
    second = read_log_page(tmp_path, channel=channel, offset=262144, limit=262144)
    assert base64.b64decode(second['data_base64']) == raw[262144:524288]
    assert second['next_offset'] == 524288
    assert second['eof'] is False
    final = read_log_page(tmp_path, channel=channel, offset=524288, limit=262144)
    assert base64.b64decode(final['data_base64']) == bytes(range(256))
    assert final['next_offset'] == len(raw)
    assert final['eof'] is True


def test_missing_metric_file_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_metric_page(tmp_path, offset=0, limit=500, terminal=False)


@pytest.mark.parametrize('channel', ['worker', 'viser'])
def test_missing_log_file_is_explicit(tmp_path, channel):
    (tmp_path / 'logs').mkdir()
    with pytest.raises(FileNotFoundError):
        read_log_page(tmp_path, channel=channel, offset=0, limit=65536)
