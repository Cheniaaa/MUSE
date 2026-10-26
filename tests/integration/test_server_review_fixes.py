"""Server组合边界：独立owner清理、真实查找缺失与内部错误归属。"""
import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('failures', [(), ('replays',), ('previews',), ('supervisor',),
                                    ('replays', 'previews', 'supervisor')])
def test_lifespan_attempts_every_owner_and_preserves_failure_identity(tmp_path, monkeypatch, failures):
    from muse_server import app as module

    calls = []
    catalog = SimpleNamespace(directory=tmp_path)
    monkeypatch.setattr(module, 'bootstrap', lambda *args: catalog)

    def shutdown(owner):
        calls.append(owner)
        if owner in failures:
            raise RuntimeError(owner + ' shutdown failed')

    class Supervisor:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def close(self, *, before_unlock):
            shutdown('supervisor')
            if before_unlock is not None:
                before_unlock()

    monkeypatch.setattr(module, 'Supervisor', Supervisor)
    monkeypatch.setattr(module, 'ReplayManager', lambda *args:
        SimpleNamespace(shutdown=lambda: shutdown('replays')))
    monkeypatch.setattr(module, 'PreviewManager', lambda *args:
        SimpleNamespace(shutdown=lambda: shutdown('previews')))
    app = module.create_app(tmp_path)

    async def enter_and_leave():
        async with app.router.lifespan_context(app):
            assert (tmp_path / 'server-instance.json').is_file()

    if failures:
        with pytest.raises(RuntimeError) as raised:
            asyncio.run(enter_and_leave())
        messages, error = [], raised.value
        while error is not None:
            messages.append(str(error))
            error = error.__context__
        assert set(messages) == {owner + ' shutdown failed' for owner in failures}
    else:
        asyncio.run(enter_and_leave())
    assert calls == ['replays', 'previews', 'supervisor']
    assert (tmp_path / 'server-instance.json').exists() == bool(failures)


@pytest.fixture
def boundary_client(pipeline):
    from fastapi.testclient import TestClient
    from muse_server.app import create_app

    with TestClient(create_app(pipeline[0].directory), raise_server_exceptions=False) as client:
        yield client


@pytest.mark.parametrize('target', ['artifacts', 'replay'])
def test_internal_missing_field_is_not_reported_as_unknown_resource(boundary_client, target):
    client = boundary_client
    record = dict(run_id='broken', state='COMPLETED', pid=None)
    if target == 'replay':
        record['artifacts'] = [{}]
    client.app.state.supervisor._save(record)
    response = (client.get('/runs/broken/artifacts') if target == 'artifacts' else
                client.post('/replays', json=dict(run_id='broken', trajectory_artifact_id='trajectory')))
    assert response.status_code == 500, response.text
    assert 'Unknown resource' not in response.text


def test_internal_view_lookup_error_is_not_a_missing_session_disconnect(boundary_client):
    client = boundary_client
    client.app.state.supervisor._save(dict(run_id='broken', state='COMPLETED', pid=None))
    with pytest.raises(KeyError, match='plan_path'):
        with client.websocket_connect('/streams/runs/broken/view?view_id=view'):
            pass


@pytest.mark.parametrize('method,path,payload', [
    ('GET', '/runs/missing', None),
    ('GET', '/runs/missing/artifacts', None),
    ('POST', '/runs/missing/cancel', None),
    ('GET', '/previews/missing', None),
    ('GET', '/replays/missing', None),
    ('DELETE', '/replays/missing', None),
    ('POST', '/replays', {'run_id': 'missing', 'trajectory_artifact_id': 'trajectory'}),
    ('POST', '/replays/missing/recordings', {'world_id': 0, 'episode': 0, 'fps': 25., 'width': 640, 'height': 480}),
    ('DELETE', '/replays/missing/recordings/missing', None),
])
def test_unknown_resources_remain_404(boundary_client, method, path, payload):
    response = boundary_client.request(method, path, json=payload)
    assert response.status_code == 404, response.text


@pytest.mark.parametrize('path', ['/streams/previews/missing', '/streams/replays/missing',
    '/streams/runs/missing/events', '/streams/runs/missing/logs', '/streams/runs/missing/entities',
    '/streams/runs/missing/view?view_id=view'])
def test_unknown_stream_resource_is_rejected_before_accept(boundary_client, path):
    from fastapi import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as raised:
        with boundary_client.websocket_connect(path):
            pass
    assert raised.value.code == 1008
