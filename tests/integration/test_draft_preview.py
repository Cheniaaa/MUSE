"""CPU预览会话边界：真实资源解析、单处理/最新待办、断连与未订阅回收。"""
import json
from pathlib import Path
import sys

import pytest

from test_supervisor import wait_for


def draft_request(pipeline, revision=1):
    catalog, source = pipeline
    env = catalog.get_resource('environment', source['environment_id'], source['environment_revision'])
    return dict(draft_id='draft', revision=revision,
        environment=env.model_dump(include={'resource_id', 'revision', 'sha256'}), entities=[dict(
            object_id='scout', group_id='scout_group', slot=0, model_id='bitcraze_crazyflie_2', model_revision=1,
            pose=dict(position=[1., 2., 3.], orientation_wxyz=[1., 0., 0., 0.]))])


CHILD = '''import json, os, sys, time
from pathlib import Path
root = Path(sys.argv[1])
(root / 'pid').write_text(str(os.getpid()))
for line in sys.stdin:
    message = json.loads(line)
    if message['kind'] == 'compile':
        revision = message['request']['revision']
        with (root / 'revisions').open('a') as f: f.write(str(revision) + '\\n')
        if revision == 1:
            while not (root / 'release').exists(): time.sleep(.01)
        print(json.dumps(dict(kind='ready', revision=revision, viewer_url='http://127.0.0.1:1234',
                              objects=[], sensor_omissions=[])), flush=True)
    elif message['kind'] == 'intent':
        print(json.dumps(dict(kind='view', event=message['intent'])), flush=True)
'''


def test_latest_revision_wins_and_disconnect_closes(pipeline, tmp_path):
    from muse_server.previews import PreviewManager
    from muse_server.supervisor import process_identity
    manager = PreviewManager(pipeline[0], process_command=lambda identity: [sys.executable, '-c', CHILD, str(tmp_path)])
    try:
        state = manager.create(draft_request(pipeline))
        identity = state['session_id']
        manager.attach(identity)
        wait_for(lambda: (tmp_path / 'pid').exists())
        pid = int((tmp_path / 'pid').read_text())
        manager.update(identity, draft_request(pipeline, 2))
        manager.update(identity, draft_request(pipeline, 3))
        (tmp_path / 'release').touch()
        wait_for(lambda: manager.get(identity)['status'] == 'ready')
        state = manager.get(identity)
        assert state['applied_revision'] == 3
        assert (tmp_path / 'revisions').read_text().splitlines() == ['1', '3']
        assert not any(event.get('status') == 'ready' and event.get('applied_revision') == 1
                       for event in manager.events(identity))
        manager.close(identity)
        assert process_identity(pid) is None
        with pytest.raises(KeyError):
            manager.get(identity)
    finally:
        manager.shutdown()


def test_unattached_preview_expires(pipeline, tmp_path):
    from muse_server.previews import PreviewManager
    from muse_server.supervisor import process_identity
    manager = PreviewManager(pipeline[0], attach_timeout=.3,
        process_command=lambda identity: [sys.executable, '-c', CHILD, str(tmp_path)])
    try:
        identity = manager.create(draft_request(pipeline))['session_id']
        wait_for(lambda: (tmp_path / 'pid').exists())
        pid = int((tmp_path / 'pid').read_text())
        wait_for(lambda: process_identity(pid) is None)
        with pytest.raises(KeyError):
            manager.get(identity)
    finally:
        manager.shutdown()


def test_preview_rejects_forged_resource_or_client_paths(pipeline):
    from muse_server.previews import PreviewManager
    from muse_sdk.config import ConfigError
    manager = PreviewManager(pipeline[0])
    request = draft_request(pipeline)
    request['environment']['sha256'] = '0' * 64
    with pytest.raises(ConfigError):
        manager.create(request)
    request = draft_request(pipeline)
    request['entities'][0]['path'] = '/somewhere'
    with pytest.raises(ValueError):
        manager.create(request)
    request = draft_request(pipeline)
    request['entities'][0]['model_revision'] = 999
    with pytest.raises(ConfigError):
        manager.create(request)
    manager.shutdown()


def test_incomplete_research_can_preview_geometry_and_control_ws_cleans(pipeline):
    from fastapi.testclient import TestClient
    from muse_server.app import create_app
    from muse_server.supervisor import process_identity
    with TestClient(create_app(pipeline[0].directory, viewer_port=0)) as client:
        response = client.post('/previews', json=draft_request(pipeline))
        assert response.status_code == 201, response.text
        identity = response.json()['session_id']
        with client.websocket_connect('/streams/previews/' + identity) as ws:
            state = ws.receive_json()
            while state.get('status') != 'ready':
                assert state.get('status') != 'error', state
                state = ws.receive_json()
            assert state['applied_revision'] == 1 and state['viewer_url'].startswith('http://127.0.0.1:')
            assert 'scout' in state['sensor_omissions']
            scout = next(obj for obj in state['objects'] if obj['object_id'] == 'scout')
            assert 'body_id' not in scout and 'geom_ids' not in scout
            assert scout['model_id'] == 'bitcraze_crazyflie_2'
            assert client.get('/runs').json()['items'] == []
        wait_for(lambda: client.get('/previews/' + identity).status_code == 404)
        assert client.get('/health').json()['gpu_busy'] is False
