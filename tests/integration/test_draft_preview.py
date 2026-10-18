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
        if revision == 2:
            while (root / 'hold-second').exists(): time.sleep(.01)
        print(json.dumps(dict(kind='ready', revision=revision, viewer_url='http://127.0.0.1:1234',
                              objects=[])), flush=True)
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
        second = draft_request(pipeline, 2)
        second['entities'][0]['sensor_profile_id'] = 'none'
        manager.update(identity, second)
        third = draft_request(pipeline, 3)
        third['entities'][0].update(sensor_profile_id='cf2_imu_rangefinder_down',
                                    sensor_config={'imu_update_rate_hz': 100.})
        manager.update(identity, third)
        (tmp_path / 'release').touch()
        wait_for(lambda: manager.get(identity)['status'] == 'ready')
        state = manager.get(identity)
        assert state['applied_revision'] == 3
        assert state['sensor_omissions'] == ['scout: missing rangefinder_update_rate_hz']
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


def test_view_intent_overtaken_by_http_compile_returns_current_state(pipeline, tmp_path):
    from fastapi.testclient import TestClient
    from muse_server.app import create_app
    from muse_server.previews import PreviewManager
    (tmp_path / 'release').touch()
    (tmp_path / 'hold-second').touch()
    app = create_app(pipeline[0].directory, viewer_port=0)
    with TestClient(app) as client:
        app.state.previews = PreviewManager(app.state.catalog,
            process_command=lambda identity: [sys.executable, '-c', CHILD, str(tmp_path)])
        identity = client.post('/previews', json=draft_request(pipeline)).json()['session_id']
        with client.websocket_connect('/streams/previews/' + identity) as ws:
            state = ws.receive_json()
            while state.get('status') != 'ready':
                state = ws.receive_json()
            assert client.put('/previews/' + identity, json=draft_request(pipeline, 2)).json()['status'] == 'preparing'
            state = ws.receive_json()
            assert state['kind'] == 'state' and state['status'] == 'preparing'
            # ready时发出的意图被另一个HTTP通道的编译超越，仍需返回真实状态而非虚构编译失败。
            ws.send_json({'kind': 'reset_camera'})
            response = ws.receive_json()
            assert response['kind'] == 'state' and response['status'] == 'preparing'
            ws.send_json({'kind': 'invalid'})
            assert ws.receive_json()['kind'] == 'error'
            (tmp_path / 'hold-second').unlink()
            state = ws.receive_json()
            while state.get('status') != 'ready':
                assert state['kind'] == 'state'
                state = ws.receive_json()
            assert state['applied_revision'] == 2
            ws.send_json({'kind': 'reset_camera'})
            assert ws.receive_json()['kind'] == 'view'


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


def test_public_sensor_omissions_follow_profile_and_applied_revision_and_ws_cleans(pipeline):
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
            assert state['sensor_omissions'] == ['scout']
            scout = next(obj for obj in state['objects'] if obj['object_id'] == 'scout')
            assert 'body_id' not in scout and 'geom_ids' not in scout
            assert scout['model_id'] == 'bitcraze_crazyflie_2'
            for revision, profile, config, omissions in (
                (2, 'cf2_imu_rangefinder_down', {},
                 ['scout: missing imu_update_rate_hz, rangefinder_update_rate_hz']),
                (3, 'cf2_imu_rangefinder_down', {'imu_update_rate_hz': 100.},
                 ['scout: missing rangefinder_update_rate_hz']),
                (4, 'none', {}, []),
                (5, 'cf2_imu_rangefinder_down',
                 {'imu_update_rate_hz': 100., 'rangefinder_update_rate_hz': 20.}, []),
                (6, None, {}, ['scout']),
            ):
                request = draft_request(pipeline, revision)
                request['entities'][0].update(sensor_profile_id=profile, sensor_config=config)
                response = client.put('/previews/' + identity, json=request)
                assert response.status_code == 200, response.text
                while True:
                    state = ws.receive_json()
                    assert state.get('status') != 'error', state
                    if state.get('status') == 'ready' and state['applied_revision'] == revision:
                        break
                assert state['sensor_omissions'] == omissions
                public = client.get('/previews/' + identity).json()
                assert public['applied_revision'] == revision
                assert public['sensor_omissions'] == omissions
            assert client.get('/runs').json()['items'] == []
        wait_for(lambda: client.get('/previews/' + identity).status_code == 404)
        assert client.get('/health').json()['gpu_busy'] is False
