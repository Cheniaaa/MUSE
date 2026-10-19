"""Run客户端私有选择与同帧实体投影；使用真实CPU Worker/Viser链路。"""
import json
import os
from pathlib import Path
import signal
import struct

from muse_sdk.run_protocol import PreviewFrame, encode_preview, preview_socketpair
from test_display_session import display_run, ready
from test_supervisor import wait_for


def test_two_view_clients_keep_selection_private(display_run):
    supervisor, plan = display_run
    first = supervisor.register_view_client(plan.run_id)
    second = supervisor.register_view_client(plan.run_id)
    assert first['view_id'] != second['view_id']
    supervisor.patch_display(plan.run_id, {'viser': {'enabled': True}})
    ready(supervisor, plan.run_id)
    snapshot = supervisor.entities(plan.run_id)
    assert snapshot['available'] and snapshot['objects']
    identity = snapshot['objects'][0]['object_id']
    with supervisor._lock:
        session = supervisor._displays[plan.run_id]
        session._viewer_event(session.slots['viser'], dict(kind='view', view_id=first['view_id'], event=dict(
            kind='selection', object_id=identity, frame=snapshot['frame'])))
    first_events = supervisor.view_events(plan.run_id, first['view_id'])
    second_events = supervisor.view_events(plan.run_id, second['view_id'])
    assert any(e.get('object_id') == identity for e in first_events)
    assert not any(e.get('object_id') == identity for e in second_events)
    supervisor.remove_view_client(plan.run_id, first['view_id'])
    assert supervisor.get_display(plan.run_id)['viser']['requested']['enabled']
    assert supervisor.get_run(plan.run_id)['state'] == 'RUNNING'


def test_pick_and_entities_follow_frame_identity(display_run):
    supervisor, plan = display_run
    client = supervisor.register_view_client(plan.run_id)
    supervisor.patch_display(plan.run_id, {'viser': {'enabled': True, 'world_id': 1}})
    ready(supervisor, plan.run_id)
    pid = supervisor.get_run(plan.run_id)['pid']
    os.kill(pid, signal.SIGSTOP)
    sender, receiver = preview_socketpair()
    try:
        snapshot = supervisor.entities(plan.run_id)
        assert snapshot['available'] and snapshot['frame']['world_id'] == 1
        assert snapshot['objects'][0]['pose']['position'] == [1., 0., 1.]
        with supervisor._lock:
            session = supervisor._displays[plan.run_id]
            old = PreviewFrame.model_validate(snapshot['frame']).model_copy(update={'world_id': 0, 'decision': 99})
            values = [99.] * (old.nbody * 3) + [1., 0., 0., 0., 1., 0., 0., 0., 1.] * old.nbody
            sender.send(encode_preview(old, struct.pack('<' + 'f' * len(values), *values)))
            previous = session.worker_socket
            session.worker_socket = receiver
            try:
                session.tick()
            finally:
                session.worker_socket = previous
            session._viewer_event(session.slots['viser'], dict(kind='view', view_id=client['view_id'], event=dict(
                kind='selection', object_id=snapshot['objects'][0]['object_id'], frame=old.model_dump())))
            assert not any(e.get('kind') == 'selection' for e in session.client_events(client['view_id']))
            current = session.entities()
            assert current['frame'] == snapshot['frame']
            assert current['objects'][0]['pose']['position'] == [1., 0., 1.]
            session._viewer_event(session.slots['viser'], dict(kind='view', view_id=client['view_id'], event=dict(
                kind='selection', object_id=current['objects'][0]['object_id'], frame=current['frame'])))
            picked = next(e for e in session.client_events(client['view_id']) if e['kind'] == 'selection')
            assert picked['frame'] == current['frame']
            # Viewer网络传输可落后同episode若干tick；选择保留其真实渲染帧，不冒称当前实体帧。
            rendered = {**current['frame'], 'decision': max(0, current['frame']['decision'] - 1),
                        'tick': max(0, current['frame']['tick'] - 20)}
            session._viewer_event(session.slots['viser'], dict(kind='view', view_id=client['view_id'], event=dict(
                kind='selection', object_id=current['objects'][0]['object_id'], frame=rendered)))
            assert next(e for e in session.client_events(client['view_id']) if e['kind'] == 'selection')['frame'] == rendered
    finally:
        os.kill(pid, signal.SIGCONT)
        sender.close()
        receiver.close()


def test_terminal_entities_remain_identifiable_without_loading_model(display_run):
    from muse_server.display import DisplaySession
    supervisor, plan = display_run
    supervisor.patch_display(plan.run_id, {'viser': {'enabled': True}})
    ready(supervisor, plan.run_id)
    supervisor.cancel(plan.run_id)
    wait_for(lambda: supervisor.get_run(plan.run_id)['state'] == 'CANCELLED')
    snapshot = supervisor.entities(plan.run_id)
    assert not snapshot['available'] and snapshot['reason'] and snapshot['objects']
    expected = json.loads(Path(plan.artifact_directory, 'preview_objects.json').read_text())
    reopened = DisplaySession(plan, lambda *args: None, send_command=lambda *args: None, on_log_started=lambda: None)
    reopened.release()
    restored = reopened.entities()
    assert not restored['available']
    assert [obj['object_id'] for obj in restored['objects']] == [obj['object_id'] for obj in expected]
    assert all('body_id' not in obj and 'local_position' not in obj for obj in restored['objects'])


def test_environment_pose_projects_local_offset_in_same_frame(tmp_path):
    from types import SimpleNamespace
    from muse_sdk.run_protocol import SceneObject
    from muse_server.display import DisplaySession
    plan = SimpleNamespace(artifact_directory=str(tmp_path), experiment=SimpleNamespace(
        preview=SimpleNamespace(world_ids=[0], rate_hz=10.)))
    session = DisplaySession(plan, lambda *args: None, send_command=lambda *args: None, on_log_started=lambda: None)
    session._objects = [SceneObject(object_id='wall', label='墙', kind='environment', group_id=None,
        group_slot=None, role_id=None, model_id=None, body_id=0, geom_ids=[0],
        local_position=[1., 2., 3.], local_orientation_matrix=[1., 0., 0., 0., 1., 0., 0., 0., 1.])]
    header = PreviewFrame(run_id='r', viewer='viser', world_id=1, selection_seq=1, episode=2, decision=3,
                          tick=60, sim_time=.12, scene_hash='a' * 64, nbody=1, payload_floats=12)
    slot = session.slots['viser']
    slot.requested = slot.effective = slot.effective.model_copy(update={'enabled': True, 'world_id': 1})
    slot.selection_seq = slot.effective_seq = 1
    # 绕Z轴90度：局部(1,2,3)→(-2,1,3)，再加body(4,5,6)。
    session._project_entities(header, struct.pack('<12f', 4., 5., 6., 0., -1., 0., 1., 0., 0., 0., 0., 1.))
    state = session.entities()
    assert state['available'] and state['frame'] == header.model_dump()
    assert state['objects'][0]['pose'] == {'position': [2., 6., 9.],
        'orientation_matrix': [0., -1., 0., 1., 0., 0., 0., 0., 1.]}
    session.release()
    assert not session.entities()['available']
    assert session.entities()['objects'][0]['pose']['position'] == [2., 6., 9.]


def test_view_control_disconnect_keeps_shared_display_and_entities_ws(tmp_path):
    from fastapi.testclient import TestClient
    from muse_server.app import create_app
    from test_supervisor import make_plan
    from test_display_session import DISPLAY_CHILD
    import sys
    catalog, plan = make_plan(tmp_path)
    worker = lambda p: [sys.executable, '-c', DISPLAY_CHILD,
                        str(Path(p.artifact_directory) / 'resolved_run_plan.json')]
    with TestClient(create_app(catalog.directory, worker_command=worker, viewer_port=0)) as client:
        run = client.post('/runs', json=dict(experiment_id=plan.experiment_id, mode='EVALUATE', max_decision_steps=8)).json()
        run_id = run['run_id']
        wait_for(lambda: client.get(f'/runs/{run_id}').json()['state'] == 'RUNNING')
        view_id = client.post(f'/runs/{run_id}/display/clients').json()['view_id']
        client.patch(f'/runs/{run_id}/display', json={'viser': {'enabled': True}})
        ready(client.app.state.supervisor, run_id)
        with client.websocket_connect(f'/streams/runs/{run_id}/entities') as entities_ws:
            snapshot = entities_ws.receive_json()
            assert snapshot['available'] and snapshot['objects']
        with client.websocket_connect(f'/streams/runs/{run_id}/view?view_id={view_id}') as ws:
            ws.send_json({'kind': 'visibility'})
            error = ws.receive_json()
            assert error['kind'] == 'error' and error['view_id'] == view_id
        assert client.get(f'/runs/{run_id}/display').json()['viser']['requested']['enabled']
        assert client.get(f'/runs/{run_id}').json()['state'] == 'RUNNING'
