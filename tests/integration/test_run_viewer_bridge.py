"""B6 的真实 CPU/Viser/JSON 帧桥；身份适配器不冒充 Qt profile。"""
import asyncio
import json
import queue
import subprocess
import sys
import threading
import urllib.request

import mujoco
import numpy as np
import pytest
import viser

from muse_sdk.view_protocol import SceneObject, ViewIntent
from test_draft_session import client_adapter, native_scene, servers


def test_two_run_views_route_selection_and_browser_remains_private(native_scene, servers):
    from muse_server.viewers.scene_interaction import SceneInteraction

    model, objects, snapshots = native_scene
    events = []
    interaction = SceneInteraction(servers(), model, objects, view_id=None, emit=events.append)
    interaction.set_view_ids(['qt-a', 'qt-b'])
    interaction.update_frame(*snapshots[0][:2], {'world_id': 1, 'episode': 2, 'tick': 100})
    clients = [client_adapter(servers(), identity, claim) for identity, claim in
               [(1, 'qt-a'), (2, 'qt-b'), (3, 'unregistered'), (4, None)]]
    try:
        for client in clients:
            interaction.connect_client(client)
        assert [event['view_id'] for event in events] == ['qt-a', 'qt-b', None, None]
        interaction.apply_intent(ViewIntent(kind='select', object_id='moving'), view_id='qt-a')
        assert interaction._clients[1].selected == 'moving'
        assert interaction._clients[2].selected is None
        assert events[-1]['view_id'] == 'qt-a'
        interaction.apply_intent(ViewIntent(kind='visibility', object_id='moving', visible=False), view_id='qt-a')
        assert not clients[0].scene.get_handle_by_name('/muse/objects/0').visible
        assert clients[1].scene.get_handle_by_name('/muse/objects/0').visible
        interaction.apply_intent(ViewIntent(kind='visibility', object_id='moving', visible=True), view_id='qt-a')
        clients[0].camera.position = (9., 9., 9.)
        own_before = clients[0].camera.position.copy()
        before = clients[1].camera.position.copy()
        interaction.apply_intent(ViewIntent(kind='focus', object_id='moving'), view_id='qt-a')
        assert not np.array_equal(clients[0].camera.position, own_before)
        np.testing.assert_array_equal(clients[1].camera.position, before)
        with pytest.raises(ValueError, match='registered'):
            interaction.apply_intent(ViewIntent(kind='select', object_id='moving'), view_id='unregistered')
        interaction.apply_intent(ViewIntent(kind='select', object_id='moving'), client_id=3)
        assert interaction._clients[3].selected is None
        interaction.apply_intent(ViewIntent(kind='select', object_id='moving'), client_id=4)
        assert interaction._clients[4].selected == 'moving'
        assert events[-1]['view_id'] is None
        assert events[-1]['event']['frame'] == {'world_id': 1, 'episode': 2, 'tick': 100}
        assert all(state.gizmo is None for state in interaction._clients.values())
    finally:
        interaction.close()


def test_late_registration_removal_and_reconnect_do_not_requery_or_cross_bind(native_scene, servers):
    from muse_server.viewers.scene_interaction import SceneInteraction

    model, objects, snapshots = native_scene
    events = []
    interaction = SceneInteraction(servers(), model, objects, view_id=None, emit=events.append)
    interaction.update_frame(*snapshots[0][:2], {'tick': 0})
    a = client_adapter(servers(), 1, 'qt-a')
    b = client_adapter(servers(), 2, 'qt-b')
    try:
        interaction.connect_client(a)
        interaction.connect_client(b)
        assert not interaction._clients[1].paired and not interaction._clients[2].paired
        interaction.set_view_ids(['qt-a'])
        interaction.apply_intent(ViewIntent(kind='select', object_id='moving'), view_id='qt-a')
        assert interaction._clients[1].selected == 'moving'
        interaction.set_view_ids(['qt-b'])
        assert interaction._clients[1].selected is None and not interaction._clients[1].paired
        removed = next(event for event in reversed(events) if event['event']['client_id'] == 1)
        assert removed['view_id'] is None and not removed['event']['paired']
        assert a.scene.get_handle_by_name('/muse/highlight') is None
        assert interaction._clients[2].paired
        interaction.apply_intent(ViewIntent(kind='select', object_id='moving'), view_id='qt-b')
        assert interaction._clients[1].selected is None
        interaction.set_view_ids(['qt-a', 'qt-b'])
        assert interaction._clients[1].paired and interaction._clients[1].selected is None
        assert interaction._clients[2].selected == 'moving'
        assert a.local_storage.calls == b.local_storage.calls == [('muse_view_id', 5.0)]
        count = len(events)
        interaction.set_view_ids(['qt-b', 'qt-a'])
        assert len(events) == count
        interaction.disconnect_client(a)
        assert events[-1]['view_id'] == 'qt-a' and not events[-1]['event']['connected']
        replacement = client_adapter(servers(), 3, 'qt-a')
        interaction.connect_client(replacement)
        assert interaction._clients[3].paired and interaction._clients[3].selected is None
        assert interaction._clients[2].selected == 'moving'
    finally:
        interaction.close()


def test_registration_during_storage_wait_is_nonblocking_and_uses_latest_registry(native_scene, servers):
    from muse_server.viewers.scene_interaction import SceneInteraction

    model, objects, _ = native_scene
    interaction = SceneInteraction(servers(), model, objects, view_id=None, emit=lambda _: None)
    client = client_adapter(servers(), 1, 'qt-a')
    entered, release = threading.Event(), threading.Event()
    class WaitingStorage:
        def get_item(self, key, timeout):
            entered.set()
            assert release.wait(5)
            return 'qt-a'
    client.local_storage = WaitingStorage()
    worker = threading.Thread(target=interaction.connect_client, args=(client,))
    worker.start()
    try:
        assert entered.wait(5)
        interaction.set_view_ids(['qt-b'])
        interaction.set_view_ids(['qt-a'])
        release.set()
        worker.join(timeout=5)
        assert not worker.is_alive() and interaction._clients[1].paired
    finally:
        release.set()
        worker.join(timeout=5)
        interaction.close()


def test_failed_storage_is_local_and_does_not_disable_other_views(native_scene, servers):
    from muse_server.viewers.scene_interaction import SceneInteraction

    model, objects, _ = native_scene
    events = []
    interaction = SceneInteraction(servers(), model, objects, view_id=None, emit=events.append)
    interaction.set_view_ids(['qt-a', 'qt-b'])
    bad, good = client_adapter(servers(), 1, 'qt-a'), client_adapter(servers(), 2, 'qt-b')
    class TimedOutStorage:
        def get_item(self, key, timeout):
            raise TimeoutError('Client did not answer')
    bad.local_storage = TimedOutStorage()
    try:
        interaction.connect_client(bad)
        interaction.connect_client(good)
        assert events[0]['view_id'] is None and events[0]['event']['message'] == 'Client did not answer'
        assert events[1]['view_id'] == 'qt-b' and interaction._clients[2].paired
        interaction.set_view_ids(['qt-b'])
        assert interaction._clients[2].paired
        assert good.local_storage.calls == [('muse_view_id', 5.0)]
    finally:
        interaction.close()


def test_browser_click_and_qt_click_report_the_current_dynamic_frame(native_scene, servers, monkeypatch):
    from muse_server.viewers.scene_interaction import SceneInteraction

    model, objects, snapshots = native_scene
    events = []
    interaction = SceneInteraction(servers(), model, objects, view_id=None, emit=events.append)
    interaction.set_view_ids(['qt'])
    qt, browser = client_adapter(servers(), 1, 'qt'), client_adapter(servers(), 2, None)
    interaction.connect_client(qt)
    interaction.connect_client(browser)
    def forbidden(*args, **kwargs):
        raise AssertionError('Run picker must consume the actual body snapshot')
    monkeypatch.setattr(mujoco, 'mj_forward', forbidden)
    try:
        for index, client in enumerate((qt, browser)):
            current = {'world_id': index, 'episode': index + 2, 'tick': 100 + index}
            interaction.update_frame(*snapshots[index][:2], current)
            event = viser.SceneClickEvent(client=client, client_id=client.client_id,
                ray_origin=tuple(snapshots[index][2]), ray_direction=(0., 1., 0.), screen_pos=(.5, .5), modifier=None)
            asyncio.run(interaction.handle_click(event))
            selected = events[-1]
            assert selected['view_id'] == ('qt' if index == 0 else None)
            assert selected['event']['object_id'] == 'moving' and selected['event']['frame'] == current
        assert interaction.pick(snapshots[0][2], [0., 1., 0.])['object_id'] is None
    finally:
        interaction.close()


def test_real_viewer_process_consumes_sidecar_routes_errors_and_filters_frames(tmp_path):
    from muse_runtime.preview import export_model
    from muse_sdk.run_protocol import PreviewFrame, encode_preview, preview_socketpair

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="probe" pos="0 0 1">
      <freejoint/><geom name="box" type="box" size=".2 .2 .2"/></body></worldbody></mujoco>''')
    obj = SceneObject(object_id='probe', label='Probe', kind='entity', group_id='probe', group_slot=0,
        role_id=None, model_id='probe', body_id=model.body('probe').id, geom_ids=[model.geom('box').id])
    metadata = export_model(model, tmp_path, objects=[obj])
    sender, receiver = preview_socketpair()
    command = [sys.executable, '-m', 'muse_server.viewers.viser_process', '--snapshot-fd', str(receiver.fileno()),
        '--model', str(tmp_path / metadata.path), '--objects', str(tmp_path / 'preview_objects.json'),
        '--sha256', metadata.sha256, '--nbody', str(model.nbody), '--run-id', 'run', '--world-count', '2', '--port', '0']
    with (tmp_path / 'viewer.log').open('wb') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=log, pass_fds=(receiver.fileno(),), text=True)
        receiver.close()
        messages = queue.Queue()
        def reader():
            try:
                with (tmp_path / 'events.jsonl').open('w', encoding='utf-8') as evidence:
                    for line in process.stdout:
                        evidence.write(line)
                        evidence.flush()
                        messages.put(json.loads(line))
            finally:
                messages.put({'kind': 'process_exit'})
        worker = threading.Thread(target=reader, daemon=True)
        worker.start()
        def send(message):
            process.stdin.write(json.dumps(message) + '\n')
            process.stdin.flush()
        def state(seq, world, ids):
            send({'kind': 'state', 'state': {'effective': {'enabled': True, 'world_id': world, 'rate_hz': 30.},
                'effective_seq': seq, 'status': 'ready', 'error': None}, 'view_ids': ids})
        def packet(seq, world, episode, decision, x):
            data = mujoco.MjData(model)
            data.qpos[:3] = [x, 0., 1.]
            mujoco.mj_forward(model, data)
            header = PreviewFrame(run_id='run', world_id=world, selection_seq=seq, episode=episode,
                decision=decision, tick=decision * 20, sim_time=decision * .04, scene_hash=metadata.sha256, nbody=model.nbody)
            arrays = data.xpos.astype('<f4').tobytes() + data.xmat.astype('<f4').tobytes()
            return header, encode_preview(header, arrays)
        try:
            loaded = messages.get(timeout=15)
            assert loaded['kind'] == 'loaded', loaded
            with urllib.request.urlopen(f"http://{loaded['host']}:{loaded['port']}", timeout=5) as response:
                assert response.status == 200
            state(2, 1, ['qt-a', 'qt-b'])
            for target, expected in [('unknown', None), ('qt-a', 'qt-a')]:
                send({'kind': 'view_intent', 'view_id': target, 'intent': {'kind': 'select', 'object_id': 'missing'}})
                event = messages.get(timeout=10)
                assert event['kind'] == 'view' and event['view_id'] == expected
                assert event['event']['kind'] == 'error'
            header, encoded = packet(2, 1, 2, 4, 2.)
            sender.send(encoded)
            displayed = messages.get(timeout=10)
            assert displayed == {'kind': 'frame', 'header': header.model_dump(mode='json')}
            state(3, 0, ['qt-b'])
            # 同一 stdin 的意图响应确认新 registry/state 已消费，再注入新选择的数据包。
            send({'kind': 'view_intent', 'view_id': 'qt-a', 'intent': {'kind': 'focus', 'object_id': 'probe'}})
            assert messages.get(timeout=10)['view_id'] is None
            current, encoded = packet(3, 0, 0, 2, 0.)
            sender.send(encoded)
            sender.send(packet(2, 1, 9, 999, 9.)[1])
            displayed = messages.get(timeout=10)
            assert displayed == {'kind': 'frame', 'header': current.model_dump(mode='json')}
            process.stdin.close()
            assert process.wait(timeout=10) == 0
        finally:
            sender.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            worker.join(timeout=5)
