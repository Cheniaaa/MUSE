"""真实 CPU/Viser/child 协议；client 适配器只测控制边界，不冒充浏览器。"""
import asyncio
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
from types import SimpleNamespace
import urllib.error
import urllib.request

import mujoco
import numpy as np
import pytest
import viser

from muse_sdk.content import hash_package
from muse_sdk.experiment import ResourceRef
from muse_sdk.view_protocol import SceneObject, ViewIntent


@pytest.fixture
def native_scene():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="moving" pos="0 0 .5"><freejoint/>
        <geom name="box" type="box" size=".2 .1 .15" pos=".1 0 0"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    snapshots = []
    for tick, x in enumerate((0., 2.)):
        data.qpos[:3] = [x, 0., .5]
        data.qpos[3:7] = [np.cos(.2), 0., 0., np.sin(.2)]
        mujoco.mj_forward(model, data)
        hit = np.array([-1], dtype=np.int32)
        origin = np.array([x + .1, -2., .5])
        distance = mujoco.mj_ray(model, data, origin, np.array([0., 1., 0.]), None, True, -1, hit)
        snapshots.append((data.xpos.copy(), data.xmat.reshape(-1, 3, 3).copy(), origin, distance))
    objects = [SceneObject(object_id='moving', label='Moving', kind='entity', group_id='g',
        group_slot=0, role_id=None, model_id='probe', body_id=model.body('moving').id,
        geom_ids=[model.geom('box').id])]
    return model, objects, snapshots


@pytest.fixture
def servers():
    created = []
    def create():
        server = viser.ViserServer(host='127.0.0.1', port=0, verbose=False)
        created.append(server)
        return server
    yield create
    for server in created:
        server.stop()


class LocalStorageAnswer:
    """边界适配器；不建立 browser/WS，不作为真实配对验收。"""
    def __init__(self, value):
        self.value = value
        self.calls = []

    def get_item(self, key, timeout):
        self.calls.append((key, timeout))
        return self.value


def client_adapter(server, client_id, view_id):
    # scene 与 camera 使用真实 Viser API；仅网络/localStorage 身份用适配器。
    return SimpleNamespace(client_id=client_id, scene=server.scene,
        camera=server.initial_camera, local_storage=LocalStorageAnswer(view_id))


def test_native_pick_and_render_follow_one_dynamic_snapshot_without_forward(native_scene, servers, monkeypatch):
    from muse_server.viewers.scene_interaction import SceneInteraction

    model, objects, snapshots = native_scene
    server = servers()
    interaction = SceneInteraction(server, model, objects, view_id='view', emit=lambda _: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('Dynamic snapshot must not mj_forward initial qpos')
    monkeypatch.setattr(mujoco, 'mj_forward', forbidden)
    for tick, (xpos, xmat, origin, distance) in enumerate(snapshots):
        interaction.update_frame(xpos.astype('f4'), xmat.astype('f4'), {'tick': tick})
        hit = interaction.pick(origin, [0., 1., 0.])
        assert hit['object_id'] == 'moving' and hit['frame'] == {'tick': tick}
        assert hit['distance'] == pytest.approx(distance, abs=1e-6)
        rendered = server.scene.get_handle_by_name('/muse/objects/0/geoms/0')
        np.testing.assert_allclose(rendered.position, xpos[objects[0].body_id], atol=1e-6)
    assert interaction.pick(snapshots[0][2], [0., 1., 0.])['object_id'] is None
    interaction.close()
    assert server.scene.get_handle_by_name('/muse') is None


def test_pairing_selection_visibility_and_camera_are_client_private(native_scene, servers):
    from muse_server.viewers.scene_interaction import SceneInteraction

    model, objects, snapshots = native_scene
    model.geom_group[:] = 4
    events = []
    interaction = SceneInteraction(servers(), model, objects, view_id='owned', emit=events.append,
                                    revision=11, editable={'moving'})
    xpos, xmat, origin, _ = snapshots[0]
    interaction.update_frame(xpos, xmat, {'revision': 11})
    a, b, ordinary = [client_adapter(servers(), index, token)
                      for index, token in [(1, 'owned'), (2, 'owned'), (3, None)]]
    for client in (a, b, ordinary):
        interaction.connect_client(client)
        assert client.local_storage.calls == [('muse_view_id', 5.0)]
    assert [event['event']['paired'] for event in events if event['kind'] == 'view'] == [True, True, False]
    interaction.apply_intent(ViewIntent(kind='select', object_id='moving'), client_id=1)
    assert interaction._clients[1].selected == 'moving'
    assert interaction._clients[2].selected is None
    interaction.apply_intent(ViewIntent(kind='select', object_id='moving'), client_id=3)
    assert interaction._clients[3].selected is None and interaction._clients[3].gizmo is None
    interaction.apply_intent(ViewIntent(kind='visibility', object_id='moving', visible=False), client_id=1)
    assert not a.scene.get_handle_by_name('/muse/objects/0').visible
    assert b.scene.get_handle_by_name('/muse/objects/0').visible
    assert interaction.pick(origin, [0., 1., 0.], client_id=1)['object_id'] is None
    assert interaction.pick(origin, [0., 1., 0.], client_id=2)['object_id'] == 'moving'
    np.testing.assert_array_equal(model.geom_group, [4])
    before = b.camera.position.copy()
    interaction.apply_intent(ViewIntent(kind='focus', object_id='moving'), client_id=1)
    np.testing.assert_array_equal(b.camera.position, before)
    interaction.close()
    assert a.scene.get_handle_by_name('/muse/gizmo') is None


def test_gizmo_phases_have_one_id_and_never_commit_geometry(native_scene, servers):
    from muse_server.viewers.scene_interaction import SceneInteraction

    model, objects, snapshots = native_scene
    events, original = [], model.qpos0.copy()
    interaction = SceneInteraction(servers(), model, objects, view_id='view', emit=events.append,
                                    revision=4, editable={'moving'})
    xpos, xmat, _, _ = snapshots[0]
    interaction.update_frame(xpos, xmat, {'revision': 4})
    client = client_adapter(servers(), 1, 'view')
    interaction.connect_client(client)
    peer = client_adapter(servers(), 2, 'view')
    interaction.connect_client(peer)
    interaction.apply_intent(ViewIntent(kind='select', object_id='moving'), client_id=1)
    gizmo = interaction._clients[1].gizmo
    for phase in ('start', 'update', 'end'):
        if phase == 'update':
            gizmo.position = (3., 0., .5)
        event = viser.TransformControlsEvent(client=client, client_id=1, target=gizmo, phase=phase)
        asyncio.run(interaction.handle_transform(event))
    gestures = [event['gesture'] for event in events if event['kind'] == 'gesture']
    assert [gesture['phase'] for gesture in gestures] == ['start', 'update', 'end']
    assert len({gesture['gesture_id'] for gesture in gestures}) == 1
    assert all(gesture['base_revision'] == 4 and gesture['object_id'] == 'moving' for gesture in gestures)
    assert gestures[-1]['pose']['position'] == [3., 0., .5]
    assert interaction.pick([3.1, -2., .5], [0., 1., 0.], client_id=1)['object_id'] == 'moving'
    assert interaction.pick([3.1, -2., .5], [0., 1., 0.], client_id=2)['object_id'] is None
    np.testing.assert_array_equal(peer.scene.get_handle_by_name('/muse/objects/0').position, [0., 0., 0.])
    np.testing.assert_array_equal(model.qpos0, original)
    interaction.close()


@pytest.mark.parametrize('revision', [None, 9])
def test_run_or_readonly_interaction_never_creates_draft_gizmo(native_scene, servers, revision):
    from muse_server.viewers.scene_interaction import SceneInteraction

    model, objects, snapshots = native_scene
    interaction = SceneInteraction(servers(), model, objects, view_id='run', emit=lambda _: None,
                                    revision=revision)
    interaction.update_frame(*snapshots[0][:2], {'tick': 0})
    client = client_adapter(servers(), 1, 'run')
    interaction.connect_client(client)
    interaction.apply_intent(ViewIntent(kind='select', object_id='moving'), client_id=1)
    assert interaction._clients[1].gizmo is None
    interaction.close()


def test_concurrent_pick_never_observes_a_mixed_render_frame(native_scene, servers):
    from muse_server.viewers.scene_interaction import SceneInteraction

    model, objects, snapshots = native_scene
    interaction = SceneInteraction(servers(), model, objects, view_id='run', emit=lambda _: None)
    interaction.update_frame(*snapshots[0][:2], {'tick': 0})
    barrier = threading.Barrier(2)
    def update():
        barrier.wait()
        for step in range(40):
            tick = step % 2
            interaction.update_frame(*snapshots[tick][:2], {'tick': tick})
    worker = threading.Thread(target=update)
    worker.start()
    barrier.wait()
    try:
        for _ in range(80):
            hit = interaction.pick(snapshots[0][2], [0., 1., 0.])
            assert hit['object_id'] == ('moving' if hit['frame']['tick'] == 0 else None)
    finally:
        worker.join(timeout=10)
        assert not worker.is_alive()
        interaction.close()


def test_client_connected_before_first_live_frame_focuses_that_frame(native_scene, servers):
    from muse_server.viewers.scene_interaction import SceneInteraction

    model, objects, snapshots = native_scene
    interaction = SceneInteraction(servers(), model, objects, view_id='run', emit=lambda _: None)
    client = client_adapter(servers(), 1, 'run')
    interaction.connect_client(client)
    xpos, xmat, _, _ = snapshots[1]
    interaction.update_frame(xpos, xmat, {'tick': 42})
    expected = xpos[objects[0].body_id] + xmat[objects[0].body_id] @ model.geom_pos[0]
    np.testing.assert_allclose(client.camera.look_at, expected, atol=1e-6)
    interaction.close()


def test_local_storage_failure_is_reported_without_edit_authority(native_scene, servers):
    from muse_server.viewers.scene_interaction import SceneInteraction

    class UnavailableStorage:
        def get_item(self, key, timeout):
            raise TimeoutError('localStorage did not answer')

    model, objects, snapshots = native_scene
    events = []
    interaction = SceneInteraction(servers(), model, objects, view_id='view', emit=events.append,
                                    revision=1, editable={'moving'})
    interaction.update_frame(*snapshots[0][:2], {'revision': 1})
    client = client_adapter(servers(), 1, 'view')
    client.local_storage = UnavailableStorage()
    interaction.connect_client(client)
    assert events[-1]['event']['paired'] is False
    assert events[-1]['event']['message'] == 'localStorage did not answer'
    interaction.apply_intent(ViewIntent(kind='select', object_id='moving'), client_id=1)
    assert interaction._clients[1].gizmo is None
    interaction.close()


def test_disconnect_during_pairing_does_not_resurrect_client_state(native_scene, servers):
    from muse_server.viewers.scene_interaction import SceneInteraction

    model, objects, _ = native_scene
    events = []
    interaction = SceneInteraction(servers(), model, objects, view_id='view', emit=events.append)
    client = client_adapter(servers(), 1, 'view')
    class DisconnectingStorage:
        def get_item(self, key, timeout):
            interaction.disconnect_client(client)
            raise RuntimeError('Client disconnected before localStorage response')
    client.local_storage = DisconnectingStorage()
    interaction.connect_client(client)
    assert 1 not in interaction._clients
    assert events[-1]['event']['connected'] is False
    interaction.close()


@pytest.mark.parametrize('exit_kind', ['close', 'eof'])
def test_child_protocol_recompile_error_preserves_old_view_and_shutdown(tmp_path, exit_kind):
    root = tmp_path / 'base'
    root.mkdir()
    (root / 'environment.yaml').write_text('scene: scene.xml\n')
    (root / 'scene.xml').write_text('<mujoco><worldbody><geom name="wall" type="box" size="1 1 1"/></worldbody></mujoco>')
    reference = ResourceRef(resource_id='env', revision=1, sha256=hash_package(root))
    request = dict(draft_id='draft', revision=1, environment=reference.model_dump(mode='json'),
        environment_scene='scene.xml', environment_edit=None, entities=[])
    script = '''import runpy, sys, threading
def guard(frame, event, arg):
    if event != 'call': return
    path, name = frame.f_code.co_filename.replace(chr(92), '/'), frame.f_code.co_name
    assert not ((path.endswith('/mjlab/sim/sim.py') and name == '__init__') or
        (path.endswith('/mjlab/scene/scene.py') and name in ('initialize', 'reset')) or
        path.endswith('/muse_runtime/env.py') or
        (path.endswith('/torch/cuda/__init__.py') and name in ('init', '_lazy_init')) or
        ('/warp/' in path and name == 'init')), (path, name)
sys.setprofile(guard)
threading.setprofile(guard)
try: runpy.run_module('muse_runtime.draft_session', run_name='__main__')
except SystemExit as exc: assert exc.code == 0
import torch, warp._src.context as context
assert not torch.cuda.is_initialized() and context.runtime is None
import ctypes
driver = ctypes.CDLL('libcuda.so.1')
driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
driver.cuCtxGetCurrent.restype = ctypes.c_int
current = ctypes.c_void_p()
assert driver.cuCtxGetCurrent(ctypes.byref(current)) == 3 and current.value is None
assert not {'muse_runtime.env', 'muse_runtime.package_loader'} & sys.modules.keys()
print('CPU_SESSION_AUDIT_OK', file=sys.stderr)
'''
    stderr_path = tmp_path / 'child.log'
    with stderr_path.open('w+') as stderr:
        child = subprocess.Popen([sys.executable, '-c', script, '--view-id', 'session'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True)
        messages = queue.Queue()
        def reader():
            try:
                with (tmp_path / 'events.jsonl').open('w', encoding='utf-8') as evidence:
                    for line in child.stdout:
                        evidence.write(line)
                        evidence.flush()
                        messages.put(json.loads(line))
            finally:
                messages.put({'kind': 'process_exit'})
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        def send(message):
            child.stdin.write(json.dumps(message) + '\n')
            child.stdin.flush()
        def compile_revision(revision, *, bad=False):
            send(dict(kind='compile', request={**request, 'revision': revision,
                'environment_scene': 'missing.xml' if bad else 'scene.xml'},
                resource_paths={'environment': str(root)}, destination=str(tmp_path / f'rev-{revision}')))
            return messages.get(timeout=40)
        try:
            first = compile_revision(1)
            assert first['kind'] == 'ready' and first['revision'] == 1
            assert first['objects'][0]['object_id'] == 'wall'
            with urllib.request.urlopen(first['viewer_url'], timeout=5) as response:
                assert response.status == 200
            send({'kind': 'intent', 'intent': {'kind': 'select', 'object_id': 'unknown'}})
            invalid_intent = messages.get(timeout=5)
            assert invalid_intent['kind'] == 'view' and invalid_intent['event']['kind'] == 'error'
            failure = compile_revision(2, bad=True)
            assert failure['kind'] == 'error' and failure['revision'] == 2 and failure['message']
            with urllib.request.urlopen(first['viewer_url'], timeout=5) as response:
                assert response.status == 200
            final = compile_revision(3)
            assert final['kind'] == 'ready' and final['revision'] == 3
            assert final['viewer_url'] != first['viewer_url']
            with pytest.raises(urllib.error.URLError):
                urllib.request.urlopen(first['viewer_url'], timeout=2)
            send({'kind': 'intent', 'intent': {'kind': 'reset_camera'}})
            if exit_kind == 'close':
                send({'kind': 'close'})
            child.stdin.close()
            assert child.wait(timeout=20) == 0
            thread.join(timeout=5)
            with pytest.raises(urllib.error.URLError):
                urllib.request.urlopen(final['viewer_url'], timeout=2)
            stderr.seek(0)
            assert 'CPU_SESSION_AUDIT_OK' in stderr.read()
            assert hash_package(root) == reference.sha256
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
            thread.join(timeout=5)
