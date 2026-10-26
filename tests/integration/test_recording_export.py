"""软件离屏出口；门控真实渲染，不创建GPU Run。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest


@pytest.mark.parametrize('fails', [False, True])
def test_export_child_only_announces_closed_final_file(tmp_path, fails):
    import hashlib
    staging = tmp_path / 'staging'
    staging.mkdir()
    camera = dict(position=[0., 0., 0.], wxyz=[1., 0., 0., 0.], fov=1., near=.1, far=100.)
    job = dict(staging_directory=str(staging), artifact_directory=str(tmp_path), directory='input',
        export_id='a' * 32, run_id='source', source_trajectory_id='source:trajectory:' + 'b' * 32,
        request=dict(world_id=0, episode=0, fps=25., width=640, height=480), camera=camera)
    path = tmp_path / 'job.json'
    path.write_text(json.dumps(job))
    # 仅替代昂贵编码边界；真实子进程、stdout隔离、文件关闭/rename/hash均由产品入口执行。
    script = '''
import sys
from muse_server.viewers import recording_process
failure = sys.argv[1] == 'fail'
def export(directory, request, camera, output, emit):
    print('encoder log must not enter protocol')
    output.write_bytes(b'closed-video')
    if failure:
        raise RuntimeError('encoder failed')
    return dict(world_id=0, episode=0, first_sample_index=0, last_sample_index=1,
        complete_episode=False, camera=vars(camera), fps=25., width=640, height=480, frame_count=2)
recording_process.export_video = export
sys.argv = [sys.argv[1], '--job', sys.argv[2]]
raise SystemExit(recording_process.main())
'''
    result = subprocess.run([sys.executable, '-c', script, 'fail' if fails else 'success', str(path)],
        capture_output=True, text=True, timeout=20)
    assert result.returncode == (1 if fails else 0), result.stderr
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    assert 'encoder log must not enter protocol' in result.stderr
    if fails:
        assert messages == [{'kind': 'error', 'message': 'encoder failed'}]
        assert not (tmp_path / 'recordings').exists()
    else:
        artifact = messages[-1]['artifact']
        final = tmp_path / artifact['path']
        assert final.read_bytes() == b'closed-video'
        assert not (staging / 'video.mp4').exists()
        assert artifact['sha256'] == hashlib.sha256(b'closed-video').hexdigest()
        assert artifact['metadata']['source_trajectory_id'] == job['source_trajectory_id']


def test_camera_snapshot_requires_paired_initialized_client_and_is_frozen():
    from types import SimpleNamespace
    import mujoco
    import viser
    from muse_server.viewers.scene_interaction import SceneInteraction
    model = mujoco.MjModel.from_xml_string('<mujoco/>')
    server = viser.ViserServer(port=0, verbose=False)
    events, callbacks = [], []
    camera = SimpleNamespace(position=np.array([1., 2., 3.]), wxyz=np.array([1., 0., 0., 0.]),
        fov=1., near=.1, far=100., on_update=callbacks.append)
    client = SimpleNamespace(client_id=1, camera=camera, scene=server.scene,
        local_storage=SimpleNamespace(get_item=lambda *_, **__: 'owned'))
    interaction = SceneInteraction(server, model, [], view_id='owned', emit=events.append, capture_camera=True)
    try:
        interaction.connect_client(client)
        with pytest.raises(ValueError, match='camera'):
            interaction.camera_snapshot('owned')
        callbacks[0](camera)
        snapshot = interaction.camera_snapshot('owned')
        assert snapshot.position == (1., 2., 3.)
        camera.position[:] = 10
        assert snapshot.position == (1., 2., 3.)
        with pytest.raises(ValueError, match='camera'):
            interaction.camera_snapshot('other')
        interaction.disconnect_client(client)
        with pytest.raises(ValueError, match='camera'):
            interaction.camera_snapshot('owned')
    finally:
        interaction.close()
        server.stop()


@pytest.mark.skipif(os.environ.get('MUSE_TEST_RECORDING_RENDER') != '1', reason='Requires software render acceptance')
def test_real_software_export_decodes_first_and_last_pose(recording_source, recorded_trajectory, tmp_path):
    plan, _ = recording_source
    directory = tmp_path / 'unpacked'
    with zipfile.ZipFile(Path(plan.artifact_directory) / recorded_trajectory.path) as archive:
        archive.extractall(directory)
    staging = tmp_path / 'staging'
    staging.mkdir()
    # 相机在(3,-12,5)看向+Y，OpenCV up=-Y映射世界+Z。
    job = dict(directory=str(directory), staging_directory=str(staging), artifact_directory=plan.artifact_directory,
        export_id='c' * 32, run_id=plan.run_id, source_trajectory_id=recorded_trajectory.id,
        request=dict(world_id=0, episode=0, fps=25., width=640, height=480),
        camera=dict(position=[3., -12., 5.], wxyz=[2**-.5, -2**-.5, 0., 0.], fov=1.2, near=.1, far=100.))
    job_path = tmp_path / 'job.json'
    job_path.write_text(json.dumps(job))
    result = subprocess.run([sys.executable, '-m', 'muse_server.viewers.recording_process', '--job', str(job_path)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    artifact = messages[-1]['artifact']
    metadata = artifact['metadata']
    output = Path(plan.artifact_directory) / artifact['path']
    assert not (staging / 'video.mp4').exists()
    assert messages[-2] == dict(kind='progress', frames_written=8, total_frames=8)
    assert metadata['frame_count'] == 8
    assert metadata['complete_episode'] is True
    import imageio_ffmpeg
    frames = imageio_ffmpeg.read_frames(str(output), pix_fmt='rgb24')
    try:
        decoded = next(frames)
        assert decoded['size'] == (640, 480)
        images = [np.frombuffer(frame, dtype=np.uint8).reshape(480, 640, 3) for frame in frames]
    finally:
        frames.close()
    assert len(images) == 8
    assert images[0].std() > 2
    assert np.count_nonzero(np.abs(images[0].astype(float) - images[-1]) > 20) > 50


@pytest.mark.skipif(os.environ.get('MUSE_TEST_RECORDING_RENDER') != '1', reason='Requires software render acceptance')
def test_real_mesh_projection_preserves_roll_and_excludes_unsourced_layers(recording_source, tmp_path):
    import hashlib
    import mujoco
    from muse_sdk.trajectory import TrajectoryWriter
    from recording_helpers import sample_values
    plan, reference = recording_source
    # 薄平面mesh使可见像素中心可用针孔公式独立预测；90°roll时+Y向右、+X向上。
    model = mujoco.MjModel.from_xml_string('''<mujoco><asset><mesh name="cube" vertex="
      -.2 -.2 -.001  -.2 -.2 .001  -.2 .2 -.001  -.2 .2 .001
       .2 -.2 -.001   .2 -.2 .001   .2 .2 -.001   .2 .2 .001"/></asset><worldbody>
      <geom type="plane" size="5 5 .1" rgba="0 0 0 0"/>
      <body><freejoint/><geom type="mesh" mesh="cube" pos=".1 0 0" rgba="1 0 0 1"/>
        <site size=".6" rgba="0 1 0 1"/><light diffuse="0 1 0"/></body>
      <body><freejoint/><geom size=".1" rgba="0 0 0 0"/></body>
      <body><freejoint/><geom size=".1" rgba="0 0 0 0"/></body>
      </worldbody></mujoco>''')
    path = Path(plan.artifact_directory) / reference.path
    mujoco.mj_saveModel(model, filename=str(path))
    reference.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    plan.recording.world_ids = [0]
    writer = TrajectoryWriter(plan, reference)
    try:
        for i in range(2):
            values = sample_values(writer.fields, step=i)
            values[0][:] = 0
            values[0][1] = [.5 + .5 * i, .25, 3.]
            values[1][1] = [0., -1., 0., 1., 0., 0., 0., 0., 1.]
            writer.append(0, (i, i, 0, i, 20 * i, i, 0), values)
        artifact = writer.seal('completed', 1)
    finally:
        writer.close()
    directory = tmp_path / 'mesh-input'
    with zipfile.ZipFile(Path(plan.artifact_directory) / artifact.path) as archive:
        archive.extractall(directory)
    output = tmp_path / 'mesh.mp4'
    script = '''
import json, math, sys
from pathlib import Path
from types import SimpleNamespace
import mujoco
from muse_server.viewers.recording_process import export_video
original = mujoco.Renderer.render
def inspect(self, *args, **kwargs):
    assert self.scene.nlight == 1 and self.scene.lights[0].headlight
    assert all(g.objtype == mujoco.mjtObj.mjOBJ_GEOM for g in self.scene.geoms[:self.scene.ngeom])
    assert any(g.type == mujoco.mjtGeom.mjGEOM_MESH for g in self.scene.geoms[:self.scene.ngeom])
    return original(self, *args, **kwargs)
mujoco.Renderer.render = inspect
request = SimpleNamespace(world_id=0, episode=0, fps=25., width=640, height=480)
camera = SimpleNamespace(position=(0.,0.,0.), wxyz=(2**-.5,0.,0.,2**-.5), fov=math.pi/2, near=.1, far=100.)
print(json.dumps(export_video(Path(sys.argv[1]), request, camera, Path(sys.argv[2]), lambda _: None)))
'''
    result = subprocess.run([sys.executable, '-c', script, str(directory), str(output)],
        env={**os.environ, 'MUJOCO_GL': 'glfw', 'LIBGL_ALWAYS_SOFTWARE': 'true'},
        capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    import imageio_ffmpeg
    decoder = imageio_ffmpeg.read_frames(str(output), pix_fmt='rgb24')
    try:
        next(decoder)
        images = [np.frombuffer(frame, dtype=np.uint8).reshape(480, 640, 3).astype(np.int16) for frame in decoder]
    finally:
        decoder.close()
    assert len(images) == 2
    for image, expected in zip(images, [(348., 200.), (348., 160.)]):
        red = (image[:, :, 0] > 50) & (image[:, :, 0] > image[:, :, 1] + 30) & (image[:, :, 0] > image[:, :, 2] + 30)
        ys, xs = np.nonzero(red)
        assert len(xs) > 100
        np.testing.assert_allclose([xs.mean(), ys.mean()], expected, atol=3.)
