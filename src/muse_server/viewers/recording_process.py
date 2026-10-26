"""独立软件离屏与逐帧编码；只消费已发布历史，不推进物理。"""
import math
from pathlib import Path

import numpy as np


def _frame_count(times, fps):
    # tick→秒→帧的乘法可在整数边界向上舍入一个ULP；不额外生成重复末帧。
    span = math.nextafter(float(times[-1] - times[0]) * fps, -math.inf)
    return max(1, math.ceil(span) + 1)


def video_sample_indices(times, fps):
    """已验证episode内时间；零阶保持，最终帧包含最后有效样本。"""
    relative = times - times[0]
    frames = _frame_count(times, fps)
    for frame in range(frames):
        yield len(times) - 1 if frame == frames - 1 else int(np.searchsorted(relative, frame / fps, side='right') - 1)


def apply_camera(scene, camera):
    import mujoco
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(camera.wxyz))
    rotation = rotation.reshape(3, 3)
    forward, up = rotation[:, 2], -rotation[:, 1]
    for eye in scene.camera:
        eye.pos[:] = camera.position
        eye.forward[:] = forward
        eye.up[:] = up
        eye.frustum_near, eye.frustum_far = camera.near, camera.far
        eye.frustum_top = camera.near * math.tan(camera.fov / 2)
        eye.frustum_bottom = -eye.frustum_top
        eye.frustum_center = 0
        eye.orthographic = 0
    for light in scene.lights[:scene.nlight]:
        if light.headlight:
            light.pos[:] = camera.position
            light.dir[:] = forward


def export_video(directory, request, camera, output, emit):
    import imageio_ffmpeg
    import mediapy
    import mujoco
    from muse_sdk.trajectory import TrajectoryReader
    from .scene_data import apply_body_poses, geom_rotations

    reader = TrajectoryReader(directory)
    try:
        samples = reader.episode_samples(request.world_id, request.episode)
        times = reader.arrays[request.world_id]['index'][samples, 4] * reader.manifest.physics_dt
        count = _frame_count(times, request.fps)
        model = mujoco.MjModel.from_binary_path(str(Path(directory) / 'model.mjb'))
        if model.nbody != reader.manifest.nbody:
            raise ValueError('Trajectory model nbody mismatch')
        model.vis.global_.offwidth, model.vis.global_.offheight = request.width, request.height
        model.light_active[:] = False
        model.vis.headlight.active = 1
        data = mujoco.MjData(model)
        rotations = geom_rotations(model)
        options = mujoco.MjvOption()
        options.flags[:] = False
        options.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = True
        options.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = True
        options.sitegroup[:] = 0
        options.flexgroup[:] = 0
        options.skingroup[:] = 0
        options.tendongroup[:] = 0
        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        with mujoco.Renderer(model, height=request.height, width=request.width,
                             max_geom=max(1, model.ngeom)) as renderer:
            image = np.empty((request.height, request.width, 3), dtype=np.uint8)
            with mediapy.VideoWriter(output, shape=(request.height, request.width), fps=request.fps,
                                     codec='libx264', encoded_format='yuv420p') as writer:
                for written, index in enumerate(video_sample_indices(times, request.fps), 1):
                    xpos, xmat, _ = reader.frame(request.world_id, int(samples[index]))
                    apply_body_poses(model, data, xpos, xmat, rotations)
                    renderer.update_scene(data, scene_option=options)
                    apply_camera(renderer.scene, camera)
                    renderer.render(out=image)
                    writer.add_image(image)
                    emit({'kind': 'progress', 'frames_written': written, 'total_frames': count})
        episode = next(item for item in reader.manifest.episodes[str(request.world_id)]
                       if item['episode'] == request.episode)
        return dict(world_id=request.world_id, episode=request.episode,
            first_sample_index=int(samples[0]), last_sample_index=int(samples[-1]),
            complete_episode=episode['complete'], camera=dict(position=list(camera.position),
                wxyz=list(camera.wxyz), fov=camera.fov, near=camera.near, far=camera.far),
            fps=request.fps, width=request.width, height=request.height, frame_count=count)
    finally:
        reader.close()


def main():
    import argparse
    import hashlib
    import json
    import os
    import sys
    import traceback
    from types import SimpleNamespace
    from muse_sdk.recording_protocol import RecordingArtifact

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    job = json.loads(parser.parse_args().job.read_text())
    os.environ.update(MUJOCO_GL='glfw', LIBGL_ALWAYS_SOFTWARE='true')
    sys.stdout.flush()
    protocol_fd = os.dup(1)
    os.dup2(2, 1)

    def emit(message):
        packet = (json.dumps(message, allow_nan=False) + '\n').encode()
        while packet:
            packet = packet[os.write(protocol_fd, packet):]

    try:
        output = Path(job['staging_directory']) / 'video.mp4'
        # 输入由manager冻结；child信任内部契约，格式化输出时按artifact DTO序列化。
        metadata = export_video(Path(job['directory']), SimpleNamespace(**job['request']),
            SimpleNamespace(**job['camera']), output, emit)
        with output.open('rb') as stream:
            os.fsync(stream.fileno())
        final = Path(job['artifact_directory']) / 'recordings' / (job['export_id'] + '.mp4')
        final.parent.mkdir(exist_ok=True)
        os.replace(output, final)
        with final.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        artifact = RecordingArtifact.model_validate_json(json.dumps(dict(
            id=f"{job['run_id']}:recording:{job['export_id']}", run=job['run_id'], kind='recording',
            path=final.relative_to(job['artifact_directory']).as_posix(), size=final.stat().st_size,
            sha256=digest, metadata={'source_trajectory_id': job['source_trajectory_id'], **metadata})))
        emit({'kind': 'artifact', 'artifact': artifact.model_dump(mode='json')})
        return 0
    except Exception as exc:
        traceback.print_exc()
        emit({'kind': 'error', 'message': str(exc)})
        return 1
    finally:
        os.close(protocol_fd)


if __name__ == '__main__':
    raise SystemExit(main())
