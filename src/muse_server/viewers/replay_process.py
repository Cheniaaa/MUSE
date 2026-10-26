"""独立CPU历史播放器；不执行策略、物理或Runtime代码。"""
import argparse
import json
import os
from pathlib import Path
import select
import sys
import threading
import time
import traceback

import numpy as np

from muse_sdk.replay_protocol import ReplayCommand, ReplayRequest, ReplayState


class ReplayTimeline:
    """权威游标与单调修订号；Run时钟不随episode reset归零。"""

    def __init__(self, reader):
        self.reader = reader
        self.world_id = reader.manifest.recording.world_ids[0]
        self.sample_index = 0
        self.playing, self.speed = False, 1.
        self.revision, self.last_command_seq = 1, 0
        self._frame = reader.frame(self.world_id, self.sample_index)
        self._anchor_time, self._anchor_run = 0., 0.
        self._step_dt = reader.manifest.physics_dt * reader.manifest.decimation

    def frame(self):
        return self._frame

    def apply(self, command: ReplayCommand, now: float) -> dict:
        if command.command_seq <= self.last_command_seq:
            return dict(command_seq=command.command_seq, status='error',
                        error='command_seq must increase', revision=self.revision)
        self.last_command_seq = command.command_seq
        error = None
        try:
            world, sample = self.world_id, self.sample_index
            kind = command.kind
            if kind == 'world':
                world, sample = command.world_id, 0
            elif kind == 'episode':
                sample = int(self.reader.episode_samples(world, command.episode)[0])
            elif kind == 'seek':
                sample = command.sample_index
            elif kind == 'step':
                sample = min(max(sample + command.delta, 0), len(self.reader.arrays[world]['index']) - 1)
            if kind in ('world', 'episode', 'seek', 'step'):
                # reader是范围校验的唯一归属；成功前不改变游标或播放状态。
                frame = self.reader.frame(world, sample)
                self.world_id, self.sample_index, self._frame = world, sample, frame
                self._anchor_time, self._anchor_run = now, float(frame[2]['run_decision'])
                if kind in ('world', 'episode', 'step') or sample == len(self.reader.arrays[world]['index']) - 1:
                    self.playing = False
            elif kind == 'play':
                self._anchor_time, self._anchor_run = now, float(self._frame[2]['run_decision'])
                self.playing = self.sample_index < len(self.reader.arrays[world]['index']) - 1
            elif kind == 'pause':
                self.playing = False
            elif kind == 'speed':
                if self.playing:
                    self._anchor_run += (now - self._anchor_time) * self.speed / self._step_dt
                    self._anchor_time = now
                self.speed = command.speed
        except ValueError as exc:
            error = str(exc)
        self.revision += 1
        return dict(command_seq=command.command_seq, status='error' if error else 'applied',
                    error=error, revision=self.revision)

    def advance(self, now: float) -> bool:
        if not self.playing:
            return False
        index = self.reader.arrays[self.world_id]['index']
        due = self._anchor_run + (now - self._anchor_time) * self.speed / self._step_dt
        sample = max(self.sample_index, int(np.searchsorted(index[:, 1], due, side='right')) - 1)
        changed = sample != self.sample_index
        if changed:
            self.sample_index = sample
            self._frame = self.reader.frame(self.world_id, sample)
        if sample == len(index) - 1:
            self.playing = False
            changed = True
        if changed:
            self.revision += 1
        return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view-id', required=True)
    args = parser.parse_args(argv)
    sys.stdout.flush()
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    protocol_lock = threading.Lock()

    def emit(message):
        data = (json.dumps(message, allow_nan=False) + '\n').encode()
        with protocol_lock:
            while data:
                data = data[os.write(protocol_fd, data):]

    reader = server = interaction = None
    try:
        # 引用验证、解包、图形import均属于child准备，不阻塞HTTP或GPU调度。
        from muse_sdk.recording_protocol import TrajectoryArtifact
        from muse_sdk.trajectory import TrajectoryReader
        from muse_sdk.view_protocol import ViewIntent
        from muse_server.storage import extract_archive, verify_recorded_file
        message = json.loads(sys.stdin.readline())
        artifact = TrajectoryArtifact.model_validate(message['artifact'])
        archive = verify_recorded_file(artifact, artifact.run, Path(message['run_directory']))
        directory = Path(message['directory']) / 'trajectory'
        extract_archive(archive, directory)
        reader = TrajectoryReader(directory)
        if reader.manifest.run_id != artifact.run:
            raise ValueError('Trajectory source Run does not match registered artifact')
        from .scene_data import load_model
        from .scene_interaction import SceneInteraction
        import viser
        model = load_model(directory / 'model.mjb', reader.manifest.model_sha256, reader.manifest.nbody)
        server = viser.ViserServer(host='127.0.0.1', port=0, verbose=False)
        interaction = SceneInteraction(server, model, reader.objects, view_id=args.view_id, emit=emit)
        timeline = ReplayTimeline(reader)
        objects = [obj.model_dump(mode='json', exclude={'body_id', 'geom_ids', 'local_position',
                                                       'local_orientation_matrix'}) for obj in reader.objects]

        def publish():
            xpos, xmat, diagnostics = timeline.frame()
            interaction.update_frame(xpos, xmat, diagnostics)
            state = ReplayState(session_id=args.view_id, status='ready',
                source=ReplayRequest(run_id=artifact.run, trajectory_artifact_id=artifact.id),
                viewer_url=f'http://127.0.0.1:{server.get_port()}', revision=timeline.revision,
                last_command_seq=timeline.last_command_seq, playing=timeline.playing, speed=timeline.speed,
                cursor={name: diagnostics[name] for name in ('world_id', 'sample_index', 'episode')},
                worlds=reader.manifest.summary['worlds'], episodes=reader.manifest.episodes,
                objects=objects, diagnostics=diagnostics)
            emit({'kind': 'state', 'state': state.model_dump(mode='json')})

        publish()
        incoming = b''
        while True:
            if select.select([sys.stdin], [], [], .01)[0]:
                chunk = os.read(0, 65536)
                if not chunk:
                    return 0
                incoming += chunk
                while b'\n' in incoming:
                    line, incoming = incoming.split(b'\n', 1)
                    message = json.loads(line)
                    if message['kind'] == 'close':
                        return 0
                    if message['kind'] == 'playback':
                        result = timeline.apply(ReplayCommand.model_validate(message['command']), time.monotonic())
                        publish()
                        emit({'kind': 'command_result', **result})
                    elif message['kind'] == 'view':
                        try:
                            interaction.apply_intent(ViewIntent.model_validate(message['intent']))
                        except ValueError as exc:
                            emit({'kind': 'view', 'event': {'kind': 'error', 'message': str(exc)}})
                    else:
                        raise ValueError('Unknown replay control: ' + message['kind'])
            if timeline.advance(time.monotonic()):
                publish()
    except Exception as exc:
        # 进程边界保留根因并失败退出；不降级或继续播放损坏的数据。
        traceback.print_exc()
        emit({'kind': 'error', 'message': str(exc)})
        return 1
    finally:
        if interaction is not None:
            interaction.close()
        if server is not None:
            server.stop()
        if reader is not None:
            reader.close()
        with protocol_lock:
            os.close(protocol_fd)


if __name__ == '__main__':
    raise SystemExit(main())
