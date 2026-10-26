"""只验证进程/发布生命周期的fake编码child，不生成或声称有效视频。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from muse_sdk.recording_protocol import RecordingArtifact, RecordingMetadata
from muse_sdk.trajectory import TrajectoryReader

parser = argparse.ArgumentParser()
parser.add_argument('--job', type=Path, required=True)
parser.add_argument('--mode', required=True)
args = parser.parse_args()
job = json.loads(args.job.read_text())
staging = Path(job['staging_directory'])
(staging / 'encoder.pid').write_text(str(os.getpid()))
(staging / 'video.mp4').write_bytes(b'partial fake encoding')
print(json.dumps(dict(kind='progress', frames_written=1, total_frames=3)), flush=True)
if args.mode == 'error':
    print(json.dumps(dict(kind='error', message='fake encoder root cause')), flush=True)
    raise SystemExit(3)
if args.mode in ('hang', 'orphan'):
    subprocess.Popen([sys.executable, '-c',
        'import os,signal,sys,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)',
        str(staging / 'descendant.pid')])
    while not (staging / 'descendant.pid').exists():
        time.sleep(.005)
if args.mode == 'hang':
    time.sleep(60)
reader = TrajectoryReader(Path(job['directory']))
try:
    request = job['request']
    episode = next(item for item in reader.manifest.episodes[str(request['world_id'])]
                   if item['episode'] == request['episode'])
    metadata = RecordingMetadata.model_validate_json(json.dumps(dict(
        source_trajectory_id=job['source_trajectory_id'], **request, camera=job['camera'],
        first_sample_index=episode['first_sample_index'], last_sample_index=episode['last_sample_index'],
        complete_episode=episode['complete'], frame_count=3)))
finally:
    reader.close()
final = Path(job['artifact_directory']) / 'recordings' / (job['export_id'] + '.mp4')
final.parent.mkdir(exist_ok=True)
final.write_bytes(b'CPU lifecycle fixture, not an actual MP4')
artifact = RecordingArtifact(id=f"{job['run_id']}:recording:{job['export_id']}", run=job['run_id'],
    path=final.relative_to(job['artifact_directory']).as_posix(), size=final.stat().st_size,
    sha256=hashlib.sha256(final.read_bytes()).hexdigest(), metadata=metadata)
if args.mode != 'missing':
    print(json.dumps(dict(kind='artifact', artifact=artifact.model_dump(mode='json'))), flush=True)
if args.mode == 'artifact_hang':
    time.sleep(60)
