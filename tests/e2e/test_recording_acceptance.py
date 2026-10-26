"""M5.2 三个有界工程 case；默认只 skip，不创建目录、Server 或 Run。

桌面消费者只连接 observer.json 中已有的 Run，不得 POST /runs。
ack 协议及人工证据要求见 m5-recording-task7-preflight-report.md。
"""
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import traceback

import httpx2
import psutil
import pytest

from acceptance_helpers import confirmed_display_error
from test_basic_flight import archive, response_json, TERMINAL


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / '.planning/mjlab-spike/m5-recording-acceptance'
CASES = ('train', 'evaluate', 'interactive')
BUDGETS = {'train': ('TRAIN', 128, 8, 600), 'evaluate': ('EVALUATE', 64, 0, 600),
           'interactive': ('INTERACTIVE', 32, 0, 300)}


def case_authorized(case, environment):
    selected = environment.get('MUSE_RECORDING_CASE')
    if selected is not None and selected not in CASES:
        raise pytest.UsageError('MUSE_RECORDING_CASE must select exactly one of train/evaluate/interactive')
    return (environment.get('MUSE_TEST_GPU') == '1' and selected == case
            and environment.get('MUSE_TEST_RECORDING_RENDER') == '1')


def reserve_attempt(directory, case, record):
    """每 case 只有一次已批准 attempt；失败也不得自行重试。"""
    attempts = directory / 'attempts'
    attempts.mkdir(parents=True, exist_ok=True)
    payload = {**record, 'case': case, 'attempt': 1}
    with (attempts / f'{case}.json').open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    with (directory / 'attempts.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_observer_ack(ack, source):
    identity = ('case', 'stage', 'catalog_id', 'instance_id', 'run_id')
    assert {key: ack[key] for key in identity} == {key: source[key] for key in identity}, 'Observer ack identity mismatch'
    assert ack['success'] is True, ack


def _write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _wait(predicate, deadline, message):
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.05)
    raise AssertionError(message)


def _resources(server):
    parent = psutil.Process(server.pid)
    processes = []
    for process in (parent, *parent.children(recursive=True)):
        try:
            processes.append(dict(pid=process.pid, create_time=process.create_time(), command=process.cmdline(),
                cpu_seconds=sum(process.cpu_times()[:2]), rss_bytes=process.memory_info().rss))
        except psutil.NoSuchProcess:
            processes.append(dict(pid=process.pid, exited_during_sample=True))
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
    return dict(processes=processes, gpu_query=dict(stdout=gpu.stdout, stderr=gpu.stderr, exit_code=gpu.returncode))


def _register_experiment(client):
    example = ROOT / 'examples/basic_flight'
    environment = response_json(client.post('/catalog/environments', content=archive(example / 'environment'),
        headers={'Content-Type': 'application/zip'}), 201)
    package = response_json(client.post('/catalog/training-packages', content=archive(example / 'training_package_ippo'),
        headers={'Content-Type': 'application/zip'}), 201)
    request = json.loads((example / 'request_train.json').read_text())
    request.update(environment_id=environment['resource_id'], environment_revision=environment['revision'],
        training_package_id=package['resource_id'], training_package_revision=package['revision'])
    request['algorithm_config'].update(collection_steps=16, num_learning_epochs=1, num_mini_batches=1)
    request['term_config'] = {'world': {'termination': {'episode_limit': {'config': {'episode_max_steps': 17}}}}}
    assert request['topology']['world_count'] == 2 and request['root_seed'] == 7
    validated = response_json(client.post('/experiments/validate', json=request))
    assert validated['valid'], validated
    identity = response_json(client.post('/experiments', json=request), 201)['experiment_id']
    return identity, response_json(client.get(f'/experiments/{identity}'))


def _download(client, run_id, artifacts, directory):
    directory.mkdir()
    for artifact in artifacts:
        response = client.get(f'/runs/{run_id}/artifacts/{artifact["id"]}')
        assert response.status_code == 200, response.text
        assert len(response.content) == artifact['size']
        assert hashlib.sha256(response.content).hexdigest() == artifact['sha256']
        (directory / (artifact['kind'] + '-' + Path(artifact['path']).name)).write_bytes(response.content)


def _trajectory_evidence(run, artifact, case, destination):
    import numpy as np
    from muse_sdk.trajectory import TrajectoryReader, trajectory_row_bytes
    from muse_server.storage import extract_archive
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        unpacked = Path(temporary) / 'trajectory'
        extract_archive(Path(run['artifact_directory']) / artifact['path'], unpacked)
        reader = TrajectoryReader(unpacked)
        try:
            manifest = reader.manifest
            assert manifest.run_id == run['run_id'] and manifest.recording.world_ids == [0, 1]
            assert manifest.summary == run['recording']['summary'] == artifact['summary']
            assert manifest.recording.every_n_decisions == (1 if case == 'evaluate' else 3)
            assert reader.plan.recording == manifest.recording
            indices, complete = {}, []
            for world in (0, 1):
                # reader 已完成格式校验；只读规范固定 index.npy，不依赖其内部数组容器。
                index = np.load(unpacked / f'worlds/{world}/index.npy', allow_pickle=False)
                indices[str(world)] = index.tolist()
                assert tuple(index[0]) == (0, 0, 0, 0, 0, 0, 0)
                for sample, identity in enumerate(index):
                    diagnostics = reader.frame(world, sample)[2]
                    assert diagnostics['sample_index'] == sample
                    if identity[5] == 1 and (diagnostics['world']['world_terminated'] or diagnostics['world']['world_truncated']):
                        following = index[sample + 1]
                        assert tuple(following[1:]) == (int(identity[1]), int(identity[2]) + 1, 0, 0, 3, 1)
                    if identity[5] == 2:
                        following = index[sample + 1]
                        assert tuple(following[1:]) == (int(identity[1]), int(identity[2]) + 1, 0, 0, 3, 2)
                for episode in manifest.episodes[str(world)]:
                    if episode['complete']:
                        first, last = episode['first_sample_index'], episode['last_sample_index']
                        complete.append(dict(world_id=world, episode=episode['episode'], first_sample_index=first,
                            last_sample_index=last, complete_episode=True,
                            duration=float(index[last, 4] - index[first, 4]) * manifest.physics_dt))
                if case == 'train':
                    assert any(row[3] == 17 and row[5] == 1 for row in index), 'Missing stride-miss episode-limit boundary'
                if case == 'evaluate':
                    assert len(index) <= 3
                if case == 'interactive':
                    assert any(row[1] == 3 and row[5] == 2 for row in index), 'Missing same-decision command reset'
            reason = {'train': 'completed', 'evaluate': 'capacity', 'interactive': 'cancelled'}[case]
            assert manifest.summary['reason'] == reason
            if case == 'evaluate':
                assert run['recording']['rows_per_world'] == 3
                assert manifest.summary['sealed_run_decision'] < run['metrics']['steps'] == 64
            else:
                assert manifest.summary['sealed_run_decision'] == run['metrics']['steps']
            result = dict(row_bytes=trajectory_row_bytes(manifest.fields), summary=manifest.summary,
                indices=indices, episodes=manifest.episodes,
                longest_complete_episode=max(complete, key=lambda item: item['duration']) if complete else None)
            _write_json(destination / 'trajectory-evidence.json', result)
            return result
        finally:
            reader.close()


def _video_evidence(client, source, artifact_id, destination):
    import imageio_ffmpeg
    source_run = response_json(client.get(f'/runs/{source["run_id"]}'))
    assert source_run['state'] == 'COMPLETED' and source_run['metrics']['steps'] == 128
    assert source_run['metrics']['updates'] == 8 and source_run['recording']['status'] == 'published'
    artifacts = response_json(client.get(f'/runs/{source["run_id"]}/artifacts'))
    assert source['checkpoint_artifact_id'] in {item['id'] for item in artifacts}
    assert source['trajectory_artifact_id'] in {item['id'] for item in artifacts}
    recordings = [item for item in artifacts if item['kind'] == 'recording']
    assert len(recordings) == 1, 'Only one completed formal video is authorized'
    artifact = recordings[0]
    assert artifact['id'] == artifact_id
    metadata = artifact['metadata']
    assert metadata['source_trajectory_id'] == source['trajectory_artifact_id']
    for key in ('world_id', 'episode', 'first_sample_index', 'last_sample_index', 'complete_episode'):
        assert metadata[key] == source['video_episode'][key]
    assert (metadata['fps'], metadata['width'], metadata['height']) == (25., 640, 480)
    expected = max(1, math.ceil(source['video_episode']['duration'] * 25.) + 1)
    assert metadata['frame_count'] == expected
    _download(client, source['run_id'], [artifact], destination / 'video-download')
    video = destination / 'video-download' / ('recording-' + Path(artifact['path']).name)
    frames = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
    count, first, last = 0, None, None
    try:
        decoded = next(frames)
        assert decoded['size'] == (640, 480)
        for frame in frames:
            if first is None:
                first = frame
            last = frame
            count += 1
    finally:
        frames.close()
    assert count == expected
    (destination / 'video-first.rgb').write_bytes(first)
    (destination / 'video-last.rgb').write_bytes(last)
    return dict(artifact=artifact, decoded_frames=count, decoded_metadata=decoded,
        first_pixels_sha256=hashlib.sha256(first).hexdigest(), last_pixels_sha256=hashlib.sha256(last).hexdigest())


@pytest.mark.parametrize('case', CASES)
def test_recording_acceptance(case):
    if not case_authorized(case, os.environ):
        pytest.skip('Requires MUSE_TEST_GPU=1, MUSE_TEST_RECORDING_RENDER=1 and one matching MUSE_RECORDING_CASE')
    mode, decisions, updates, wall_seconds = BUDGETS[case]
    started, deadline = time.monotonic(), time.monotonic() + wall_seconds
    backend = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    row = dict(case=case, attempt=1, started_at=time.time(), backend_commit=backend,
        authorized_budget=dict(mode=mode, decisions=decisions, updates=updates, wall_seconds=wall_seconds),
        run_id=None, post_runs_attempts=0, invariants_passed=False)
    reserve_attempt(EVIDENCE, case, row)
    result_path = EVIDENCE / f'{case}-result.json'
    destination = EVIDENCE / case
    destination.mkdir()
    catalog = EVIDENCE / 'catalog'
    source = None
    if case == 'train':
        assert not catalog.exists(), 'TRAIN must start with this phase\'s new Catalog'
    else:
        source = json.loads((EVIDENCE / 'train-source.json').read_text())
        train = json.loads((EVIDENCE / 'train-result.json').read_text())
        assert (train['invariants_passed'] and train['desktop_completed'] and train['cleanup_confirmed']), 'TRAIN evidence must be complete first'
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    row['base_url'] = f'http://127.0.0.1:{port}'
    server, run_id, health = None, None, None
    owned = {}
    next_resources = 0.

    def sample_resources():
        nonlocal next_resources
        if time.monotonic() < next_resources:
            return None
        snapshot = _resources(server)
        for process in snapshot['processes']:
            if 'create_time' in process:
                previous = owned.get(process['pid'], {})
                owned[process['pid']] = dict(process, sampled_peak_rss_bytes=max(
                    previous.get('sampled_peak_rss_bytes', 0), process['rss_bytes']))
        with (destination / 'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(), **snapshot), allow_nan=False) + '\n')
        next_resources = time.monotonic() + 1.
        return snapshot

    def save():
        _write_json(result_path, row)

    def publish(stage, **extra):
        value = dict(case=case, stage=stage, catalog_id=health['catalog_id'], instance_id=health['instance_id'],
            run_id=run_id, base_url=row['base_url'], deadline_unix=row['started_at'] + wall_seconds,
            qt_scale_factor='1.5' if case == 'evaluate' else '1', backend_commit=backend, train_source=source, **extra)
        _write_json(destination / 'observer.json', value)
        return value

    def ack(stage, **extra):
        value = publish(stage, **extra)
        path = EVIDENCE / 'desktop' / case / (stage + '.json')
        def read_ack():
            # Run终态后的历史/导出仍在运行；等待GUI时继续记录其CPU/RSS与子进程。
            sample_resources()
            return json.loads(path.read_text()) if path.exists() else None
        received = _wait(read_ack, deadline, f'Observer ack {stage} exceeded case budget')
        validate_observer_ack(received, value)
        row.setdefault('observer_acks', {})[stage] = received
        save()
        return received

    save()
    with (destination / 'server.log').open('xb') as log, (destination / 'samples.jsonl').open('x') as samples:
        try:
            server = subprocess.Popen([sys.executable, '-m', 'muse_server.app', '--port', str(port),
                '--viewer-port', '0', '--data-dir', str(catalog)], stdout=log, stderr=log,
                start_new_session=True, cwd=ROOT)
            row['server_pid'] = server.pid
            save()
            with httpx2.Client(base_url=row['base_url'], timeout=10., trust_env=False) as client:
                def healthy():
                    assert server.poll() is None, 'Server exited before health response'
                    try:
                        response = client.get('/health')
                    except httpx2.ConnectError:
                        return None
                    return response_json(response) if response.is_success else None

                health = _wait(healthy, min(deadline, time.monotonic() + 30.), 'Server startup timeout')
                row.update(catalog_id=health['catalog_id'], instance_id=health['instance_id'])
                if case == 'train':
                    experiment_id, experiment = _register_experiment(client)
                    _write_json(EVIDENCE / 'experiment.json', dict(experiment_id=experiment_id, experiment=experiment))
                else:
                    assert health['catalog_id'] == source['catalog_id']
                    saved = json.loads((EVIDENCE / 'experiment.json').read_text())
                    experiment_id, experiment = saved['experiment_id'], saved['experiment']
                    assert response_json(client.get(f'/experiments/{experiment_id}')) == experiment
                ready = ack('ready_to_submit')
                row['gui_commit'] = ready['gui_commit']
                assert isinstance(row['gui_commit'], str) and row['gui_commit']
                assert ready['qt_scale_factor'] == ('1.5' if case == 'evaluate' else '1')
                if case == 'evaluate':
                    prepared = ack('prepare_evaluate')
                    replay = response_json(client.get(f'/replays/{prepared["session_id"]}'))
                    assert replay['status'] == 'ready' and replay['camera_ready']
                    assert replay['source'] == dict(run_id=source['run_id'], trajectory_artifact_id=source['trajectory_artifact_id'])
                    row['prepared_replay_id'] = prepared['session_id']
                row['baseline'] = _resources(server)
                for process in row['baseline']['processes']:
                    if 'create_time' in process:
                        owned[process['pid']] = process
                recording = dict(world_ids=[0, 1], every_n_decisions=1 if case == 'evaluate' else 3,
                    max_data_bytes=2 * source['row_bytes'] * 3 if case == 'evaluate' else 16 * 1024 * 1024)
                request = dict(experiment_id=experiment_id, mode=mode, max_decision_steps=decisions,
                    checkpoint_interval_updates=None, recording=recording)
                if source is not None:
                    request['checkpoint_artifact_id'] = source['checkpoint_artifact_id']
                previous_submissions = []
                for previous_case in CASES:
                    previous_path = EVIDENCE / f'{previous_case}-result.json'
                    if previous_path.exists():
                        previous = json.loads(previous_path.read_text())
                        if previous['post_runs_attempts']:
                            previous_submissions.append(previous)
                row['remaining_before_submission'] = dict(
                    run_slots=3 - len(previous_submissions),
                    decision_upper_bound=224 - sum(item['authorized_budget']['decisions'] for item in previous_submissions),
                    update_upper_bound=8 - sum(item['authorized_budget']['updates'] for item in previous_submissions))
                assert row['remaining_before_submission']['run_slots'] > 0
                row.update(request=request, post_runs_attempts=1, submitting_at=time.time())
                save()
                assert time.monotonic() < deadline, 'Case budget exhausted before submission'
                # 唯一 POST /runs；先永久记录 attempt 和提交意图，网络失败也不重试。
                run = response_json(client.post('/runs', json=request), 201)
                run_id = row['run_id'] = run['run_id']
                save()
                publish('run_active')
                viewers = ('viser', 'native') if os.environ.get('MUSE_TEST_NATIVE') == '1' else ('viser',)
                # 准备期 pause 先投递；首次资源采样不抢占该准入窗口。
                next_resources = time.monotonic() + 1.

                def observe():
                    assert time.monotonic() < deadline, 'Authorized case wall-clock budget exhausted'
                    current = response_json(client.get(f'/runs/{run_id}'))
                    display = response_json(client.get(f'/runs/{run_id}/display'))
                    error = confirmed_display_error(client, run_id, display, viewers)
                    assert error is None, error
                    sample = dict(time=time.time(), run=current, display=display)
                    resources = sample_resources()
                    if resources is not None:
                        sample['resources'] = resources
                    samples.write(json.dumps(sample, allow_nan=False) + '\n')
                    samples.flush()
                    return current

                def command(kind, world_id=None):
                    current = observe()
                    if kind == 'pause':
                        row['preparation_pause_state'] = current['state']
                        assert current['state'] in ('PREPARING', 'COMPILING'), 'Preparation pause window missed'
                    previous = current['interaction']['command']
                    sequence = previous['command_seq'] + 1 if previous else 1
                    payload = dict(command_seq=sequence, command=kind)
                    if world_id is not None:
                        payload['world_id'] = world_id
                    response_json(client.post(f'/runs/{run_id}/commands', json=payload))

                    def applied():
                        snapshot = observe()
                        assert snapshot['state'] not in TERMINAL, snapshot
                        result = snapshot['interaction']['result']
                        return result if result is not None and result['command_seq'] == sequence else None

                    result = _wait(applied, deadline, f'{kind} command result timed out')
                    assert result['status'] == 'applied' and result['updates'] == 0, result
                    return result

                if case == 'interactive':
                    live = _wait(lambda: value if (value := observe())['pid'] is not None else None,
                        deadline, 'Worker did not start')
                    assert live['state'] in ('PREPARING', 'COMPILING'), 'Preparation pause window missed; do not retry'
                    assert command('pause')['steps'] == 0
                    attached = ack('interactive_paused')
                    assert attached['viewer_attached'] is True and attached['world_id'] == 0
                    for _ in range(3):
                        command('step')
                    assert command('reset_world', 0)['steps'] == 3
                    assert command('reset_world', 1)['steps'] == 3
                    for viewer in viewers:
                        response_json(client.patch(f'/runs/{run_id}/display', json={viewer: {'enabled': True, 'world_id': 0}}))
                        response_json(client.patch(f'/runs/{run_id}/display', json={viewer: {'world_id': 1}}))
                        def switched():
                            observe()
                            slot = response_json(client.get(f'/runs/{run_id}/display'))[viewer]
                            return slot if slot['status'] == 'ready' and slot['frame'] is not None and slot['frame']['world_id'] == 1 else None
                        _wait(switched, deadline, f'{viewer} world switch was not applied')
                    for _ in range(2):
                        command('step')
                    closed = ack('close_interactive_gui')
                    assert closed['gui_closed'] is True
                    before = observe()
                    assert before['state'] == 'RUNNING'
                    assert command('step')['steps'] == 6, 'GUI close must not cancel the Run'
                    response_json(client.post(f'/runs/{run_id}/cancel'))
                run = _wait(lambda: value if (value := observe())['state'] in TERMINAL else None,
                    deadline, 'Worker terminal timed out')
                row.update(state=run['state'], metrics=run['metrics'], exit_code=run['exit_code'],
                    artifact_directory=run['artifact_directory'], recording=run['recording'])
                save()
                assert run['exit_code'] == 0 and not run['forced'], run
                assert run['state'] == ('CANCELLED' if case == 'interactive' else 'COMPLETED'), run
                assert run['metrics']['updates'] == updates
                assert run['metrics']['steps'] == (6 if case == 'interactive' else decisions)
                assert run['recording']['status'] == 'published'
                frozen = json.loads(Path(run['artifact_directory'], 'resolved_run_plan.json').read_text())
                assert frozen['experiment'] == experiment and frozen['recording'] == recording
                assert frozen['checkpoint_interval_updates'] is None
                events = [json.loads(line) for line in Path(run['artifact_directory'], 'events.jsonl').read_text().splitlines()]
                update_steps = [event['payload']['steps'] for event in events
                    if event['kind'] == 'metric' and event['payload'].get('source') == 'update']
                row['capture_statistics'] = [event['payload'] for event in events
                    if event['kind'] == 'metric' and event['payload'].get('source') == 'recording']
                assert len(row['capture_statistics']) == 1
                assert update_steps == (list(range(16, 129, 16)) if case == 'train' else [])
                artifacts = response_json(client.get(f'/runs/{run_id}/artifacts'))
                row['artifacts'] = artifacts
                _download(client, run_id, artifacts, destination / 'downloads')
                trajectories = [item for item in artifacts if item['kind'] == 'trajectory']
                assert len(trajectories) == 1 and trajectories[0]['id'] == run['recording']['artifact_id']
                recorded = _trajectory_evidence(run, trajectories[0], case, destination)
                if case == 'train':
                    checkpoints = [item for item in artifacts if item['kind'] == 'checkpoint']
                    assert len(checkpoints) == 1
                    assert recorded['longest_complete_episode'] is not None, 'No complete episode; no automatic extra Run'
                    source = dict(catalog_id=health['catalog_id'], run_id=run_id, experiment_id=experiment_id,
                        trajectory_artifact_id=trajectories[0]['id'], checkpoint_artifact_id=checkpoints[0]['id'],
                        row_bytes=recorded['row_bytes'], video_episode=recorded['longest_complete_episode'])
                    _write_json(EVIDENCE / 'train-source.json', source)
                completed = ack('completed', trajectory_artifact_id=trajectories[0]['id'])
                assert completed['gui_closed'] is True
                assert completed['evidence'] and all((EVIDENCE / path).is_file() for path in completed['evidence'])
                row['desktop_metrics'] = completed['desktop_metrics']
                checks = {'train': ('replay_after_terminal', 'object_diagnostics_same_sample', 'backward_seek', 'download_verified', 'dpr_verified'),
                    'evaluate': ('train_replay_preopened', 'export_requested', 'download_verified', 'local_player_playback', 'cancelled_export_cleaned', 'dpr_verified'),
                    'interactive': ('viewer_world_switch', 'viewer_disconnect', 'gui_closed_run_survived', 'reopen_cancelled_trajectory', 'dpr_verified')}[case]
                assert all(completed['checks'][key] is True for key in checks), completed
                if case == 'evaluate':
                    row['video'] = _video_evidence(client, source, completed['artifact_id'], destination)
                    row['overlap_observed'] = completed['overlap_observed']
                    if not row['overlap_observed']:
                        row['overlap_limitation'] = '64 decisions completed before an observable export overlap; no extension/retry authorized'
                for session_id in completed['session_ids']:
                    _wait(lambda: client.get(f'/replays/{session_id}').status_code == 404,
                        deadline, 'GUI replay session not released')
                assert response_json(client.get('/health'))['gpu_busy'] is False
                row.update(invariants_passed=True, desktop_completed=True, completed_at=time.time())
                save()
        except BaseException as exc:
            row.update(test_error=repr(exc), traceback=traceback.format_exc(), failed_at=time.time())
            save()
            raise
        finally:
            try:
                if server is not None and server.poll() is None:
                    try:
                        if run_id is not None:
                            with httpx2.Client(base_url=row['base_url'], timeout=10., trust_env=False) as cleanup:
                                current = response_json(cleanup.get(f'/runs/{run_id}'))
                                if current['state'] not in TERMINAL:
                                    response_json(cleanup.post(f'/runs/{run_id}/cancel'))
                    finally:
                        server.terminate()
                        try:
                            server.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            for process in _resources(server)['processes']:
                                if 'create_time' in process:
                                    owned[process['pid']] = process
                            for process in reversed(list(owned.values())):
                                try:
                                    live = psutil.Process(process['pid'])
                                    if live.create_time() == process['create_time']:
                                        live.kill()
                                except psutil.NoSuchProcess:
                                    pass
                            server.wait(timeout=5)
                            raise AssertionError('Server cleanup exceeded stop deadline; owned processes were killed')
                alive = []
                for process in owned.values():
                    try:
                        live = psutil.Process(process['pid'])
                        if live.create_time() == process['create_time'] and live.status() != psutil.STATUS_ZOMBIE:
                            alive.append(process)
                    except psutil.NoSuchProcess:
                        pass
                assert not alive, f'Owned process exit unconfirmed: {alive}'
                row['cleanup_confirmed'] = True
            finally:
                row.update(server_exit_code=server.returncode if server else None,
                    elapsed_seconds=time.monotonic() - started, cleanup_finished_at=time.time(),
                    observed_processes=list(owned.values()))
                if row['elapsed_seconds'] > wall_seconds:
                    row['invariants_passed'] = False
                    row['wall_budget_exceeded'] = True
                save()
                assert row['elapsed_seconds'] <= wall_seconds, 'Case including cleanup exceeded authorized wall-clock budget'
