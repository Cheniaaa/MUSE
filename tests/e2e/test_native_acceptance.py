"""M5.1官方Native工程验收：每次只创建一个有界Run，独立保留本期证据。"""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx2
import psutil
import pytest

from test_basic_flight import archive, response_json, TERMINAL


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / '.planning/mjlab-spike/m5-official-native-acceptance'
CASES = ('native', 'dual', 'interactive')
SELECTED = os.environ.get('MUSE_NATIVE_CASE')
if SELECTED is not None and SELECTED not in CASES:
    raise pytest.UsageError('MUSE_NATIVE_CASE must name one of: ' + ', '.join(CASES))


@pytest.mark.parametrize('case', CASES)
def test_native_acceptance(case):
    if os.environ.get('MUSE_TEST_GPU') != '1' or SELECTED != case:
        pytest.skip('Requires explicit MUSE_TEST_GPU=1 and this authorized MUSE_NATIVE_CASE')
    if os.environ.get('MUSE_TEST_NATIVE') != '1':
        pytest.skip('Display acceptance requires explicit desktop validation')
    EVIDENCE.mkdir(exist_ok=True)
    report_path = EVIDENCE / 'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {'cases': {}}
    attempt = int(os.environ.get('MUSE_NATIVE_ATTEMPT', '1'))
    previous = report['cases'].get(case)
    history = report.get('failed_attempts', {}).get(case, [])
    assert attempt == (len(history) + 2 if previous else 1), 'Select the next authorized attempt explicitly'
    if previous:
        assert not previous.get('invariants_passed'), 'Do not repeat successful cases'
        report.setdefault('failed_attempts', {}).setdefault(case, []).append(previous)
    evidence_prefix = case if attempt == 1 else f'{case}-attempt{attempt}'
    if case != 'native':
        assert 'experiment_id' in report, 'Use the Experiment registered by the authorized Native case'
    if case == 'interactive':
        assert report['cases']['dual']['state'] == 'COMPLETED'
        assert report['cases']['dual']['checkpoint_artifact_id']
    def save():
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    row = report['cases'][case] = {'started_at': time.time(), 'state': 'preparing_test',
                                  'attempt': attempt, 'evidence_prefix': evidence_prefix,
                                  'base_url': f'http://127.0.0.1:{port}'}
    save()
    server = None
    run_id = None
    with (EVIDENCE / f'{evidence_prefix}-server.log').open('xb') as log:
        try:
            server = subprocess.Popen([sys.executable, '-m', 'muse_server.app', '--port', str(port),
                '--viewer-port', '0', '--data-dir', str(EVIDENCE / 'catalog')], stdout=log, stderr=log)
            row['server_pid'] = server.pid
            save()
            with httpx2.Client(base_url=row['base_url'], timeout=10.) as client:
                startup = time.monotonic() + 30
                while True:
                    assert server.poll() is None, 'Server exited during startup'
                    try:
                        if client.get('/health').is_success:
                            break
                    except httpx2.ConnectError:
                        pass
                    assert time.monotonic() < startup, 'Server startup timeout'
                    time.sleep(.1)
                if case == 'native':
                    example = ROOT / 'examples/basic_flight'
                    env = response_json(client.post('/catalog/environments', content=archive(example / 'environment'),
                        headers={'Content-Type': 'application/zip'}), 201)
                    package = response_json(client.post('/catalog/training-packages',
                        content=archive(example / 'training_package_ippo'), headers={'Content-Type': 'application/zip'}), 201)
                    request = json.loads((example / 'request_train.json').read_text())
                    request.update(environment_id=env['resource_id'], environment_revision=env['revision'],
                        training_package_id=package['resource_id'], training_package_revision=package['revision'])
                    request['algorithm_config'].update(collection_steps=16, num_learning_epochs=1, num_mini_batches=1)
                    validated = response_json(client.post('/experiments/validate', json=request))
                    assert validated['valid'], validated
                    report['experiment_id'] = response_json(client.post('/experiments', json=request), 201)['experiment_id']
                    save()
                experiment = response_json(client.get(f"/experiments/{report['experiment_id']}"))
                request = dict(experiment_id=report['experiment_id'], mode='INTERACTIVE' if case == 'interactive' else 'TRAIN',
                               max_decision_steps=64 if case == 'interactive' else 512)
                if case == 'interactive':
                    request['checkpoint_artifact_id'] = report['cases']['dual']['checkpoint_artifact_id']
                row['state'] = 'submitting'
                save()
                # 整个case只有这一处POST /runs；先保存attempt，失败不会自动重复提交。
                run = response_json(client.post('/runs', json=request), 201)
                run_id = row['run_id'] = run['run_id']
                row['state'] = run['state']
                row['submitted_at'] = time.time()
                save()
                deadline = time.monotonic() + (300 if case == 'interactive' else 600)
                enabled = ('native',) if case == 'native' else ('viser', 'native')
                configured = False
                ready_at = {}
                next_resources = 0.
                with (EVIDENCE / f'{evidence_prefix}-samples.jsonl').open('x') as samples:
                    while True:
                        observed = time.time()
                        run = response_json(client.get(f'/runs/{run_id}'))
                        if run['pid'] is not None and not configured and run['state'] not in TERMINAL:
                            row['worker_pid'] = run['pid']
                            row['patch_at'] = observed
                            if case == 'interactive':
                                response_json(client.post(f'/runs/{run_id}/commands',
                                    json={'command_seq': 1, 'command': 'pause'}))
                            for viewer in enabled:
                                response_json(client.patch(f'/runs/{run_id}/display',
                                    json={viewer: {'enabled': True, 'world_id': 0, 'rate_hz': 10.}}))
                            configured = True
                            save()
                        display = response_json(client.get(f'/runs/{run_id}/display'))
                        sample = {'time': observed, 'run': run, 'display': display}
                        if observed >= next_resources:
                            parent = psutil.Process(server.pid)
                            sample['processes'] = []
                            for process in [parent, *parent.children(recursive=True)]:
                                try:
                                    sample['processes'].append({'pid': process.pid, 'command': process.cmdline(),
                                        'cpu_seconds': sum(process.cpu_times()[:2]), 'rss_bytes': process.memory_info().rss})
                                except psutil.NoSuchProcess:
                                    sample['processes'].append({'pid': process.pid, 'exited_during_sample': True})
                            # WSL/WDDM可能返回[N/A]；保留原文，不能把不可用误记为0MiB。
                            gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                            sample['gpu_memory_query'] = {'stdout': gpu.stdout, 'stderr': gpu.stderr, 'exit_code': gpu.returncode}
                            next_resources = observed + 1.
                        samples.write(json.dumps(sample, allow_nan=False) + '\n')
                        samples.flush()
                        for viewer in enabled:
                            slot = display[viewer]
                            if slot['status'] == 'ready' and slot['frame'] is not None:
                                ready_at.setdefault(viewer, observed)
                            # Worker终态之后Server仍须等viewer writer退出再发布可靠终态。
                            if (case != 'interactive' and slot['status'] == 'error'
                                    and run['state'] not in TERMINAL and run['worker_terminal'] is None):
                                pytest.fail(f'{viewer}: {slot["error"]}')
                        if run['state'] in TERMINAL:
                            break
                        assert time.monotonic() < deadline, 'Authorized wall-clock budget exhausted'
                        time.sleep(.1)
                row.update(state=run['state'], exit_code=run['exit_code'], metrics=run['metrics'],
                           ready_at=ready_at, finished_at=time.time(), artifact_directory=run['artifact_directory'])
                save()
                directory = Path(run['artifact_directory'])
                frozen = json.loads((directory / 'resolved_run_plan.json').read_text())
                assert frozen['experiment'] == experiment
                assert frozen['max_decision_steps'] == request['max_decision_steps']
                events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
                metrics = [json.loads(line)['payload'] for line in (directory / 'metrics.jsonl').read_text().splitlines()]
                assert run['exit_code'] == 0 and not run['forced'], run
                if case == 'interactive':
                    assert run['state'] == 'CANCELLED' and run['metrics']['steps'] <= 64 and run['metrics']['updates'] == 0
                    commands = [event['payload'] for event in events if event['kind'] == 'command_result']
                    assert {'pause', 'step', 'reset_world'} <= {command['command'] for command in commands}
                    assert all(command['status'] == 'applied' for command in commands)
                else:
                    assert run['state'] == 'COMPLETED'
                    assert run['metrics']['steps'] == 512 and run['metrics']['updates'] == 32
                    assert [value['steps'] for value in metrics if value['source'] == 'update'] == list(range(16, 513, 16))
                assert set(ready_at) == set(enabled)
                if case != 'interactive':
                    observed_samples = [json.loads(line) for line in (EVIDENCE / f'{evidence_prefix}-samples.jsonl').read_text().splitlines()]
                    start = max(ready_at.values())
                    first = next((value for value in observed_samples if value['time'] >= start + 10.), None)
                    last = next((value for value in observed_samples if value['time'] >= start + 30.), None)
                    if first is not None and last is not None:
                        elapsed = last['time'] - first['time']
                        decisions = last['run']['metrics']['steps'] - first['run']['metrics']['steps']
                        row['steady_observation'] = dict(seconds=elapsed, batch_decisions=decisions,
                            batch_decisions_per_second=decisions / elapsed)
                    else:
                        row['steady_observation'] = 'insufficient 10s warmup + 20s observation window'
                artifacts = response_json(client.get(f'/runs/{run_id}/artifacts'))
                row['artifacts'] = artifacts
                for artifact in artifacts:
                    raw = client.get(f'/runs/{run_id}/artifacts/{artifact["id"]}')
                    assert raw.status_code == 200
                    assert len(raw.content) == artifact['size']
                    assert hashlib.sha256(raw.content).hexdigest() == artifact['sha256']
                checkpoints = [item for item in artifacts if item['kind'] == 'checkpoint']
                if case != 'interactive':
                    assert checkpoints
                    row['checkpoint_artifact_id'] = checkpoints[-1]['id']
                row['invariants_passed'] = True
                # 桌面截图、真实WebEngine连接、CPU/显存和稳态区间由同次现场记录补充，不伪造自动通过。
                row['desktop_evidence'] = 'pending independent observation'
                save()
                # 本验收的真实GUI是同次Run消费者；收齐其终态取证后才停Server。
                scale = '1' if case == 'native' else '1.5'
                desktop = EVIDENCE / f'desktop/{case}-scale{scale}/result.json'
                while not desktop.exists():
                    assert time.monotonic() < deadline, 'Desktop observation exceeded authorized budget'
                    time.sleep(.1)
                desktop_result = json.loads(desktop.read_text())
                assert desktop_result['success'] and desktop_result['gui_closed'], desktop_result
                row['desktop_evidence'] = str(desktop.relative_to(EVIDENCE))
                save()
        except BaseException as exc:
            row['test_error'] = repr(exc)
            row['test_failed_at'] = time.time()
            save()
            raise
        finally:
            try:
                if run_id is not None and server.poll() is None:
                    with httpx2.Client(base_url=row['base_url'], timeout=10.) as client:
                        state = response_json(client.get(f'/runs/{run_id}'))
                        if state['state'] not in TERMINAL:
                            response_json(client.post(f'/runs/{run_id}/cancel'))
            finally:
                if server is not None and server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=5)
                        pytest.fail('Server shutdown exceeded deadline; cleanup requires inspection')
