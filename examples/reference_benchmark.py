"""通过正式 HTTP API 执行固定的五配置有限试验，不创建本地环境。"""
from collections.abc import Iterable
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import time
import traceback
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

import yaml

from muse_sdk.run_protocol import RunTimeouts


CONFIGURATIONS = (
    ('isac-mappo-cost-on', 'isac_cost', 'builtin:mappo', True, 0),
    ('isac-mappo-cost-off', 'isac_cost', 'builtin:mappo', False, 0),
    ('isac-ippo-cost-off', 'isac_cost', 'builtin:ippo', False, 0),
    ('history-ippo-l1', 'hybrid_history_markov', 'builtin:ippo', False, 1),
    ('history-ippo-l4', 'hybrid_history_markov', 'builtin:ippo', False, 4),
)
ALGORITHM_CONFIG = dict(collection_steps=32, hidden_dims=[64, 64], learning_rate=.0003,
                        num_learning_epochs=4, num_mini_batches=4, gamma=.99, lam=.95,
                        clip_param=.2, value_loss_coef=1., entropy_coef=.01, max_grad_norm=1.)
TERMINAL = frozenset(('COMPLETED', 'CANCELLED', 'FAILED', 'LOST'))


class _BudgetExpired(TimeoutError):
    """由唯一脚本 owner 的 deadline 触发，不代表可重试的网络错误。"""


def first_episode_metrics(events: Iterable[dict], *, worlds: int, discount: float, sensing: bool) -> list[dict]:
    """只收各 world 首段；缺帧不能用下个 episode 或最后快照补齐。"""
    rows = [dict(world=world, length=0, complete=False, failed=False, service_valid=True,
                 raw_return=0., discounted_return=0., served_bits=0., rf_energy_j=0.)
            for world in range(worlds)]
    if sensing:
        for row in rows:
            row.update(position_violation_steps=0., position_violation_rate=0., discounted_mission_cost=0.)
    for event in events:
        if event['kind'] != 'metric' or event['payload']['source'] != 'step':
            continue
        task = event['payload']['diagnostics']['task']
        if any(len(values) != worlds for values in task.values()):
            raise ValueError('Task diagnostics world count mismatch')
        for world, row in enumerate(rows):
            if row['complete']:
                continue
            decision = task['episode_decision'][world]
            if type(decision) is not int or decision != row['length'] + 1:
                raise ValueError(f'world {world}: discontinuous episode_decision {decision}')
            weight = discount ** (decision - 1)
            row['length'] = decision
            row['complete'] = task['task_done'][world]
            row['failed'] = task['failed'][world]
            row['service_valid'] = row['service_valid'] and task['service_valid'][world]
            row['raw_return'] += task['task_reward'][world]
            row['discounted_return'] += weight * task['task_reward'][world]
            row['served_bits'] += task['served_bits'][world]
            row['rf_energy_j'] += task['rf_energy_j'][world]
            if sensing:
                row['position_violation_steps'] += task['position_violation'][world]
                row['position_violation_rate'] = row['position_violation_steps'] / decision
                row['discounted_mission_cost'] += weight * task['mission_cost'][world]
    if any(row['length'] == 0 for row in rows):
        raise ValueError('missing first episode diagnostics')
    return rows


def _archive(directory, composition=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                relative = path.relative_to(directory).as_posix()
                if relative == 'training_package.yaml' and composition is not None:
                    archive.writestr(relative, yaml.safe_dump(composition, sort_keys=False))
                else:
                    archive.write(path, relative)
    return buffer.getvalue()


def build_variants(examples: Path) -> list[dict]:
    """IPPO 注册另一份真实 composition；其余差异只用已有请求覆盖。"""
    variants = []
    for name, scenario, algorithm, costs, history in CONFIGURATIONS:
        source = examples / scenario
        request = json.loads((source / 'request_train.json').read_text())
        composition = None
        request['algorithm_id'] = algorithm
        request['algorithm_config'] = deepcopy(ALGORITHM_CONFIG)
        if algorithm == 'builtin:mappo':
            request['algorithm_config'].update(share_actor_parameters=True, recurrent=None)
        if not costs:
            request['cost_config'] = dict(roles={}, world={})
        if scenario == 'isac_cost':
            if algorithm == 'builtin:ippo':
                composition = yaml.safe_load((source / 'training_package/training_package.yaml').read_text())
                composition['algorithm'] = algorithm
                composition['algorithm_config'] = deepcopy(ALGORITHM_CONFIG)
                del composition['critic_inputs']
                del composition['world_reward_terms']
                del composition['world_cost_terms']
                for role in composition['agent_roles'].values():
                    for selection in role['reward_terms']:
                        selection['default_enabled'] = True
            elif not costs:
                request['term_config'] = {'world': {'cost': {'mission': {'enabled': False}}}}
        else:
            request['term_config'] = {'roles': {'pilot': {'observation': {'feedback': {'history_length': history}}}}}
        variants.append(dict(name=name, scenario=scenario, request=request,
                             package=_archive(source / 'training_package', composition),
                             environment=_archive(source / 'environment')))
    return variants


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _api(server, method, route, payload=None, *, binary=False, download=False, deadline=None):
    remaining = 60. if deadline is None else min(60., deadline - time.monotonic())
    if remaining <= 0:
        raise _BudgetExpired('wall-clock budget expired')
    body = payload if binary else (json.dumps(payload).encode() if payload is not None else None)
    request = Request(server.rstrip('/') + route, data=body, method=method,
                      headers={'Content-Type': 'application/zip' if binary else 'application/json'})
    try:
        with urlopen(request, timeout=remaining) as response:
            content = response.read()
    except HTTPError as exc:
        exc.add_note(exc.read().decode('utf-8'))
        raise
    except (TimeoutError, URLError) as exc:
        timed_out = isinstance(exc, TimeoutError) or isinstance(exc.reason, TimeoutError)
        if timed_out and deadline is not None and time.monotonic() >= deadline:
            raise _BudgetExpired('wall-clock budget expired during HTTP request') from exc
        raise
    return content if download else json.loads(content)


def _download_artifacts(server, run_id, directory):
    artifacts = _api(server, 'GET', f'/runs/{run_id}/artifacts')
    _write_json(directory / 'artifacts.json', artifacts)
    root = (directory / 'artifacts').resolve()
    for artifact in artifacts:
        path = (root / artifact['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Artifact path escapes output directory')
        content = _api(server, 'GET', f"/runs/{run_id}/artifacts/{artifact['id']}", download=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        if len(content) != artifact['size'] or hashlib.sha256(content).hexdigest() != artifact['sha256']:
            raise ValueError(f"Artifact size/SHA256 mismatch: {artifact['id']}")
    return artifacts


def _parameter_count(path):
    import torch
    state = torch.load(path, weights_only=True, map_location='cpu')['learning_state']
    count = 0
    for owner in state['roles'].values():
        for name in owner['parameter_names']:
            model, key = name.split('.', 1)
            count += owner[model][key].numel()
    for cost in state.get('costs', {}).values():
        count += sum(cost['critic'][name].numel() for name in cost['parameter_names'])
    return count


def _schedule():
    runs = []
    for seed in (7, 17, 27):
        for name, scenario, _, _, _ in CONFIGURATIONS:
            for stage, mode, steps, worlds, root_seed in (
                ('initial', 'TRAIN', 1, 16, seed),
                ('initial-eval-101', 'EVALUATE', 500, 4, 101),
                ('initial-eval-211', 'EVALUATE', 500, 4, 211),
                ('main', 'TRAIN', 4096, 16, seed),
                ('final-eval-101', 'EVALUATE', 500, 4, 101),
                ('final-eval-211', 'EVALUATE', 500, 4, 211),
            ):
                runs.append(dict(key=f'seed-{seed}/{name}/{stage}', config=name, scenario=scenario,
                                 training_seed=seed, root_seed=root_seed, stage=stage, mode=mode,
                                 max_decision_steps=steps, worlds=worlds,
                                 budget_seconds=900 if stage == 'main' else 300,
                                 state='NOT_STARTED', complete=False))
    return runs


def _cancel_and_wait(server, row, directory):
    run_id = row['run_id']
    row['cancel_requested'] = True
    record = _api(server, 'POST', f'/runs/{run_id}/cancel')
    # 默认 Supervisor 先 cooperative cancel，再 TERM/KILL；留一次 poll/EOF 确认时间。
    limits = RunTimeouts()
    deadline = time.monotonic() + limits.cancel + limits.stop + 1.
    _write_json(directory / 'cancel.json', record)
    row['state'] = record['state']
    while record['state'] not in TERMINAL:
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Cancellation terminal unconfirmed: {run_id}')
        time.sleep(.1)
        try:
            record = _api(server, 'GET', f'/runs/{run_id}', deadline=deadline)
        except _BudgetExpired as exc:
            raise TimeoutError(f'Cancellation terminal unconfirmed: {run_id}') from exc
        row['state'] = record['state']
        _write_json(directory / 'run.json', record)
    return record


def run_benchmark(server: str, output: Path) -> dict:
    """90 个 Run 串行执行；预算或根因失败即停，保留所有未开始项。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    global_deadline = started + 240 * 60
    ledger = dict(status='RUNNING', server=server, budget_seconds=240 * 60, runs=_schedule())
    ledger_path = output / 'ledger.json'
    _write_json(ledger_path, ledger)
    active = None
    try:
        variants = {v['name']: v for v in build_variants(Path(__file__).resolve().parent)}
        for variant in variants.values():
            directory = output / 'resources' / variant['name']
            directory.mkdir(parents=True)
            for kind, route in (('environment', 'environments'), ('package', 'training-packages')):
                (directory / f'{kind}.zip').write_bytes(variant[kind])
                variant[kind + '_record'] = _api(server, 'POST', '/catalog/' + route, variant[kind],
                                                binary=True, deadline=global_deadline)
                _write_json(directory / f'{kind}.json', variant[kind + '_record'])
            _write_json(directory / 'request-template.json', variant['request'])
        checkpoints = {}
        for row in ledger['runs']:
            if time.monotonic() >= global_deadline:
                raise _BudgetExpired('global wall-clock budget expired')
            variant = variants[row['config']]
            directory = output / row['key']
            directory.mkdir(parents=True)
            request = deepcopy(variant['request'])
            environment, package = variant['environment_record'], variant['package_record']
            request.update(environment_id=environment['resource_id'], environment_revision=environment['revision'],
                           training_package_id=package['resource_id'], training_package_revision=package['revision'],
                           topology=dict(world_count=row['worlds']), root_seed=row['root_seed'])
            _write_json(directory / 'request.json', request)
            validation = _api(server, 'POST', '/experiments/validate', request, deadline=global_deadline)
            _write_json(directory / 'validation.json', validation)
            if not validation['valid']:
                raise RuntimeError(f"Experiment validation failed: {validation['errors']}")
            experiment = _api(server, 'POST', '/experiments', request, deadline=global_deadline)
            row['experiment_id'] = experiment['experiment_id']
            frozen = _api(server, 'GET', '/experiments/' + row['experiment_id'], deadline=global_deadline)
            _write_json(directory / 'experiment.json', frozen)
            run_request = dict(experiment_id=row['experiment_id'], mode=row['mode'],
                               max_decision_steps=row['max_decision_steps'])
            group = (row['training_seed'], row['config'])
            if row['stage'] != 'initial':
                source = 'main' if row['stage'].startswith('final-') else 'initial'
                run_request['checkpoint_artifact_id'] = checkpoints[(*group, source)]
            _write_json(directory / 'run_request.json', run_request)
            run_started = time.monotonic()
            deadline = min(global_deadline, run_started + row['budget_seconds'])
            record = _api(server, 'POST', '/runs', run_request, deadline=deadline)
            row.update(run_id=record['run_id'], state=record['state'])
            active = (row, directory, record)
            _write_json(directory / 'run.json', record)
            _write_json(ledger_path, ledger)
            expired = False
            try:
                while record['state'] not in TERMINAL:
                    record = _api(server, 'GET', '/runs/' + row['run_id'], deadline=deadline)
                    active = (row, directory, record)
                    _write_json(directory / 'run.json', record)
                    if time.monotonic() >= deadline:
                        raise _BudgetExpired('Run/global wall-clock budget expired')
                    if record['state'] not in TERMINAL:
                        time.sleep(min(.1, max(0., deadline - time.monotonic())))
            except _BudgetExpired:
                expired = True
                record = _cancel_and_wait(server, row, directory)
            row.update(state=record['state'], wall_time=time.monotonic() - run_started,
                       steps=record['metrics'].get('steps', 0), updates=record['metrics'].get('updates', 0))
            active = (row, directory, record)
            _write_json(directory / 'run.json', record)
            artifacts = _download_artifacts(server, row['run_id'], directory)
            if not expired and record['state'] != 'COMPLETED':
                raise RuntimeError(f"Run {row['run_id']} {record['state']}: {record['failure']}")
            event_artifacts = [a for a in artifacts if a['kind'] == 'events']
            try:
                if not event_artifacts:
                    raise ValueError('missing first episode: events artifact unavailable')
                with (directory / 'artifacts' / event_artifacts[0]['path']).open() as events:
                    summary = first_episode_metrics((json.loads(line) for line in events), worlds=row['worlds'],
                                                    discount=.99, sensing=row['scenario'] == 'isac_cost')
                _write_json(directory / 'first_episode.json', summary)
            except ValueError as exc:
                if not expired:
                    raise
                row['first_episode_error'] = str(exc)
                _write_json(directory / 'first_episode-error.json', dict(error=str(exc)))
            published = [a for a in artifacts if a['kind'] == 'checkpoint']
            if row['mode'] == 'TRAIN' and (not expired or published):
                checkpoint = published[-1]
                row['checkpoint_artifact_id'] = checkpoint['id']
                row['parameter_count'] = _parameter_count(directory / 'artifacts' / checkpoint['path'])
                checkpoints[(*group, row['stage'])] = checkpoint['id']
            if expired:
                ledger.update(status='BUDGET_EXHAUSTED', reason='Run/global wall-clock budget expired')
                break
            expected_updates = row['max_decision_steps'] // 32 if row['mode'] == 'TRAIN' else 0
            if row['steps'] != row['max_decision_steps'] or row['updates'] != expected_updates:
                raise ValueError(f"Run {row['run_id']} steps/updates differ from fixed budget")
            row['complete'] = True
            active = None
            _write_json(ledger_path, ledger)
        else:
            ledger['status'] = 'COMPLETED'
    except _BudgetExpired as exc:
        ledger.update(status='BUDGET_EXHAUSTED', reason=str(exc))
    except Exception:
        # 此处只保存根因并重新抛出，不将 HTTP/科学证据错误转换成成功或重试。
        try:
            if active is not None:
                row, directory, record = active
                if record['state'] not in TERMINAL and not row.get('cancel_requested', False):
                    record = _cancel_and_wait(server, row, directory)
                    row.update(state=record['state'], steps=record['metrics'].get('steps', 0),
                               updates=record['metrics'].get('updates', 0))
                    _write_json(directory / 'run.json', record)
                    _download_artifacts(server, row['run_id'], directory)
        finally:
            if active is not None:
                active[0]['wall_time'] = time.monotonic() - run_started
            ledger.update(status='FAILED', wall_time=time.monotonic() - started)
            (output / 'failure.txt').write_text(traceback.format_exc())
            _write_json(ledger_path, ledger)
        raise
    ledger['wall_time'] = time.monotonic() - started
    _write_json(ledger_path, ledger)
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ledger = run_benchmark(args.server, args.output)
    print(json.dumps(dict(status=ledger['status'], output=str(args.output)), indent=2), flush=True)
    return 0 if ledger['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
