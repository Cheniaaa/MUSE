"""固定试验的 CPU 证据；仅 HTTP 和时间是替身。"""
import importlib.util
import hashlib
import io
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.error import URLError
import zipfile

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def benchmark():
    spec = importlib.util.spec_from_file_location('reference_benchmark', ROOT / 'examples/reference_benchmark.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def step(decisions, rewards, done, failed, *, valid=None, violations=None, costs=None):
    worlds = len(decisions)
    task = dict(episode_decision=decisions, task_reward=rewards, task_done=done,
                failed=failed, service_valid=valid or [True] * worlds,
                served_bits=[10.] * worlds, rf_energy_j=[.25] * worlds)
    if violations is not None:
        task.update(position_violation=violations, mission_cost=costs)
    return dict(kind='metric', payload=dict(source='step', diagnostics=dict(task=task)))


def test_interleaved_first_episode_keeps_early_failure_and_single_tail_discount(benchmark):
    events = [
        step([1, 1], [2., 4.], [False, False], [False, False], violations=[0., 1.], costs=[0., 1.]),
        dict(kind='metric', payload=dict(source='update', metrics={'loss': 200})),
        step([2, 2], [-1.5, 6.], [True, False], [True, False],
             valid=[False, True], violations=[1., 0.], costs=[1.5, 0.]),
        step([1, 3], [1000., 8.], [False, True], [False, False], violations=[0., 1.], costs=[0., 1.]),
    ]
    result = benchmark.first_episode_metrics(iter(events), worlds=2, discount=.5, sensing=True)
    assert result == [
        dict(world=0, length=2, complete=True, failed=True, service_valid=False,
             raw_return=.5, discounted_return=1.25, served_bits=20., rf_energy_j=.5,
             position_violation_steps=1., position_violation_rate=.5, discounted_mission_cost=.75),
        dict(world=1, length=3, complete=True, failed=False, service_valid=True,
             raw_return=18., discounted_return=9., served_bits=30., rf_energy_j=.75,
             position_violation_steps=2., position_violation_rate=2 / 3, discounted_mission_cost=1.25),
    ]


def test_unfinished_first_episode_is_not_complete_and_has_no_sensing_keys(benchmark):
    result = benchmark.first_episode_metrics([
        step([1], [2.], [False], [False]), step([2], [3.], [False], [False])
    ], worlds=1, discount=.5, sensing=False)
    assert result == [dict(world=0, length=2, complete=False, failed=False, service_valid=True,
                           raw_return=5., discounted_return=3.5, served_bits=20., rf_energy_j=.5)]


@pytest.mark.parametrize('events', [
    [], [step([2], [1.], [True], [False])],
    [step([1], [1.], [False], [False]), step([3], [1.], [True], [False])],
    [step([1], [1.], [False], [False]), step([1], [100.], [True], [False])],
])
def test_missing_or_discontinuous_first_episode_is_exposed(benchmark, events):
    with pytest.raises(ValueError, match='episode_decision|missing first episode'):
        benchmark.first_episode_metrics(events, worlds=1, discount=.99, sensing=False)


def test_step_task_diagnostics_cannot_silently_disappear(benchmark):
    with pytest.raises(KeyError):
        benchmark.first_episode_metrics([dict(kind='metric', payload=dict(source='step', diagnostics={}))],
                                        worlds=1, discount=.99, sensing=False)


def test_variants_use_real_composition_and_keep_scenario_and_actor_inputs(benchmark):
    variants = benchmark.build_variants(ROOT / 'examples')
    assert [v['name'] for v in variants] == [
        'isac-mappo-cost-on', 'isac-mappo-cost-off', 'isac-ippo-cost-off', 'history-ippo-l1', 'history-ippo-l4']
    packages = []
    for variant in variants:
        with zipfile.ZipFile(io.BytesIO(variant['package'])) as archive:
            packages.append(yaml.safe_load(archive.read('training_package.yaml')))
    on, off, ippo, l1, l4 = packages
    assert on == off
    assert ippo['algorithm'] == 'builtin:ippo'
    assert not ippo.get('critic_inputs')
    assert not ippo.get('world_reward_terms')
    assert not ippo.get('world_cost_terms')
    for role in ('alpha', 'beta'):
        assert on['agent_roles'][role]['observation_terms'] == ippo['agent_roles'][role]['observation_terms']
        assert ippo['agent_roles'][role]['reward_terms'][0]['default_enabled'] is True
    assert on['task_config'] == ippo['task_config']
    assert variants[0]['request']['scenario'] == variants[1]['request']['scenario'] == variants[2]['request']['scenario']
    assert variants[0]['request']['cost_config']['world']['mission_position_violation']['budget'] == 5
    assert variants[1]['request']['cost_config'] == variants[2]['request']['cost_config'] == {'roles': {}, 'world': {}}
    assert variants[1]['request']['term_config']['world']['cost']['mission']['enabled'] is False
    assert [v['request']['term_config']['roles']['pilot']['observation']['feedback']['history_length']
            for v in variants[3:]] == [1, 4]
    assert l1 == l4
    for variant in variants:
        request = variant['request']
        from muse_sdk.experiment import ExperimentRequest
        ExperimentRequest.model_validate(request)
        assert request['algorithm_config']['collection_steps'] == 32
        assert request['algorithm_config']['hidden_dims'] == [64, 64]
        assert request['algorithm_config']['learning_rate'] == .0003
        assert request['algorithm_config']['num_learning_epochs'] == 4
        assert request['algorithm_config']['num_mini_batches'] == 4
        assert request['algorithm_config']['gamma'] == .99
        assert request['algorithm_config']['lam'] == .95
        assert request['algorithm_config']['clip_param'] == .2
        assert request['algorithm_config']['value_loss_coef'] == 1
        assert request['algorithm_config']['entropy_coef'] == .01
        assert request['algorithm_config']['max_grad_norm'] == 1
        if request['algorithm_id'] == 'builtin:mappo':
            assert request['algorithm_config']['share_actor_parameters'] is True
            assert request['algorithm_config']['recurrent'] is None
        else:
            assert 'share_actor_parameters' not in request['algorithm_config']


class Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class HttpBoundary:
    """只替换 urlopen；收到真实 Request，返回正式协议 JSON/二进制。"""
    def __init__(self, *, stalled=False, global_stall=False, fail=False, corrupt=False, no_steps=False,
                 stall_index=0, cancel_stuck=False, poll_error=False, poll_timeout=None, cancel_timeout=False):
        self.now = 0.
        self.requests = []
        self.experiments = {}
        self.runs = {}
        self.bodies = {}
        self.stalled, self.global_stall = stalled, global_stall
        self.fail, self.corrupt, self.no_steps = fail, corrupt, no_steps
        self.stall_index, self.cancel_stuck, self.poll_error = stall_index, cancel_stuck, poll_error
        self.poll_timeout, self.cancel_timeout = poll_timeout, cancel_timeout
        self.cancelled = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def urlopen(self, request, timeout):
        assert 0 < timeout <= 60
        route = request.full_url.removeprefix('http://muse')
        method = request.get_method()
        body = request.data
        self.requests.append((method, route, body))
        if route.startswith('/catalog/'):
            assert request.get_header('Content-type') == 'application/zip'
            assert zipfile.is_zipfile(io.BytesIO(body))
            value = dict(resource_id=f'resource-{len(self.requests)}', revision=1, sha256='a' * 64)
        elif route == '/experiments/validate':
            value = dict(valid=True, errors=[])
        elif route == '/experiments' and method == 'POST':
            identity = f'exp-{len(self.experiments)}'
            self.experiments[identity] = json.loads(body)
            value = dict(experiment_id=identity)
        elif route.startswith('/experiments/'):
            value = self.experiments[route.split('/')[2]]
        elif route == '/runs':
            payload = json.loads(body)
            identity = f'run-{len(self.runs)}'
            experiment = self.experiments[payload['experiment_id']]
            self.runs[identity] = dict(payload, run_id=identity, state='RUNNING', experiment=experiment)
            self.make_artifacts(identity)
            value = self.status(identity)
        else:
            identity = route.split('/')[2]
            if route.endswith('/cancel'):
                self.cancelled.append(identity)
                self.runs[identity]['state'] = 'STOP_REQUESTED' if self.cancel_stuck else 'CANCELLED'
                value = self.status(identity)
            elif '/artifacts/' in route:
                value = self.bodies[(identity, route.split('/')[-1])]
                if self.corrupt:
                    value += b'corrupt'
                return Response(value)
            elif route.endswith('/artifacts'):
                value = self.runs[identity]['artifacts']
            else:
                run = self.runs[identity]
                if self.cancel_timeout and run['state'] == 'STOP_REQUESTED':
                    self.now += 11
                    raise TimeoutError('cancel poll timed out')
                if self.poll_timeout is not None:
                    exhausted, wrapped = self.poll_timeout
                    if exhausted:
                        self.now += 300
                    error = TimeoutError('poll timed out')
                    raise URLError(error) if wrapped else error
                if self.poll_error:
                    raise HTTPError(request.full_url, 503, 'poll root cause', {}, io.BytesIO(b'upstream failure'))
                targeted = int(identity.removeprefix('run-')) == self.stall_index
                if self.global_stall and targeted:
                    self.now += 240 * 60
                elif self.stalled and targeted:
                    self.now += 301 if run['max_decision_steps'] != 4096 else 901
                else:
                    run['state'] = 'FAILED' if self.fail else 'COMPLETED'
                value = self.status(identity)
        return Response(json.dumps(value).encode())

    def status(self, identity):
        run = self.runs[identity]
        steps = (run['max_decision_steps'] if run['state'] == 'COMPLETED'
                 else (0 if self.no_steps else min(2, run['max_decision_steps'])))
        return dict(run_id=identity, state=run['state'], failure='worker root cause' if self.fail else None,
                    metrics=dict(steps=steps, updates=steps // 32 if run['mode'] == 'TRAIN' else 0))

    def make_artifacts(self, identity):
        run = self.runs[identity]
        worlds = run['experiment']['topology']['world_count']
        sensing = run['experiment']['task_id'] == 'isac_task'
        events = []
        decisions = min(run['max_decision_steps'], 500)
        if (self.stalled or self.global_stall) and int(identity.removeprefix('run-')) == self.stall_index:
            decisions = min(2, decisions)
        for decision in range(1, decisions + 1):
            events.append(step([decision] * worlds, [1.] * worlds,
                               [decision == 500] * worlds, [False] * worlds,
                               violations=[0.] * worlds if sensing else None,
                               costs=[0.] * worlds if sensing else None))
        if self.no_steps:
            events = []
        contents = [('run_plan', 'resolved_run_plan.json', json.dumps(dict(experiment=run['experiment'],
                    timeouts=dict(cancel=5., stop=5.))).encode()),
                    ('events', 'events.jsonl', b''.join(json.dumps(e).encode() + b'\n' for e in events)),
                    ('metrics', 'metrics.jsonl', b''.join(json.dumps(e).encode() + b'\n' for e in events))]
        if run['mode'] == 'TRAIN':
            state = dict(roles=dict(pilot=dict(
                actor={'weight': torch.ones(2, 3), 'buffer': torch.ones(100)}, critic={'bias': torch.ones(2)},
                parameter_names=['actor.weight', 'critic.bias'])))
            if run['experiment']['cost_config']['world']:
                state['costs'] = {
                    'world': dict(critic={'weight': torch.ones(3)}, parameter_names=['weight'],
                                  **{'lambda': torch.ones(1)})}
            checkpoint = dict(learning_state=state)
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            contents.append(('checkpoint', 'checkpoints/final.pt', buffer.getvalue()))
        run['artifacts'] = []
        for kind, path, body in contents:
            artifact_id = identity + ':' + kind
            self.bodies[(identity, artifact_id)] = body
            run['artifacts'].append(dict(id=artifact_id, run=identity, kind=kind, path=path,
                                         size=len(body), sha256=hashlib.sha256(body).hexdigest()))


def install_boundary(monkeypatch, benchmark, boundary):
    monkeypatch.setattr(benchmark, 'urlopen', boundary.urlopen)
    monkeypatch.setattr(benchmark.time, 'monotonic', boundary.monotonic)
    monkeypatch.setattr(benchmark.time, 'sleep', boundary.sleep)


def test_fixed_http_experiment_runs_all_90_in_frozen_order_and_saves_evidence(benchmark, monkeypatch, tmp_path):
    boundary = HttpBoundary()
    install_boundary(monkeypatch, benchmark, boundary)
    output = tmp_path / 'new-output'
    ledger = benchmark.run_benchmark('http://muse', output)
    assert ledger['status'] == 'COMPLETED'
    assert len(ledger['runs']) == len(boundary.runs) == 90
    assert [r['training_seed'] for r in ledger['runs'][::30]] == [7, 17, 27]
    assert [r['config'] for r in ledger['runs'][:30:6]] == [
        'isac-mappo-cost-on', 'isac-mappo-cost-off', 'isac-ippo-cost-off', 'history-ippo-l1', 'history-ippo-l4']
    for start in range(0, 90, 6):
        rows = ledger['runs'][start:start + 6]
        assert [r['stage'] for r in rows] == ['initial', 'initial-eval-101', 'initial-eval-211',
                                             'main', 'final-eval-101', 'final-eval-211']
        assert [r['steps'] for r in rows] == [1, 500, 500, 4096, 500, 500]
        assert [r['updates'] for r in rows] == [0, 0, 0, 128, 0, 0]
        assert [r['budget_seconds'] for r in rows] == [300, 300, 300, 900, 300, 300]
        runs = [boundary.runs[r['run_id']] for r in rows]
        assert [r['experiment']['topology']['world_count'] for r in runs] == [16, 4, 4, 16, 4, 4]
        assert [r['experiment']['root_seed'] for r in runs] == [rows[0]['training_seed'], 101, 211,
                                                              rows[0]['training_seed'], 101, 211]
        initial = rows[0]['checkpoint_artifact_id']
        final = rows[3]['checkpoint_artifact_id']
        assert [r.get('checkpoint_artifact_id') for r in runs] == [None, initial, initial, initial, final, final]
        assert rows[0]['parameter_count'] == rows[3]['parameter_count'] == (
            11 if rows[0]['config'] == 'isac-mappo-cost-on' else 8)
        for row, run in zip(rows, runs):
            directory = output / row['key']
            assert json.loads((directory / 'run.json').read_text())['state'] == 'COMPLETED'
            assert (directory / 'request.json').is_file()
            assert (directory / 'run_request.json').is_file()
            assert (directory / 'artifacts/resolved_run_plan.json').is_file()
            assert (directory / 'artifacts/metrics.jsonl').is_file()
            if run['mode'] == 'EVALUATE':
                summary = json.loads((directory / 'first_episode.json').read_text())
                assert len(summary) == 4 and all(r['complete'] for r in summary)
                assert all(r['discounted_return'] == pytest.approx((1 - .99 ** 500) / .01) for r in summary)
    assert not boundary.cancelled
    assert json.loads((output / 'ledger.json').read_text()) == ledger


@pytest.mark.parametrize('global_stall', [False, True])
def test_budget_cancel_confirms_terminal_and_retains_all_unstarted_runs(benchmark, monkeypatch, tmp_path, global_stall):
    boundary = HttpBoundary(stalled=not global_stall, global_stall=global_stall, no_steps=True)
    install_boundary(monkeypatch, benchmark, boundary)
    output = tmp_path / 'budget'
    ledger = benchmark.run_benchmark('http://muse', output)
    assert ledger['status'] == 'BUDGET_EXHAUSTED'
    assert len(boundary.runs) == 1 and boundary.cancelled == ['run-0']
    assert ledger['runs'][0]['state'] == 'CANCELLED'
    assert ledger['runs'][0]['complete'] is False
    assert all(r['state'] == 'NOT_STARTED' for r in ledger['runs'][1:])
    assert 'budget' in ledger['reason'].lower()
    assert (output / ledger['runs'][0]['key'] / 'artifacts/events.jsonl').is_file()
    assert 'missing first episode' in ledger['runs'][0]['first_episode_error']


def test_main_budget_is_15_minutes_and_keeps_partial_episode_evidence(benchmark, monkeypatch, tmp_path):
    boundary = HttpBoundary(stalled=True, stall_index=3)
    install_boundary(monkeypatch, benchmark, boundary)
    output = tmp_path / 'main-budget'
    ledger = benchmark.run_benchmark('http://muse', output)
    assert ledger['status'] == 'BUDGET_EXHAUSTED'
    assert len(boundary.runs) == 4 and boundary.cancelled == ['run-3']
    row = ledger['runs'][3]
    assert row['state'] == 'CANCELLED' and row['steps'] == 2
    assert row['budget_seconds'] == 900 and row['wall_time'] == 901
    summary = json.loads((output / row['key'] / 'first_episode.json').read_text())
    assert all(r['length'] == 2 and not r['complete'] for r in summary)
    assert all(r['state'] == 'NOT_STARTED' for r in ledger['runs'][4:])


def test_unconfirmed_cancellation_is_failure_not_a_second_cancel_or_success(benchmark, monkeypatch, tmp_path):
    boundary = HttpBoundary(stalled=True, cancel_stuck=True)
    install_boundary(monkeypatch, benchmark, boundary)
    output = tmp_path / 'cancel-unconfirmed'
    with pytest.raises(TimeoutError, match='Cancellation terminal unconfirmed'):
        benchmark.run_benchmark('http://muse', output)
    ledger = json.loads((output / 'ledger.json').read_text())
    assert ledger['status'] == 'FAILED'
    assert ledger['runs'][0]['state'] == 'STOP_REQUESTED'
    assert not ledger['runs'][0]['complete']
    assert boundary.cancelled == ['run-0']
    assert all(r['state'] == 'NOT_STARTED' for r in ledger['runs'][1:])


def test_cancel_socket_deadline_does_not_skip_terminal_confirmation(benchmark, monkeypatch, tmp_path):
    boundary = HttpBoundary(stalled=True, cancel_stuck=True, cancel_timeout=True)
    install_boundary(monkeypatch, benchmark, boundary)
    output = tmp_path / 'cancel-timeout'
    with pytest.raises(TimeoutError, match='Cancellation terminal unconfirmed'):
        benchmark.run_benchmark('http://muse', output)
    ledger = json.loads((output / 'ledger.json').read_text())
    assert ledger['status'] == 'FAILED'
    assert boundary.cancelled == ['run-0']


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('exhausted', [False, True])
def test_network_timeout_is_budget_only_when_owner_deadline_has_expired(benchmark, monkeypatch, tmp_path,
                                                                     wrapped, exhausted):
    boundary = HttpBoundary(poll_timeout=(exhausted, wrapped), no_steps=True)
    install_boundary(monkeypatch, benchmark, boundary)
    output = tmp_path / 'socket-timeout'
    if exhausted:
        ledger = benchmark.run_benchmark('http://muse', output)
        assert ledger['status'] == 'BUDGET_EXHAUSTED'
    else:
        with pytest.raises((TimeoutError, URLError), match='poll timed out'):
            benchmark.run_benchmark('http://muse', output)
        ledger = json.loads((output / 'ledger.json').read_text())
        assert ledger['status'] == 'FAILED'
    assert boundary.cancelled == ['run-0']
    assert ledger['runs'][0]['state'] == 'CANCELLED'
    assert ledger['runs'][0]['wall_time'] >= 0
    assert all(r['state'] == 'NOT_STARTED' for r in ledger['runs'][1:])


def test_http_poll_failure_cancels_owned_active_run_and_preserves_root(benchmark, monkeypatch, tmp_path):
    boundary = HttpBoundary(poll_error=True)
    install_boundary(monkeypatch, benchmark, boundary)
    output = tmp_path / 'poll-error'
    with pytest.raises(HTTPError, match='poll root cause'):
        benchmark.run_benchmark('http://muse', output)
    ledger = json.loads((output / 'ledger.json').read_text())
    assert ledger['status'] == 'FAILED' and ledger['runs'][0]['state'] == 'CANCELLED'
    assert boundary.cancelled == ['run-0']
    assert 'upstream failure' in (output / 'failure.txt').read_text()


@pytest.mark.parametrize('pending_cancel', [False, True])
def test_user_interrupt_cancels_active_run_and_preserves_interruption(benchmark, monkeypatch, tmp_path, pending_cancel):
    boundary = HttpBoundary(cancel_stuck=pending_cancel)
    install_boundary(monkeypatch, benchmark, boundary)
    interrupted = False
    def interrupt_poll(request, timeout):
        nonlocal interrupted
        if request.get_method() == 'GET' and request.full_url == 'http://muse/runs/run-0':
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt('user interrupt')
            boundary.runs['run-0']['state'] = 'CANCELLED'
            return Response(json.dumps(boundary.status('run-0')).encode())
        return boundary.urlopen(request, timeout)
    monkeypatch.setattr(benchmark, 'urlopen', interrupt_poll)
    output = tmp_path / 'interrupted'
    with pytest.raises(KeyboardInterrupt, match='user interrupt'):
        benchmark.run_benchmark('http://muse', output)
    ledger = json.loads((output / 'ledger.json').read_text())
    assert ledger['status'] == 'FAILED' and ledger['runs'][0]['state'] == 'CANCELLED'
    assert not ledger['runs'][0]['complete']
    assert boundary.cancelled == ['run-0'] and len(boundary.runs) == 1
    assert all(row['state'] == 'NOT_STARTED' for row in ledger['runs'][1:])
    directory = output / ledger['runs'][0]['key']
    assert (directory / 'artifacts/events.jsonl').is_file()
    assert 'KeyboardInterrupt: user interrupt' in (output / 'failure.txt').read_text()


@pytest.mark.parametrize('fault', ['fail', 'corrupt'])
def test_failure_saves_root_cause_and_does_not_start_followups(benchmark, monkeypatch, tmp_path, fault):
    boundary = HttpBoundary(**{fault: True})
    install_boundary(monkeypatch, benchmark, boundary)
    output = tmp_path / 'failure'
    with pytest.raises((RuntimeError, ValueError), match='worker root cause|SHA256'):
        benchmark.run_benchmark('http://muse', output)
    ledger = json.loads((output / 'ledger.json').read_text())
    assert ledger['status'] == 'FAILED'
    assert len(boundary.runs) == 1
    assert all(r['state'] == 'NOT_STARTED' for r in ledger['runs'][1:])
    assert (output / 'failure.txt').read_text()


def test_http_failure_before_run_is_not_retried_and_keeps_schedule(benchmark, monkeypatch, tmp_path):
    calls = []

    def fail(request, timeout):
        calls.append(request)
        raise HTTPError(request.full_url, 422, 'validation root cause', {}, io.BytesIO(b'{"detail":"bad package"}'))

    monkeypatch.setattr(benchmark, 'urlopen', fail)
    output = tmp_path / 'http-error'
    with pytest.raises(HTTPError):
        benchmark.run_benchmark('http://muse', output)
    assert len(calls) == 1
    ledger = json.loads((output / 'ledger.json').read_text())
    assert len(ledger['runs']) == 90 and all(r['state'] == 'NOT_STARTED' for r in ledger['runs'])
    assert 'bad package' in (output / 'failure.txt').read_text()
