"""验收判错与执行门的 CPU 测试；不创建 Server/Worker/Run。"""
import importlib
import json
from types import SimpleNamespace

import pytest


class SnapshotClient:
    def __init__(self, *snapshots):
        self.snapshots = iter(snapshots)
        self.paths = []

    def get(self, path):
        self.paths.append(path)
        value = next(self.snapshots)
        return SimpleNamespace(status_code=200, text=json.dumps(value), json=lambda: value)


@pytest.mark.parametrize('consumer', ['test_native_acceptance', 'test_recording_acceptance'])
@pytest.mark.parametrize('fresh', [
    {'state': 'COMPLETED', 'worker_terminal': {'state': 'COMPLETED'}},
    {'state': 'CANCELLED', 'worker_terminal': {'state': 'CANCELLED'}},
    {'state': 'FAILED', 'worker_terminal': None},
    {'state': 'RUNNING', 'worker_terminal': {'state': 'COMPLETED'}},
])
def test_terminal_readback_prevents_false_native_error(fresh, consumer):
    confirmed_display_error = importlib.import_module(consumer).confirmed_display_error
    client = SnapshotClient({'state': 'RUNNING', 'worker_terminal': None}, fresh)
    stale = client.get('/runs/r').json()
    assert stale['state'] == 'RUNNING'
    display = {'native': {'status': 'error', 'error': 'Native released after Worker exit'}}
    assert confirmed_display_error(client, 'r', display, ('native',)) is None
    assert client.paths == ['/runs/r', '/runs/r']


def test_running_readback_preserves_real_display_error():
    from acceptance_helpers import confirmed_display_error
    client = SnapshotClient({'state': 'RUNNING', 'worker_terminal': None})
    display = {'native': {'status': 'error', 'error': 'GLFW context lost'},
               'viser': {'status': 'ready', 'error': None}}
    assert confirmed_display_error(client, 'r', display, ('viser', 'native')) == 'native: GLFW context lost'
    assert client.paths == ['/runs/r']


def test_healthy_or_unselected_display_does_not_request_extra_snapshot():
    from acceptance_helpers import confirmed_display_error
    client = SnapshotClient()
    display = {'native': {'status': 'error', 'error': 'not selected'},
               'viser': {'status': 'ready', 'error': None}}
    assert confirmed_display_error(client, 'r', display, ('viser',)) is None
    assert client.paths == []


@pytest.mark.parametrize('environment,case,allowed', [
    ({}, 'train', False),
    ({'MUSE_TEST_GPU': '1'}, 'train', False),
    ({'MUSE_RECORDING_CASE': 'train'}, 'train', False),
    ({'MUSE_TEST_GPU': '1', 'MUSE_RECORDING_CASE': 'train'}, 'train', False),
    ({'MUSE_TEST_GPU': '1', 'MUSE_RECORDING_CASE': 'train', 'MUSE_TEST_RECORDING_RENDER': '1'}, 'train', True),
    ({'MUSE_TEST_GPU': '1', 'MUSE_RECORDING_CASE': 'train', 'MUSE_TEST_RECORDING_RENDER': '1'}, 'evaluate', False),
])
def test_case_gate_is_pure_and_requires_single_explicit_authorization(environment, case, allowed):
    module = importlib.import_module('test_recording_acceptance')
    assert module.case_authorized(case, environment) is allowed


@pytest.mark.parametrize('case', ['train', 'evaluate', 'interactive'])
def test_unapproved_entry_skips_before_creating_server_or_run(case, monkeypatch, tmp_path):
    module = importlib.import_module('test_recording_acceptance')
    for name in ('MUSE_TEST_GPU', 'MUSE_RECORDING_CASE', 'MUSE_TEST_RECORDING_RENDER'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(module, 'EVIDENCE', tmp_path / 'must-not-exist')

    def forbidden(*args, **kwargs):
        pytest.fail('Unapproved entry reached process/network creation')

    monkeypatch.setattr(module.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(module.httpx2, 'Client', forbidden)
    with pytest.raises(pytest.skip.Exception):
        module.test_recording_acceptance(case)
    assert not module.EVIDENCE.exists()


def test_attempt_reservation_is_permanent_and_does_not_overwrite_failure(tmp_path):
    module = importlib.import_module('test_recording_acceptance')
    module.reserve_attempt(tmp_path, 'train', {'case': 'train', 'attempt': 1, 'backend_commit': 'baseline', 'budget': 128})
    original = (tmp_path / 'attempts/train.json').read_bytes()
    (tmp_path / 'train-result.json').write_text('{"test_error": "original failure"}')
    with pytest.raises(FileExistsError):
        module.reserve_attempt(tmp_path, 'train', {'backend_commit': 'new', 'budget': 128})
    assert (tmp_path / 'attempts/train.json').read_bytes() == original
    assert json.loads((tmp_path / 'train-result.json').read_text())['test_error'] == 'original failure'
    assert len((tmp_path / 'attempts.jsonl').read_text().splitlines()) == 1


def test_stale_observer_ack_is_rejected_without_new_requests():
    module = importlib.import_module('test_recording_acceptance')
    source = dict(case='train', stage='review', catalog_id='catalog', instance_id='new-server', run_id='run')
    ack = dict(source, instance_id='old-server', success=True)
    with pytest.raises(AssertionError, match='identity'):
        module.validate_observer_ack(ack, source)
