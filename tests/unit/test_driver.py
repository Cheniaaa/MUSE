from types import SimpleNamespace

import pytest

from muse_sdk.runtime_types import ActResult, PolicyInput, StepOutput


class CountingEnv:
    def __init__(self, fail_at=None):
        self.steps = 0
        self.fail_at = fail_at
        self.rng = object()
        self.policy_input = PolicyInput('initial', {})

    def step(self, actions):
        if self.steps == self.fail_at:
            raise RuntimeError('physics failed')
        self.steps += 1
        return StepOutput(self.steps, {}, self.steps, {'step': self.steps})


class CountingAlgorithm:
    def __init__(self):
        self.transitions = []
        self.updates = []

    def set_mode(self, mode):
        self.mode = mode

    def act(self, observations, role_masks, rng, deterministic):
        return ActResult({}, observations)

    def process_env_step(self, transition, aux):
        self.transitions.append((transition, aux))

    def update(self):
        self.updates.append(len(self.transitions))
        return {'loss': 0.5}


def test_driver_collects_h_successful_transitions_then_updates():
    from muse_runtime.driver import run

    env, algorithm, events = CountingEnv(), CountingAlgorithm(), []
    result = run(env, algorithm, lambda: None, lambda kind, payload: events.append((kind, payload)),
                 mode='TRAIN', collection_steps=4, max_decision_steps=8)
    assert result == {'state': 'COMPLETED', 'steps': 8, 'updates': 2}
    assert algorithm.updates == [4, 8]
    assert algorithm.transitions[:2] == [(1, 'initial'), (2, 1)]
    assert [payload['steps'] for kind, payload in events if kind == 'update'] == [4, 8]


def test_cancel_partial_block_does_not_update():
    from muse_runtime.driver import run

    env, algorithm = CountingEnv(), CountingAlgorithm()
    result = run(env, algorithm, lambda: 'cancel' if env.steps == 3 else None, lambda *_: None,
                 mode='TRAIN', collection_steps=4, max_decision_steps=8)
    assert result == {'state': 'CANCELLED', 'steps': 3, 'updates': 0}
    assert algorithm.updates == []


def test_failed_physics_does_not_commit_transition_or_update():
    from muse_runtime.driver import run

    env, algorithm = CountingEnv(fail_at=2), CountingAlgorithm()
    with pytest.raises(RuntimeError, match='physics failed'):
        run(env, algorithm, lambda: None, lambda *_: None, mode='TRAIN', collection_steps=4, max_decision_steps=8)
    assert len(algorithm.transitions) == 2
    assert algorithm.updates == []


def test_rule_mode_does_not_require_update_method():
    from muse_runtime.driver import run

    source = CountingAlgorithm()
    rule = SimpleNamespace(act=source.act, set_mode=source.set_mode, process_env_step=source.process_env_step)
    result = run(CountingEnv(), rule, lambda: None, lambda *_: None,
                 mode='EVALUATE', collection_steps=None, max_decision_steps=5)
    assert result == {'state': 'COMPLETED', 'steps': 5, 'updates': 0}


@pytest.mark.parametrize('h', [None, 0, -1, 1.5, True])
def test_train_requires_positive_integer_h(h):
    from muse_runtime.driver import run

    with pytest.raises(ValueError, match='collection_steps'):
        run(CountingEnv(), CountingAlgorithm(), lambda: None, lambda *_: None,
            mode='TRAIN', collection_steps=h, max_decision_steps=8)


@pytest.mark.parametrize('cancel_at,expected', [
    (None, [(4, 1), (8, 2), (8, 2)]),
    (6, [(4, 1), (6, 1)]),
    (4, [(4, 0)]),
    (0, [(0, 0)]),
])
def test_checkpoint_snapshots_at_update_and_safe_end(cancel_at, expected):
    from muse_runtime.driver import run
    env, algorithm, snapshots = CountingEnv(), CountingAlgorithm(), []
    def snapshot(*, steps, updates):
        snapshots.append((steps, updates))
        assert updates == len(algorithm.updates)
    result = run(env, algorithm, lambda: 'cancel' if env.steps == cancel_at else None,
        lambda *_: None, mode='TRAIN', collection_steps=4, max_decision_steps=8,
        checkpoint_sink=snapshot, checkpoint_interval_updates=1)
    assert snapshots == expected
    assert result['state'] == ('COMPLETED' if cancel_at is None else 'CANCELLED')
    assert result['updates'] == expected[-1][1]


def test_checkpoint_default_only_saves_at_end_and_evaluate_never_saves():
    from muse_runtime.driver import run
    for mode, expected in [('TRAIN', [(8, 2)]), ('EVALUATE', [])]:
        saved = []
        run(CountingEnv(), CountingAlgorithm(), lambda: None, lambda *_: None,
            mode=mode, collection_steps=4, max_decision_steps=8,
            checkpoint_sink=lambda **counts: saved.append((counts['steps'], counts['updates'])))
        assert saved == expected


@pytest.mark.parametrize('failure', ['step', 'update', 'save'])
def test_failure_does_not_publish_a_successful_terminal_checkpoint(failure):
    from muse_runtime.driver import run
    env, algorithm, saved = CountingEnv(fail_at=6 if failure == 'step' else None), CountingAlgorithm(), []
    def fail():
        raise RuntimeError('injected failure')
    if failure == 'update':
        algorithm.update = fail
    def save(**counts):
        if failure == 'save':
            fail()
        saved.append(counts)
    with pytest.raises(RuntimeError):
        run(env, algorithm, lambda: None, lambda *_: None, mode='TRAIN', collection_steps=4,
            max_decision_steps=8, checkpoint_sink=save)
    assert saved == []
