"""唯一 driver 的交互边界；CPU 环境替身不推进真实物理。"""
from collections import deque
from types import SimpleNamespace as NS

import pytest
import torch

from muse_runtime.driver import run
from muse_sdk import run_protocol as wire
from muse_sdk.runtime_types import ActResult, PolicyInput, StepOutput


class Env:
    num_envs = 2
    device = 'cpu'
    rng = None

    def __init__(self, failure=None):
        self.steps = 0
        self.resets = []
        self.failure = failure
        self.policy_input = PolicyInput('initial', {})

    def step(self, actions):
        if self.failure == 'step':
            raise RuntimeError('physics failed')
        self.steps += 1
        return StepOutput(f'step{self.steps}', {}, self.steps, {})

    def reset(self, mask):
        if self.failure == 'reset':
            raise RuntimeError('physics failed')
        self.resets.append(mask.clone())
        self.policy_input = PolicyInput('reset', {'fresh': True})
        return self.policy_input


class Algorithm:
    def __init__(self):
        self.acts, self.transitions, self.resets = [], [], []

    def set_mode(self, mode):
        self.mode = mode

    def act(self, observations, masks, rng, deterministic):
        self.acts.append((observations, masks, deterministic))
        return ActResult({}, observations)

    def process_env_step(self, transition, aux):
        self.transitions.append((transition, aux))

    def reset_worlds(self, mask):
        self.resets.append(mask.clone())


def command(seq, name, world_id=None):
    assert hasattr(wire, 'InteractiveCommand'), 'InteractiveCommand missing'
    return wire.InteractiveCommand(command_seq=seq, command=name, world_id=world_id)


def drive(commands, *, env=None, mode='INTERACTIVE', limit=20):
    env, algorithm = env or Env(), Algorithm()
    queue, events, saved = deque(commands), [], []
    result = run(env, algorithm, lambda: None if queue else 'cancel',
        lambda kind, payload: events.append((kind, payload)), mode=mode,
        collection_steps=10, max_decision_steps=limit,
        command_reader=lambda: queue.popleft(), checkpoint_sink=lambda **kw: saved.append(kw))
    if mode != 'TRAIN':
        assert not saved
    return result, env, algorithm, events


def results(events):
    return [p for k, p in events if k == 'command_result']


def test_default_run_pause_step_reset_refresh_policy_and_ack_order():
    _, env, algo, events = drive([None, command(1, 'pause'), command(2, 'step'),
        command(3, 'reset_world', 1), command(4, 'step'), command(5, 'resume')])
    assert env.steps == 4
    assert algo.mode == 'EVALUATE'
    assert algo.acts[2] == ('reset', {'fresh': True}, True)
    assert len(algo.transitions) == 4
    assert env.resets[0].tolist() == algo.resets[0].tolist() == [False, True]
    assert [(r['steps'], r['paused'], r['updates']) for r in results(events)] == [
        (1, True, 0), (2, True, 0), (2, True, 0), (3, True, 0), (3, False, 0)]
    assert [k for k, _ in events] == ['step', 'command_result', 'step', 'command_result',
        'reset_world', 'command_result', 'step', 'command_result', 'command_result', 'step']


def test_retries_conflicts_gaps_and_old_sequences_do_not_repeat_effects():
    pause, step, reset = command(1, 'pause'), command(2, 'step'), command(3, 'reset_world', 0)
    result, env, algo, events = drive([command(2, 'pause'), pause, step, step,
        command(2, 'reset_world', 0), step, pause, command(4, 'resume'), reset, reset])
    replies = results(events)
    assert [r['status'] for r in replies] == ['rejected', 'applied', 'applied', 'applied',
        'rejected', 'applied', 'rejected', 'rejected', 'applied', 'applied']
    assert replies[2] == replies[3] == replies[5]
    assert replies[8] == replies[9]
    assert result == {'state': 'CANCELLED', 'steps': 2, 'updates': 0}
    assert len(env.resets) == len(algo.resets) == 1
    assert len(algo.transitions) == 2
    assert sum(k == 'reset_world' for k, _ in events) == 1


def test_invalid_state_consumes_sequence_and_bad_world_has_no_effect():
    _, env, _, events = drive([command(1, 'step'), command(2, 'reset_world', 0),
        command(3, 'pause'), command(4, 'reset_world', 2), command(5, 'step')])
    assert [r['status'] for r in results(events)] == ['rejected', 'rejected', 'applied', 'rejected', 'applied']
    assert env.steps == 3
    assert not env.resets


def test_paused_idle_services_control_and_yields_cpu(monkeypatch):
    waits = []
    monkeypatch.setattr('time.sleep', lambda duration: waits.append(duration))
    result, env, _, _ = drive([command(1, 'pause'), None, None, None])
    assert result['state'] == 'CANCELLED' and env.steps == 0
    assert len(waits) >= 3 and all(0 < wait <= .1 for wait in waits)


def test_cancel_has_priority_over_pending_command():
    events = []
    result = run(Env(), Algorithm(), lambda: 'cancel', lambda *e: events.append(e),
        mode='INTERACTIVE', collection_steps=None, max_decision_steps=2,
        command_reader=lambda: pytest.fail('cancel must not consume command'))
    assert result == {'state': 'CANCELLED', 'steps': 0, 'updates': 0}
    assert events == []


@pytest.mark.parametrize('mode', ['TRAIN', 'EVALUATE'])
def test_other_modes_reject_commands_without_reset(mode):
    _, env, _, events = drive([command(1, 'reset_world', 0)], mode=mode)
    assert results(events)[0]['status'] == 'rejected'
    assert env.steps == 1 and not env.resets


@pytest.mark.parametrize('failure,name,world', [('step', 'step', None), ('reset', 'reset_world', 0)])
def test_physical_errors_propagate_without_success_ack(failure, name, world):
    events, queue = [], deque([command(1, 'pause'), command(2, name, world)])
    with pytest.raises(RuntimeError, match='physics failed'):
        run(Env(failure), Algorithm(), lambda: None, lambda k, p: events.append((k, p)),
            mode='INTERACTIVE', collection_steps=None, max_decision_steps=3,
            command_reader=lambda: queue.popleft())
    assert len(results(events)) == 1


def test_last_allowed_step_acknowledged_after_transition():
    result, _, algo, events = drive([command(1, 'pause'), command(2, 'step')], limit=1)
    assert result['steps'] == len(algo.transitions) == 1
    assert results(events)[-1]['steps'] == 1
    assert results(events)[-1]['status'] == 'applied'


def test_preview_explicit_reset_invalidates_pending_frame_without_step():
    from muse_runtime.preview import WorkerPreview
    preview = WorkerPreview.__new__(WorkerPreview)
    preview.episodes = torch.tensor([4, 7])
    preview.pending = b'old'
    preview.next_sample = float('inf')
    assert hasattr(preview, 'after_reset'), 'Explicit preview reset hook missing'
    preview.after_reset(1)
    assert preview.episodes.tolist() == [4, 8]
    assert preview.pending is None and preview.next_sample == 0


@pytest.mark.parametrize('cancel_immediately', [False, True])
def test_worker_routes_ack_and_services_preview_while_paused(monkeypatch, cancel_immediately):
    """保留真实 worker/driver/preview，仅替换 GPU、包加载和物理边界。"""
    from contextlib import nullcontext
    import sys
    from muse_runtime import package_loader, preview, rng, worker

    events, packets = [], []
    experiment = NS(environment=NS(sha256='hash'), entity_models={},
        training_package=NS(sha256='hash'), root_seed=1, topology=NS(world_count=2),
        task_id=None, algorithm_id='test', algorithm_config={}, training_schedule=None, model_bindings={},
        preview=NS(world_ids=[0], rate_hz=10),
        scenario=NS(scene=NS(physics_dt=.01, control_substeps=2)))
    env = Env()
    env.experiment = experiment
    env.episode_length_buf = torch.tensor([5, 9])
    env.cfg, env.physics_dt = NS(decimation=2), .01
    env.sim = NS(mj_model=NS(nbody=1), data=NS(xpos=torch.ones(2, 1, 3),
                                            xmat=torch.ones(2, 1, 9)))
    env.close = lambda: None
    original_reset, original_step = env.reset, env.step

    def reset(mask):
        env.episode_length_buf[mask] = 0
        env.sim.data.xpos[mask] = 0
        return original_reset(mask)

    def step(actions):
        env.episode_length_buf.add_(1)
        return original_step(actions)

    env.reset, env.step = reset, step
    algorithm = Algorithm()
    algorithm.initialize = lambda *_: None
    package = NS(package_hash='hash', definition=NS(terms={}, algorithms={'test': lambda _: algorithm},
                                                  radio_models={}, sensing_models={}))
    monkeypatch.setattr(package_loader, 'hash_package', lambda _: 'hash')
    monkeypatch.setattr(package_loader, 'load_package', lambda _: nullcontext(package))
    monkeypatch.setitem(sys.modules, 'muse_runtime.env', NS(MuseEnv=lambda *_: env,
        RuntimeContext=lambda *_: NS(device='cpu')))
    monkeypatch.setitem(sys.modules, 'warp', NS(init=lambda: None, stream_from_torch=lambda _: None,
        ScopedStream=lambda _: nullcontext()))
    monkeypatch.setattr(rng, 'WorldRng', lambda *_: None)
    stream = NS(synchronize=lambda: None)
    monkeypatch.setattr(torch.cuda, 'Stream', lambda **_: stream)
    monkeypatch.setattr(torch.cuda, 'stream', lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda _: stream)
    monkeypatch.setattr(preview, 'check_preview_capacity', lambda *_: None)
    monkeypatch.setattr(preview, 'export_model', lambda *_:
        wire.PreviewModel(path='preview.mjb', sha256='a' * 64, nbody=1))
    monkeypatch.setenv('MUSE_PREVIEW_FD', '999')
    monkeypatch.setattr(worker.socket, 'socket', lambda **_: NS(setblocking=lambda _: None,
        send=lambda packet: packets.append(packet), close=lambda: None))
    monkeypatch.setattr('time.sleep', lambda _: None)
    queue = deque([command(1, 'pause'), command(2, 'reset_world', 1),
                   command(2, 'reset_world', 1), None, command(3, 'step'), None])
    selections = deque([wire.WorkerControl(run_id='run', kind='preview', selection_seq=1,
        display=wire.DisplaySettings(enabled=True, world_id=1, rate_hz=10))])
    control_polls = []

    def read_control():
        control_polls.append(env.steps)
        return 'cancel' if cancel_immediately or not queue else None

    protocol = NS(emit=lambda k, p=None, **_: events.append((k, p)), read_control=read_control,
        take_command=lambda: queue.popleft(),
        take_preview_controls=lambda: [selections.popleft()] if selections else [])
    plan = NS(experiment=experiment, device='cuda:0', run_id='run', mode='INTERACTIVE',
        artifact_directory='unused', resource_paths={'environment': 'env', 'training_package': 'pkg'},
        checkpoint=None, max_decision_steps=10, checkpoint_interval_updates=None)
    result = worker.execute(plan, protocol)
    if cancel_immediately:
        assert result == {'state': 'CANCELLED', 'steps': 0, 'updates': 0}
        assert not [item for item in events if item[0] in ('preview_applied', 'metric', 'command_result')]
        assert packets == []
        return
    assert result == {'state': 'CANCELLED', 'steps': 1, 'updates': 0}
    assert control_polls.count(0) >= 4
    replies = results(events)
    assert len(replies) == 4 and replies[1] == replies[2]
    assert replies[-1]['steps'] == 1
    assert [p['source'] for k, p in events if k == 'metric'] == ['step']
    frames = [wire.decode_preview(packet) for packet in packets]
    reset_frames = [(header, data) for header, data in frames if header.episode == 1]
    assert reset_frames and reset_frames[0][0].decision == 0
    assert all(header.world_id == 1 and header.episode <= 1 for header, _ in frames)
    import numpy as np
    assert not np.frombuffer(reset_frames[0][1], dtype='<f4')[:3].any()


def test_interactive_ippo_and_rule_reset_preserve_actions_and_learning():
    from test_returns import make_algorithm, transition
    from test_mappo import assert_tree_equal
    from muse_runtime.checkpoint import snapshot_learning_state
    from muse_builtins.rules import ConstantVelocity

    algorithm, obs, masks = make_algorithm()
    env = Env()
    env.policy_input = PolicyInput(obs, masks)
    env.reset = lambda _: env.policy_input
    env.step = lambda _: StepOutput(obs, masks, transition(obs, masks), {})
    before = snapshot_learning_state(algorithm.learning_state_dict())
    commands = deque([command(1, 'pause'), command(2, 'step'), command(3, 'reset_world', 0),
                      command(4, 'step')])
    result = run(env, algorithm, lambda: None if commands else 'cancel', lambda *_: None,
        mode='INTERACTIVE', collection_steps=None, max_decision_steps=10,
        command_reader=lambda: commands.popleft())
    assert result['steps'] == 2 and result['updates'] == 0
    assert_tree_equal(before, algorithm.learning_state_dict())
    assert all(c.storage.step == 0 and c.storage.pending is None for c in algorithm.children.values())
    first = algorithm.act(obs, masks, None, True)
    algorithm.reset_worlds(torch.tensor([False, True]))
    assert_tree_equal(first.role_actions, algorithm.act(obs, masks, None, True).role_actions)

    rule = ConstantVelocity({'motion': [1., 0., 0., 0.]})
    rule.initialize(NS(topology=NS(world_count=2), bindings={'role': NS(role_slot_count=1)}),
                    NS(device='cpu'), None)
    queue = deque([command(1, 'pause'), command(2, 'reset_world', 0), command(3, 'step')])
    result = run(env, rule, lambda: None if queue else 'cancel', lambda *_: None,
        mode='INTERACTIVE', collection_steps=None, max_decision_steps=10,
        command_reader=lambda: queue.popleft())
    assert result['steps'] == 1 and result['updates'] == 0
    assert rule.actions['role'].motion.tolist() == [[[1., 0., 0., 0.]], [[1., 0., 0., 0.]]]
