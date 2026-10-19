"""冻结的交互 wire 契约与 CPU 控制队列。"""
from collections import deque
import threading

import pytest
from pydantic import ValidationError

from muse_sdk import run_protocol as wire
from preview_helpers import preview_control


def command_type():
    assert hasattr(wire, 'InteractiveCommand'), 'InteractiveCommand missing'
    return wire.InteractiveCommand


@pytest.mark.parametrize('payload', [
    {'command_seq': value, 'command': 'pause'} for value in (0, -1, True, 1.0, '1')
] + [
    {'command_seq': 1, 'command': 'reset_world', 'world_id': value}
    for value in (None, -1, True, 0.0, '0')
] + [
    {'command_seq': 1, 'command': name, 'world_id': 0}
    for name in ('pause', 'resume', 'step')
] + [{'command_seq': 1, 'command': 'unknown'}])
def test_command_rejects_invalid_identity_and_target(payload):
    cls = command_type()
    with pytest.raises(ValidationError):
        cls.model_validate(payload)


def test_commands_results_and_interactive_request_roundtrip():
    cls = command_type()
    assert hasattr(wire, 'CommandResult'), 'CommandResult missing'
    for name, world in [('pause', None), ('resume', None), ('step', None), ('reset_world', 0)]:
        command = cls(command_seq=1, command=name, world_id=world)
        control = wire.WorkerControl(run_id='run', kind='command', command=command)
        assert wire.WorkerControl.model_validate_json(control.model_dump_json()).command == command
        result = wire.CommandResult(**command.model_dump(), status='applied', paused=True, steps=2, updates=0)
        event = wire.WorkerEvent(run_id='run', event_seq=1, kind='command_result',
                                 sim_time=0, wall_time=0, payload=result.model_dump())
        assert event.payload['error'] is None
    request = wire.RunRequest(experiment_id='exp', mode='INTERACTIVE', max_decision_steps=10)
    assert request.mode == 'INTERACTIVE'
    with pytest.raises(ValidationError):
        wire.WorkerControl(run_id='run', kind='command')
    with pytest.raises(ValidationError):
        wire.WorkerControl(run_id='run', kind='cancel', command=command)


def test_protocol_keeps_commands_separate_and_cancel_priority(monkeypatch):
    from muse_runtime.protocol import WorkerProtocol
    cls = command_type()
    protocol = WorkerProtocol.__new__(WorkerProtocol)
    protocol.run_id = 'run'
    protocol._cancel = threading.Event()
    protocol._error = None
    protocol._control_lock = threading.Lock()
    protocol._preview_controls = deque()
    protocol._commands = deque()
    commands = [cls(command_seq=1, command='pause'), cls(command_seq=2, command='step')]
    controls = [wire.WorkerControl(run_id='run', kind='command', command=c) for c in commands]
    preview = preview_control('run', 1)
    controls.insert(1, preview)
    latest = preview_control('run', 101, world=1, rate=20)
    controls.extend(preview_control('run', seq) for seq in range(2, 101))
    controls.append(latest)
    controls.append(wire.WorkerControl(run_id='run', kind='cancel'))
    chunks = iter([('\n'.join(c.model_dump_json() for c in controls) + '\n').encode(), b''])
    monkeypatch.setattr('muse_runtime.protocol.os.read', lambda *_: next(chunks))
    protocol._read_controls()
    assert protocol._error is None
    assert protocol.read_control() == 'cancel'
    assert protocol.take_preview_controls() == [latest]
    assert protocol.take_preview_controls() == []
    assert protocol.take_command() == commands[0]
    assert protocol.take_command() == commands[1]
    assert protocol.take_command() is None


@pytest.mark.parametrize('hook', ['missing', None])
def test_package_algorithm_missing_reset_hook_rejected_before_run(hook):
    from dataclasses import replace
    from pathlib import Path
    from muse_runtime.package_loader import load_package
    from muse_sdk.config import ConfigError
    from muse_sdk.descriptor import build_descriptor

    with load_package(Path(__file__).parents[1] / 'fixtures/cpu_package') as package:
        class UserAlgorithm:
            describe = package.definition.algorithms['cpu_rule'].describe
        if hook is None:
            UserAlgorithm.reset_worlds = None
        definition = replace(package.definition, algorithms={'cpu_rule': UserAlgorithm})
        with pytest.raises(ConfigError, match='reset_worlds'):
            build_descriptor(definition, package.compositions, package.package_hash, default_composition_id=package.default_composition_id)


@pytest.mark.parametrize('inherit', [False, True])
def test_package_algorithm_reset_hook_supports_explicit_and_sdk_default(inherit):
    from dataclasses import replace
    from pathlib import Path
    from muse_runtime.package_loader import load_package
    from muse_sdk.descriptor import build_descriptor
    from muse_sdk.interfaces import Algorithm

    with load_package(Path(__file__).parents[1] / 'fixtures/cpu_package') as package:
        class UserAlgorithm(Algorithm if inherit else object):
            describe = package.definition.algorithms['cpu_rule'].describe
        if not inherit:
            def reset_worlds(self, mask):
                self.mask = mask
            UserAlgorithm.reset_worlds = reset_worlds
        definition = replace(package.definition, algorithms={'cpu_rule': UserAlgorithm})
        descriptor = build_descriptor(definition, package.compositions, package.package_hash, default_composition_id=package.default_composition_id)
        assert 'cpu_rule' in descriptor.algorithms
        instance = UserAlgorithm()
        instance.reset_worlds([True, False])
        if not inherit:
            assert instance.mask == [True, False]
