"""具名双槽的公开控制契约、真实管道合并与socket分流。"""
import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from muse_sdk import run_protocol as wire


@pytest.mark.parametrize('patch', [{}, {'enabled': True}, {'native': None},
    {'viser': {}, 'native': {}}, {'native': {'enabled': None}},
    {'native': {'world_id': -1}}, {'native': {'rate_hz': 0}}, {'native': {'rate_hz': 31}}])
def test_patch_requires_one_named_viewer(patch):
    with pytest.raises(ValidationError):
        wire.DisplayPatch.model_validate(patch)


@pytest.mark.parametrize('viewer,rate', [('viser', 1), ('native', 30)])
def test_named_patch_keeps_supplied_fields(viewer, rate):
    value = {viewer: {'enabled': True, 'world_id': 1, 'rate_hz': rate}}
    assert wire.DisplayPatch.model_validate(value).model_dump(exclude_unset=True) == value


def test_latest_preview_control_keeps_both_viewers():
    child = '''
import time
from muse_runtime.protocol import WorkerProtocol
p = WorkerProtocol('run')
p._control_thread.join(timeout=5)
assert p.read_control() == 'cancel'
p.emit('heartbeat', {'preview': [c.model_dump() for c in p.take_preview_controls()],
    'commands': [p.take_command().command_seq, p.take_command().command_seq]})
p.close()
'''
    def selection(seq, world, rate):
        return dict(selection_seq=seq, display=dict(enabled=True, world_id=world, rate_hz=rate))
    lines = [dict(run_id='run', kind='preview', preview=dict(viser=selection(1, 0, 10), native=selection(1, 0, 10))),
             dict(run_id='run', kind='command', command=dict(command_seq=1, command='pause')),
             dict(run_id='run', kind='preview', preview=dict(viser=selection(2, 1, 10), native=selection(2, 1, 7))),
             dict(run_id='run', kind='command', command=dict(command_seq=2, command='step')),
             dict(run_id='run', kind='preview', preview=dict(viser=selection(3, 0, 17), native=selection(2, 1, 7))),
             dict(run_id='run', kind='cancel')]
    result = subprocess.run([sys.executable, '-c', child], input=''.join(json.dumps(c) + '\n' for c in lines),
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)['payload']
    assert payload['commands'] == [1, 2]
    assert len(payload['preview']) == 1
    latest = payload['preview'][0]['preview']
    assert latest == {'viser': selection(3, 0, 17), 'native': selection(2, 1, 7)}


def test_drain_keeps_latest_per_viewer():
    assert hasattr(wire, 'PreviewSelection'), 'Dual preview selection missing'
    selections = {name: wire.PreviewSelection(selection_seq=seq,
        display=wire.DisplaySettings(enabled=True, world_id=world, rate_hz=10))
        for name, seq, world in [('viser', 3, 0), ('native', 2, 1)]}
    sender, receiver = wire.preview_socketpair()
    try:
        for name, seq, world, episode, decision in [
            ('viser', 2, 0, 50, 10), ('native', 2, 0, 99, 10),
            ('viser', 3, 0, 1, 10), ('native', 2, 1, 3, 4),
            ('viser', 3, 0, 2, 0), ('native', 2, 1, 2, 40)]:
            header = wire.PreviewFrame(viewer=name, run_id='run', selection_seq=seq,
                world_id=world, episode=episode, decision=decision, tick=decision,
                sim_time=decision * .01, scene_hash='a' * 64, nbody=1)
            sender.send(wire.encode_preview(header, bytes(48)))
        frames = wire.drain_preview(receiver, run_id='run', selections=selections, scene_hash='a' * 64)
        assert {name: (value[0].episode, value[0].decision) for name, value in frames.items()} == {
            'viser': (2, 0), 'native': (3, 4)}
        assert wire.drain_preview(receiver, run_id='run', selections=selections, scene_hash='a' * 64) == {}
    finally:
        sender.close()
        receiver.close()


def test_worker_two_slots_bound_backlog_and_reset(monkeypatch, tmp_path):
    """慢消费者恢复后两槽仍前进，reset丢弃旧episode，而非阻塞driver。"""
    from types import SimpleNamespace as NS
    import torch
    from muse_runtime import preview as module

    sender, receiver = wire.preview_socketpair()
    env = NS(num_envs=2, device='cpu', experiment=NS(preview=NS(world_ids=[0], rate_hz=10)),
             sim=NS(mj_model=NS(nbody=1024), data=NS(xpos=torch.ones(2, 1024, 3),
                 xmat=torch.ones(2, 1024, 9))), scene=None,
             episode_length_buf=torch.tensor([5, 8]), cfg=NS(decimation=2), physics_dt=.01)
    monkeypatch.setattr(module, 'scene_objects', lambda *_: [])
    monkeypatch.setattr(module, 'export_model', lambda *_, **__: wire.PreviewModel(
        path='preview.mjb', sha256='a' * 64, nbody=1024, objects=[]))
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda _: NS(synchronize=lambda: None))
    events = []
    try:
        preview = module.WorkerPreview('run', env, tmp_path, sender, lambda k, p: events.append((k, p)))
        choices = wire.PreviewSelections(**{name: wire.PreviewSelection(selection_seq=1,
            display=wire.DisplaySettings(enabled=True, world_id=world, rate_hz=30))
            for name, world in [('viser', 0), ('native', 1)]})
        preview.apply(wire.WorkerControl(run_id='run', kind='preview', preview=choices))
        assert {p['viewer'] for k, p in events if k == 'preview_applied'} == {'viser', 'native'}
        for _ in range(30):
            preview.sample(force=True)
        env.episode_length_buf[1] = 0
        preview.after_reset(1)
        current = {name: getattr(choices, name) for name in ('viser', 'native')}
        wire.drain_preview(receiver, run_id='run', selections=current, scene_hash='a' * 64)
        preview.sample(force=True)
        frames = wire.drain_preview(receiver, run_id='run', selections=current, scene_hash='a' * 64)
        assert {name: (h.episode, h.decision) for name, (h, _) in frames.items()} == {
            'viser': (0, 5), 'native': (1, 0)}
        for _ in range(30):
            preview.sample(force=True)
        # 缓存只有两个最新包；限制缓冲、取消仍由调用方的安全边界消费。
        assert len([packet for packet in preview.pending.values() if packet is not None]) <= 2
        choices = choices.model_copy(deep=True)
        for choice in (choices.viser, choices.native):
            choice.selection_seq += 1
            choice.display.enabled = False
        preview.apply(wire.WorkerControl(run_id='run', kind='preview', preview=choices))
        monkeypatch.setattr(torch.Tensor, 'cpu', lambda *_: pytest.fail('disabled copied to CPU'))
        preview.sample(force=True)
    finally:
        sender.close()
        receiver.close()
