"""只替代 Viser transport；按钮回调及状态转换使用正式实现。"""
from types import SimpleNamespace

import pytest


class Gui:
    def __init__(self):
        self.controls = {}

    def add_button(self, label, *, disabled=False):
        handle = SimpleNamespace(disabled=disabled, callback=None)
        handle.on_click = lambda callback: setattr(handle, 'callback', callback)
        self.controls[label] = handle
        return handle

    def add_text(self, label, *, initial_value, disabled):
        handle = SimpleNamespace(value=initial_value, disabled=disabled)
        self.controls[label] = handle
        return handle


@pytest.mark.parametrize('interactive', [False, True])
def test_controls_only_interactive(interactive):
    from muse_server.viewers.viser_process import create_interaction_controls
    gui = Gui()
    create_interaction_controls(gui, SimpleNamespace(value='1'), lambda message: None,
                                interactive=interactive)
    assert set(gui.controls) == ({'Pause', 'Resume', 'Step', 'Reset world', 'Interaction'} if interactive else set())


def test_controls_pending_ack_error_and_selected_world():
    from muse_server.viewers.viser_process import create_interaction_controls
    gui, messages = Gui(), []
    world = SimpleNamespace(value='1')
    update = create_interaction_controls(gui, world, messages.append, interactive=True)
    gui.controls['Pause'].callback(SimpleNamespace(client_id=1))
    assert messages == [{'kind': 'command', 'command': 'pause', 'world_id': None}]
    assert 'pending' in gui.controls['Interaction'].value
    assert all(gui.controls[name].disabled for name in ('Pause', 'Resume', 'Step', 'Reset world'))
    # 已在 pipe 中的旧显示 state 不能把刚发出的命令显示成已生效。
    update({'kind': 'state', 'state': {}, 'interaction': {'command': None, 'result': None}})
    assert 'pending' in gui.controls['Interaction'].value
    assert gui.controls['Pause'].disabled
    interaction = {'command': {'command_seq': 1, 'command': 'pause', 'world_id': None}, 'result': None}
    update({'kind': 'state', 'state': {}, 'interaction': interaction})
    assert 'pending' in gui.controls['Interaction'].value
    interaction['result'] = dict(command_seq=1, command='pause', world_id=None,
                                 status='applied', paused=True, steps=7, updates=0, error=None)
    update({'kind': 'state', 'state': {}, 'interaction': interaction})
    assert 'applied' in gui.controls['Interaction'].value and 'paused' in gui.controls['Interaction'].value
    assert gui.controls['Pause'].disabled and not gui.controls['Step'].disabled
    gui.controls['Reset world'].callback(SimpleNamespace(client_id=1))
    assert messages[-1] == {'kind': 'command', 'command': 'reset_world', 'world_id': 1}
    update({'kind': 'command_error', 'message': 'Run has ended'})
    assert 'Run has ended' in gui.controls['Interaction'].value
    gui.controls['Step'].callback(SimpleNamespace(client_id=1))
    assert messages[-1] == {'kind': 'command', 'command': 'step', 'world_id': None}
    interaction = {'command': {'command_seq': 2, 'command': 'step', 'world_id': None},
                   'result': dict(command_seq=2, command='step', world_id=None, status='rejected',
                                  paused=False, steps=7, updates=0, error='step requires pause')}
    update({'kind': 'state', 'state': {}, 'interaction': interaction})
    assert 'rejected' in gui.controls['Interaction'].value and 'step requires pause' in gui.controls['Interaction'].value
    assert gui.controls['Step'].disabled and not gui.controls['Pause'].disabled
