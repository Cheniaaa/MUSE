"""交错slot编码必须按最后轴切分，RF held请求不推进仿真。"""
from types import SimpleNamespace as NS

import pytest
import torch

from muse_runtime import actions
from muse_sdk.runtime_types import MotionRadioAction, RadioAction
from test_hybrid_actions import make_spec


@pytest.mark.parametrize('width', [2, 4])
def test_radio_slices_preserve_interleaved_role_slots(width, monkeypatch):
    spec = make_spec(width=width)
    entries = [NS(role_id='pilot', field='motion', start=0, stop=3*width, shape=spec.shape),
               NS(role_id='pilot', field='radio', start=3*width, stop=3*(width+2), shape=[2, 3, 2])]
    experiment = NS(action_specs={'pilot': spec}, action_layout=entries,
                    bindings={'pilot': NS(role_slot_to_group_slot=[2, 0, 1])})
    motion = torch.linspace(-1, 1, 6*width).reshape(2, 3, width)
    request = RadioAction(torch.tensor([[.1, .2, .3], [.4, .5, .6]]),
                          torch.tensor([[7, -3, 19], [19, 7, -3]], dtype=torch.int32))
    calls = []
    original = actions.encode_role_action

    def encode(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(actions, 'encode_role_action', encode)
    packed = actions.encode_actions(experiment, {'pilot': MotionRadioAction(motion, request)}, 'cpu')
    assert packed.shape == (2, 3*(width+2))
    torch.testing.assert_close(packed[:, :3*width], motion.flatten(1))
    radio = packed[:, 3*width:].reshape(2, 3, 2)
    torch.testing.assert_close(radio[..., 0], request.power_w)
    torch.testing.assert_close(radio[..., 1], request.channel.float())
    assert len(calls) == 1


def test_radio_term_holds_requests_and_resets_only_selected_world():
    assert hasattr(actions, 'RadioActionTermCfg'), '需要实际RadioActionTerm'
    env = NS(num_envs=2, device='cpu', scene={'vehicle': object()},
             experiment=NS(action_specs={'pilot': make_spec()}))
    term = actions.RadioActionTermCfg(entity_name='vehicle', role_id='pilot').build(env)
    packed = torch.tensor([[.1, 7, .2, -3, .3, 19], [.4, 19, .5, 7, .6, -3]])
    term.process_actions(packed)
    pointer = term.request.power_w.data_ptr(), term.request.channel.data_ptr()
    for _ in range(3):
        term.apply_actions()
    torch.testing.assert_close(term.request.power_w, packed.reshape(2, 3, 2)[..., 0])
    assert term.request.channel.dtype == torch.int32
    term.reset(torch.tensor([0]))
    torch.testing.assert_close(term.request.power_w, torch.tensor([[0., 0., 0.], [.4, .5, .6]]))
    assert term.request.channel.tolist() == [[7, 7, 7], [19, 7, -3]]
    assert pointer == (term.request.power_w.data_ptr(), term.request.channel.data_ptr())
