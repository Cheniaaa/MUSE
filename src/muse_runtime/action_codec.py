"""Role 动作的唯一无状态编码边界；物理单位原样保存，不 round/clip。"""
import torch

from muse_sdk.experiment import ActionSpec
from muse_sdk.runtime_types import MotionAction, MotionRadioAction, RadioAction


def _tensor(value, shape, dtype, device, name):
    if (not isinstance(value, torch.Tensor) or list(value.shape) != list(shape)
            or value.dtype != dtype or value.device != device):
        raise ValueError(f'Invalid {name} shape/dtype/device')
    if not torch.isfinite(value).all():
        raise ValueError(f'Invalid {name} finite values')


def _bounds(value, low, high, name):
    low = torch.as_tensor(low, dtype=value.dtype, device=value.device)
    high = torch.as_tensor(high, dtype=value.dtype, device=value.device)
    if ((value < low) | (value > high)).any():
        raise ValueError(f'Invalid {name} bounds')


def _channel(spec, channel):
    # 相等比较发生在任何整数转换之前，不把非法 float channel 截断为合法 ID。
    ids = torch.tensor(spec.radio.channel_ids, dtype=channel.dtype, device=channel.device)
    if not (channel.unsqueeze(-1) == ids).any(dim=-1).all():
        raise ValueError('Invalid channel ID: expected an exact integer from channel_ids')


def encode_role_action(
    spec: ActionSpec, action: MotionAction | MotionRadioAction, device: str | torch.device,
) -> torch.Tensor:
    """返回 float32[B,N,width(+2)]，顺序 motion、power_w、实际 channel ID。"""
    expected = MotionAction if spec.radio is None else MotionRadioAction
    if not isinstance(action, expected):
        raise ValueError('Action variant does not match radio spec')
    device = torch.device(device)
    _tensor(action.motion, spec.shape, torch.float32, device, 'motion')
    _bounds(action.motion, spec.minimum, spec.maximum, 'motion')
    if spec.radio is None:
        return action.motion
    if not isinstance(action.radio, RadioAction):
        raise ValueError('Invalid radio action variant')
    _tensor(action.radio.power_w, spec.shape[:2], torch.float32, device, 'power_w')
    _tensor(action.radio.channel, spec.shape[:2], torch.int32, device, 'channel')
    _bounds(action.radio.power_w, spec.radio.power_minimum_w, spec.radio.power_maximum_w, 'power_w')
    _channel(spec, action.radio.channel)
    return torch.cat((action.motion, action.radio.power_w.unsqueeze(-1),
                      action.radio.channel.to(torch.float32).unsqueeze(-1)), dim=-1)


def decode_role_action(spec: ActionSpec, packed: torch.Tensor) -> MotionAction | MotionRadioAction:
    """验证 packed 后解码；输出 leaf 与 packed 同 device，channel 恢复 int32。"""
    shape = [*spec.shape[:2], spec.shape[2] + (2 if spec.radio is not None else 0)]
    if not isinstance(packed, torch.Tensor):
        raise ValueError('Invalid packed tensor')
    _tensor(packed, shape, torch.float32, packed.device, 'packed')
    motion = packed[..., :spec.shape[2]]
    _bounds(motion, spec.minimum, spec.maximum, 'motion')
    if spec.radio is None:
        return MotionAction(motion)
    power, channel = packed[..., -2], packed[..., -1]
    _bounds(power, spec.radio.power_minimum_w, spec.radio.power_maximum_w, 'power_w')
    _channel(spec, channel)
    return MotionRadioAction(motion, RadioAction(power, channel.to(torch.int32)))
