"""执行进程共享的轻量容器；tensor由Worker创建，SDK不导入Torch。"""
from typing import Any, NamedTuple
from .network import WirelessSnapshot


class SensorSample(NamedTuple):
    """group 级借用只读测量；采样 tick 与 age 均属于当前 world episode。"""
    data: Any
    valid: Any
    age: Any
    sample_tick: Any


class MotionAction(NamedTuple):
    motion: Any


class RadioAction(NamedTuple):
    """power_w 为 float32[B,N]，channel 为 int32[B,N]；codec 负责验证。"""
    power_w: Any
    channel: Any


class MotionRadioAction(NamedTuple):
    motion: Any
    radio: RadioAction


RoleActions = dict[str, MotionAction | MotionRadioAction]


class ActResult(NamedTuple):
    role_actions: RoleActions
    act_aux: Any


class Observations(NamedTuple):
    actors: dict[str, dict[str, Any]]
    critic_inputs: dict[str, dict[str, Any]]


class RoleMask(NamedTuple):
    present: Any
    alive: Any
    controllable: Any


RoleMasks = dict[str, RoleMask]


class TaskUpdateResult(NamedTuple):
    """本阶段 role_status 必须为空；learning_done 不改变物理状态。"""
    role_status: dict[str, Any]
    learning_done: dict[str, Any]
    diagnostics: dict[str, Any]


class PolicyInput(NamedTuple):
    observations: Observations
    role_masks: RoleMasks


class TransitionView(NamedTuple):
    act_masks: RoleMasks
    pre_reset_next_observations: Observations
    rewards: dict[str, Any]
    world_reward: Any
    world_terminated: Any
    world_truncated: Any
    world_end_reason: Any
    learning_done: dict[str, Any]
    world_reset: Any
    costs: dict[str, Any]
    wireless: WirelessSnapshot | None


class StepOutput(NamedTuple):
    policy_observations: Observations
    policy_role_masks: RoleMasks
    transition: TransitionView
    diagnostics: dict[str, Any]
