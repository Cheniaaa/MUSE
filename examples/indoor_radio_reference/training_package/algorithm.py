"""配置驱动的开环动作；无线真值仅用于诊断收据。"""
from typing import Annotated

from pydantic import Field

from muse_sdk.config import DataModel
from muse_sdk.interfaces import Algorithm
from muse_sdk.runtime_types import ActResult, MotionAction, MotionRadioAction, RadioAction


class Phase(DataModel):
    duration_decisions: int = Field(gt=0)
    motion: list[float] = Field(min_length=4, max_length=4)
    power_w: float = Field(ge=0)
    channel_id: int


class Config(DataModel):
    phases: list[Phase] = Field(min_length=1)
    world_power_scales: list[Annotated[float, Field(ge=0)]] = Field(min_length=1)


class IndoorRadioRule(Algorithm):
    def __init__(self, config):
        self.config = config

    @classmethod
    def describe(cls):
        return dict(algorithm_id='indoor_radio_rule', config_schema=Config, observation_requirements=[],
            action_requirements=['body_velocity', 'power_channel'], actor_observation={}, critic_inputs={},
            reward_inputs=[], act_aux={}, diagnostics={}, training_schedule=None)

    def initialize(self, experiment, context, rng):
        import torch
        self.specs = experiment.action_specs
        b = experiment.topology.world_count
        if len(self.config['world_power_scales']) != b:
            raise ValueError('world_power_scales must match world_count')
        phases = self.config['phases']
        self.step = torch.zeros(b, dtype=torch.int64, device=context.device)
        self.ends = torch.tensor([p['duration_decisions'] for p in phases], device=context.device).cumsum(0)
        self.motion = torch.tensor([p['motion'] for p in phases], device=context.device)
        self.power = torch.tensor([p['power_w'] for p in phases], device=context.device)
        self.channel = torch.tensor([p['channel_id'] for p in phases], dtype=torch.int32, device=context.device)
        self.scales = torch.tensor(self.config['world_power_scales'], device=context.device)

    def set_mode(self, mode):
        if mode != 'EVALUATE':
            raise ValueError('IndoorRadioRule supports EVALUATE only')

    def act(self, observations, role_masks, rng, deterministic):
        index = (self.step[:, None] >= self.ends[:-1]).sum(-1)
        actions = {}
        for role, spec in self.specs.items():
            motion = self.motion[index, None, :].expand(spec.shape)
            if spec.radio is None:
                actions[role] = MotionAction(motion)
            else:
                power = (self.power[index] * self.scales)[:, None].expand(spec.shape[:2])
                channel = self.channel[index, None].expand(spec.shape[:2])
                actions[role] = MotionRadioAction(motion, RadioAction(power, channel))
        return ActResult(actions, ())

    def process_env_step(self, transition, act_aux):
        import json
        wireless = transition.wireless
        data = dict(reset=transition.world_reset.tolist(), sample_tick=wireless.radio_sample_tick.tolist(),
            interval_s=wireless.radio_interval_s.tolist(),
            radio={name: {field: value.tolist() for field, value in block._asdict().items()}
                   for name, block in wireless.radio._asdict().items()},
            services={name: dict(remaining=value.remaining_bits.tolist(), served=value.served_bits.tolist(),
                start=value.interval_start_tick.tolist(), end=value.interval_end_tick.tolist())
                for name, value in wireless.services.items()})
        print('INDOOR_RADIO_RESULT ' + json.dumps(data), flush=True)
        self.step.add_(1)
        self.reset_worlds(transition.world_reset)

    def reset_worlds(self, mask):
        self.step[mask] = 0
