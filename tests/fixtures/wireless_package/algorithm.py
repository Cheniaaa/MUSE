"""真实Algorithm接口的规则动作与pre-reset验收收据。"""
from muse_sdk.config import DataModel
from muse_sdk.interfaces import Algorithm
from muse_sdk.runtime_types import ActResult, MotionAction, MotionRadioAction, RadioAction


class Config(DataModel):
    motion: list[float]
    power_w: float
    channel_id: int
    world_power_scales: list[float]


class WirelessRule(Algorithm):
    def __init__(self, config):
        self.config = config

    @classmethod
    def describe(cls):
        return dict(algorithm_id='wireless_rule', config_schema=Config, observation_requirements=[],
            action_requirements=['body_velocity', 'power_channel'], actor_observation={}, critic_inputs={},
            reward_inputs=[], act_aux={}, diagnostics={}, training_schedule=None)

    def initialize(self, experiment, context, rng):
        import torch
        self.actions = {}
        for role, spec in experiment.action_specs.items():
            motion = torch.tensor(self.config['motion'], device=context.device).expand(spec.shape).clone()
            if spec.radio is None:
                self.actions[role] = MotionAction(motion)
            else:
                power = torch.tensor(self.config['world_power_scales'], device=context.device)[:, None].expand(spec.shape[:2]).clone()
                power *= self.config['power_w']
                channel = torch.full(spec.shape[:2], self.config['channel_id'], dtype=torch.int32, device=context.device)
                self.actions[role] = MotionRadioAction(motion, RadioAction(power, channel))

    def set_mode(self, mode):
        assert mode == 'EVALUATE'

    def act(self, observations, role_masks, rng, deterministic):
        return ActResult(self.actions, ())

    def process_env_step(self, transition, act_aux):
        import json
        wireless = transition.wireless
        assert wireless is not None
        data = dict(reset=transition.world_reset.tolist(), sample_tick=(
            wireless.radio_sample_tick.tolist() if wireless.radio is not None else None),
            services={name: dict(remaining=value.remaining_bits.tolist(), served=value.served_bits.tolist(),
                start=value.interval_start_tick.tolist(), end=value.interval_end_tick.tolist())
                for name, value in wireless.services.items()})
        print('WIRELESS_ORACLE ' + json.dumps(data), flush=True)
