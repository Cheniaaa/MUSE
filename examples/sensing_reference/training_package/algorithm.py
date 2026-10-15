"""显式逐role开环控制；记录实际收到的局部measurement而非真值策略。"""
from pydantic import Field

from muse_sdk.config import DataModel
from muse_sdk.interfaces import Algorithm
from muse_sdk.runtime_types import ActResult, MotionAction, MotionRadioAction, RadioAction


class RoleControl(DataModel):
    motion: list[float] = Field(min_length=4, max_length=4)
    power_w: float = Field(ge=0)
    channel_id: int


class Phase(DataModel):
    duration_decisions: int = Field(gt=0)
    roles: dict[str, RoleControl]


class Config(DataModel):
    phases: list[Phase] = Field(min_length=1)


class SensingReferenceRule(Algorithm):
    def __init__(self, config):
        self.config = config

    @classmethod
    def describe(cls):
        return dict(algorithm_id='sensing_reference_rule', config_schema=Config, observation_requirements=[],
            action_requirements=['body_velocity', 'power_channel'], actor_observation={}, critic_inputs={},
            reward_inputs=['world'], cost_inputs=['world'], act_aux={}, diagnostics={}, training_schedule=None)

    def initialize(self, experiment, context, rng):
        import torch
        self.specs = experiment.action_specs
        phases = self.config['phases']
        if any(set(phase['roles']) != set(self.specs) for phase in phases):
            raise ValueError('Phase roles must match experiment roles')
        self.step = torch.zeros(experiment.topology.world_count, dtype=torch.int64, device=context.device)
        self.ends = torch.tensor([p['duration_decisions'] for p in phases], device=context.device).cumsum(0)
        self.motion, self.power, self.channel = {}, {}, {}
        for role in self.specs:
            controls = [phase['roles'][role] for phase in phases]
            self.motion[role] = torch.tensor([c['motion'] for c in controls], device=context.device)
            self.power[role] = torch.tensor([c['power_w'] for c in controls], device=context.device)
            self.channel[role] = torch.tensor([c['channel_id'] for c in controls], dtype=torch.int32, device=context.device)

    def set_mode(self, mode):
        if mode != 'EVALUATE':
            raise ValueError('SensingReferenceRule supports EVALUATE only')

    def act(self, observations, role_masks, rng, deterministic):
        # 策略输入来自action前；同一步结束的新measurement另记于收据。
        self.policy_observations = {role: {name: value.tolist() for name, value in fields.items()}
                                    for role, fields in observations.actors.items()}
        index = (self.step[:, None] >= self.ends[:-1]).sum(-1)
        actions = {}
        for role, spec in self.specs.items():
            motion = self.motion[role][index, None, :].expand(spec.shape)
            if spec.radio is None:
                actions[role] = MotionAction(motion)
            else:
                power = self.power[role][index, None].expand(spec.shape[:2])
                channel = self.channel[role][index, None].expand(spec.shape[:2])
                actions[role] = MotionRadioAction(motion, RadioAction(power, channel))
        return ActResult(actions, ())

    def process_env_step(self, transition, act_aux):
        import json
        wireless = transition.wireless
        data = dict(policy_observations=self.policy_observations,
            reset=transition.world_reset.tolist(),
            world_reward=None if transition.world_reward is None else transition.world_reward.tolist(),
            world_cost={name: value.tolist() for name, value in transition.costs['world'].items()},
            sensing={name: {field: value.tolist() for field, value in view._asdict().items()}
                     for name, view in wireless.sensing.items()},
            sample_tick=None if wireless.radio is None else wireless.radio_sample_tick.tolist(),
            interval_s=None if wireless.radio is None else wireless.radio_interval_s.tolist(),
            radio={} if wireless.radio is None else {
                name: None if block is None else {field: value.tolist() for field, value in block._asdict().items()}
                for name, block in wireless.radio._asdict().items()})
        print('SENSING_RESULT ' + json.dumps(data), flush=True)
        self.step.add_(1)
        self.reset_worlds(transition.world_reset)

    def reset_worlds(self, mask):
        self.step[mask] = 0
