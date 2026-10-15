"""无学习状态的规则EVALUATE动作源。"""
from pydantic import Field

from muse_sdk.config import DataModel
from muse_sdk.interfaces import Algorithm
from muse_sdk.runtime_types import ActResult, MotionAction


class ConstantVelocityConfig(DataModel):
    motion: list[float] = Field(min_length=4, max_length=4)


class ConstantVelocity(Algorithm):
    def __init__(self, config):
        self.config = config

    @classmethod
    def describe(cls):
        return dict(algorithm_id='constant_velocity', config_schema=ConstantVelocityConfig,
                    observation_requirements=[], action_requirements=['body_velocity'],
                     actor_observation={}, critic_inputs={}, reward_inputs=[], act_aux={}, diagnostics={},
                    training_schedule=None)

    def initialize(self, experiment, context, rng):
        import torch
        motion = torch.tensor(self.config['motion'], dtype=torch.float32, device=context.device)
        self.actions = {role: MotionAction(motion.expand(experiment.topology.world_count,
                        binding.role_slot_count, 4).clone()) for role, binding in experiment.bindings.items()}

    def set_mode(self, mode):
        if mode != 'EVALUATE':
            raise ValueError('ConstantVelocity only supports EVALUATE')

    def act(self, observations, role_masks, rng, deterministic):
        return ActResult(self.actions, ())

    def process_env_step(self, transition, act_aux):
        # 规则源没有episode state或学习storage。
        return None
