from .terms import EmptyConfig
from muse_sdk.interfaces import Algorithm


class CpuRule(Algorithm):
    @classmethod
    def describe(cls):
        return {'algorithm_id': 'cpu_rule', 'config_schema': EmptyConfig,
                'observation_requirements': [], 'action_requirements': ['motion'],
                'actor_observation': {}, 'critic_inputs': {}, 'reward_inputs': [],
                'act_aux': {}, 'diagnostics': {}, 'training_schedule': None}


class AlternateRule(CpuRule):
    @classmethod
    def describe(cls):
        return {**super().describe(), 'algorithm_id': 'alternate'}
