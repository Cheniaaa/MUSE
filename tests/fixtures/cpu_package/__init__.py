"""CPU契约fixture，非飞行场景。"""
from muse_sdk.registration import TermRegistration, TrainingPackageDefinition
from .terms import EmptyConfig, LimitConfig, ProgressConfig, ProgressReward, collision, time_limit
from .algorithms import CpuRule, AlternateRule


def build_training_package():
    role_output = {'shape': ['B', 'N_role'], 'dtype': 'float32', 'units': 'm', 'frame': 'world', 'rate': 'decision'}
    world_output = {'shape': ['B'], 'dtype': 'bool', 'units': '1', 'frame': 'none', 'rate': 'decision'}
    return TrainingPackageDefinition(
        terms={
            'progress': TermRegistration(ProgressReward, 'reward', 'role', ProgressConfig,
                ({'path': ['role_positions_world', {'binding': 'role_id'}], 'phase': 'post_physics',
                  'tensor': {**role_output, 'shape': ['B', 'N_role', 3]}},), role_output),
            'collision': TermRegistration(collision, 'termination', 'world', EmptyConfig, (), world_output),
            'time_limit': TermRegistration(time_limit, 'termination', 'world', LimitConfig, (), world_output, time_out=True),
        }, tasks={}, algorithms={'cpu_rule': CpuRule, 'alternate': AlternateRule},
        radio_models={}, sensing_models={},
    )
