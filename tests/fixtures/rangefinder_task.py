"""仅Task读取测距的包内fixture。"""
from muse_sdk.config import DataModel
from muse_sdk.runtime_types import TaskUpdateResult


class EmptyConfig(DataModel):
    pass


class RangefinderTask:
    @classmethod
    def describe(cls):
        return dict(task_id='rangefinder_task', config_schema=EmptyConfig,
            input_fields=[dict(path=['sensor_outputs', 'scout_group', 'range', 'data'],
                phase='post_physics', tensor=dict(shape=['B', 'N_group', 1], dtype='float32',
                    units='m', frame='none', rate='decision'))],
            published_fields={}, role_status_bindings={}, learning_done_roles=[])

    def __init__(self, config, env):
        self.env = env
        sample = env.sensor_outputs['scout_group']['range']
        self.initial_valid = sample.valid.clone()
        self.reading = sample.data.clone()

    def reset(self, env_ids):
        self.reading[env_ids] = self.env.sensor_outputs['scout_group']['range'].data[env_ids]

    def update(self, env):
        self.reading.copy_(env.sensor_outputs['scout_group']['range'].data)
        return TaskUpdateResult(role_status={}, learning_done={}, diagnostics={})
