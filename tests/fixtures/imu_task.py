"""仅Task消费IMU的真实包fixture；构造时公开缓存尚未采样。"""
from muse_sdk.config import DataModel
from muse_sdk.runtime_types import TaskUpdateResult


class EmptyConfig(DataModel):
    pass


class ImuTask:
    @classmethod
    def describe(cls):
        return dict(task_id='imu_task', config_schema=EmptyConfig,
            input_fields=[dict(path=['sensor_outputs', 'scout_group', 'gyro', 'data'],
                phase='post_physics', tensor=dict(shape=['B', 'N_group', 3], dtype='float32',
                    units='rad/s', frame='imu_flu', rate='decision'))],
            published_fields={}, role_status_bindings={}, learning_done_roles=[])

    def __init__(self, config, env):
        self.env = env
        sample = env.sensor_outputs['scout_group']['gyro']
        self.initial_valid = sample.valid.clone()
        self.reading = sample.data.clone()

    def reset(self, env_ids):
        self.reading[env_ids] = self.env.sensor_outputs['scout_group']['gyro'].data[env_ids]

    def update(self, env):
        self.reading.copy_(env.sensor_outputs['scout_group']['gyro'].data)
        return TaskUpdateResult(role_status={}, learning_done={}, diagnostics={})
