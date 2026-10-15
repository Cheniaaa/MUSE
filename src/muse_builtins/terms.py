"""普通飞行计算项，仅读取声明的env缓存。"""
from typing import Literal

from pydantic import Field

from muse_sdk.config import DataModel
from muse_sdk.registration import TermRegistration


class EmptyConfig(DataModel):
    pass


class DistanceConfig(DataModel):
    scale: float = 1.0


class CollisionConfig(DataModel):
    controlled_role: str
    reduction: Literal['any'] = 'any'


class TimeoutConfig(DataModel):
    episode_max_steps: int = Field(gt=0, description='每个world的回合最大决策步数；碰撞等终止条件可使回合提前结束')


def relative_target(env, *, role_id):
    return env.commands.target_world[role_id] - env.role_positions_world[role_id]


def velocity_world(env, *, role_id):
    return env.physics_outputs.velocity_world[role_id]


def orientation_wxyz(env, *, role_id):
    return env.physics_outputs.orientation_wxyz[role_id]


def angular_velocity_body(env, *, role_id):
    return env.physics_outputs.angular_velocity_body[role_id]


def imu_gyro(env, *, role_id):
    binding = env.experiment.bindings[role_id]
    return env.sensor_outputs[binding.group_id]['gyro'].data[:, binding.role_slot_to_group_slot]


def imu_specific_force(env, *, role_id):
    binding = env.experiment.bindings[role_id]
    return env.sensor_outputs[binding.group_id]['specific_force'].data[:, binding.role_slot_to_group_slot]


def imu_valid(env, *, role_id):
    import torch
    binding = env.experiment.bindings[role_id]
    samples = env.sensor_outputs[binding.group_id]
    return torch.stack((samples['gyro'].valid, samples['specific_force'].valid), dim=-1)[
        :, binding.role_slot_to_group_slot].to(dtype=torch.float32)


def imu_age(env, *, role_id):
    import torch
    binding = env.experiment.bindings[role_id]
    samples = env.sensor_outputs[binding.group_id]
    return torch.stack((samples['gyro'].age, samples['specific_force'].age), dim=-1)[
        :, binding.role_slot_to_group_slot]


def rangefinder_distance(env, *, role_id):
    binding = env.experiment.bindings[role_id]
    return env.sensor_outputs[binding.group_id]['range'].data[:, binding.role_slot_to_group_slot]


def rangefinder_valid(env, *, role_id):
    import torch
    binding = env.experiment.bindings[role_id]
    return env.sensor_outputs[binding.group_id]['range'].valid[
        :, binding.role_slot_to_group_slot].unsqueeze(-1).to(dtype=torch.float32)


def rangefinder_age(env, *, role_id):
    binding = env.experiment.bindings[role_id]
    return env.sensor_outputs[binding.group_id]['range'].age[
        :, binding.role_slot_to_group_slot].unsqueeze(-1)


def previous_action(env, *, role_id):
    return env.physics_outputs.previous_action[role_id]


def planar_target(env, *, role_id):
    return relative_target(env, role_id=role_id)[..., :2]


def planar_velocity(env, *, role_id):
    return env.physics_outputs.velocity_world[role_id][..., :2]


def heading(env, *, role_id):
    import torch
    w, x, y, z = env.physics_outputs.orientation_wxyz[role_id].unbind(dim=-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)


def yaw_rate(env, *, role_id):
    # 当前平面车体的 z 角速度；不外推为任意倾斜姿态的 world yaw 导数。
    return env.physics_outputs.angular_velocity_body[role_id][..., 2:3]


def previous_drive_action(env, *, role_id):
    return env.physics_outputs.previous_action[role_id]


def distance_reward(env, *, role_id, scale):
    import torch
    return -scale * torch.linalg.vector_norm(
        env.role_positions_world[role_id] - env.act_commands.target_world[role_id], dim=-1)


class ProgressReward:
    def __init__(self, cfg, env):
        import torch
        self.env = env
        self.role_id = cfg.params['role_id']
        self.previous_distance = torch.zeros_like(env.role_positions_world[self.role_id][..., 0])

    def reset(self, env_ids):
        import torch
        distance = torch.linalg.vector_norm(self.env.role_positions_world[self.role_id]
            - self.env.commands.target_world[self.role_id], dim=-1)
        self.previous_distance[env_ids] = distance[env_ids]

    def __call__(self, env, *, role_id, scale):
        import torch
        distance = torch.linalg.vector_norm(env.role_positions_world[role_id]
            - env.act_commands.target_world[role_id], dim=-1)
        reward = scale * (self.previous_distance - distance)
        self.previous_distance.copy_(distance)
        return reward


def collision(env, *, controlled_role, reduction):
    return env.physics_outputs.crashed[controlled_role].any(dim=-1)


def timeout(env, *, episode_max_steps):
    return env.physics_outputs.episode_decision >= episode_max_steps


def tensor_spec(shape, units='1', frame='none', dtype='float32'):
    return dict(shape=shape, dtype=dtype, units=units, frame=frame, rate='decision')


def registrations():
    role = {'binding': 'role_id'}
    position = dict(path=['role_positions_world', role], phase='post_physics',
                    tensor=tensor_spec(['B', 'N_role', 3], 'm', 'world_z_up'))
    next_target = dict(path=['commands', 'target_world', role], phase='next_command',
                       tensor=tensor_spec(['B', 'N_role', 3], 'm', 'world_z_up'))
    act_target = dict(path=['act_commands', 'target_world', role], phase='action',
                      tensor=tensor_spec(['B', 'N_role', 3], 'm', 'world_z_up'))
    result = {
        'relative_target': TermRegistration(relative_target, 'observation', 'role', EmptyConfig,
            (position, next_target), tensor_spec(['B', 'N_role', 3], 'm', 'world_z_up')),
        'planar_target': TermRegistration(planar_target, 'observation', 'role', EmptyConfig,
            (position, next_target), tensor_spec(['B', 'N_role', 2], 'm', 'world_z_up')),
        'distance': TermRegistration(distance_reward, 'reward', 'role', DistanceConfig,
            (position, act_target), tensor_spec(['B', 'N_role'], 'm')),
        'progress': TermRegistration(ProgressReward, 'reward', 'role', DistanceConfig,
            (position, act_target), tensor_spec(['B', 'N_role'], 'm')),
        'collision': TermRegistration(collision, 'termination', 'world', CollisionConfig,
            (dict(path=['physics_outputs', 'crashed', role], phase='post_physics',
                  tensor=tensor_spec(['B', 'N_role'], dtype='bool')),),
            tensor_spec(['B'], dtype='bool')),
        'timeout': TermRegistration(timeout, 'termination', 'world', TimeoutConfig,
            (dict(path=['physics_outputs', 'episode_decision'], phase='post_physics',
                  tensor=tensor_spec(['B'], dtype='int64')),),
            tensor_spec(['B'], dtype='bool'), time_out=True),
    }
    for name, func, field, width, units, frame in (
        ('velocity_world', velocity_world, 'velocity_world', 3, 'm/s', 'world_z_up'),
        ('orientation_wxyz', orientation_wxyz, 'orientation_wxyz', 4, '1', 'world_z_up'),
        ('angular_velocity_body', angular_velocity_body, 'angular_velocity_body', 3, 'rad/s', 'body_flu'),
        ('previous_action', previous_action, 'previous_action', 4, 'mixed', 'mixed'),
        ('previous_drive_action', previous_drive_action, 'previous_action', 2, 'mixed', 'mixed'),
    ):
        output = tensor_spec(['B', 'N_role', width], units, frame)
        result[name] = TermRegistration(func, 'observation', 'role', EmptyConfig,
            (dict(path=['physics_outputs', field, role], phase='post_physics', tensor=output),), output)
    for name, func, field, input_width, output_width, units, frame in (
        ('planar_velocity', planar_velocity, 'velocity_world', 3, 2, 'm/s', 'world_z_up'),
        ('heading', heading, 'orientation_wxyz', 4, 2, '1', 'world_z_up'),
        ('yaw_rate', yaw_rate, 'angular_velocity_body', 3, 1, 'rad/s', 'body_flu'),
    ):
        result[name] = TermRegistration(func, 'observation', 'role', EmptyConfig,
            (dict(path=['physics_outputs', field, role], phase='post_physics',
                  tensor=tensor_spec(['B', 'N_role', input_width], units, frame)),),
             tensor_spec(['B', 'N_role', output_width], units, frame))
    group = {'binding': 'group_id'}
    for name, func, signal, units in (
        ('imu_gyro', imu_gyro, 'gyro', 'rad/s'),
        ('imu_specific_force', imu_specific_force, 'specific_force', 'm/s²'),
    ):
        result[name] = TermRegistration(func, 'observation', 'role', EmptyConfig,
            (dict(path=['sensor_outputs', group, signal, 'data'], phase='post_physics',
                  tensor=tensor_spec(['B', 'N_group', 3], units, 'imu_flu')),),
            tensor_spec(['B', 'N_role', 3], units, 'imu_flu'))
    for name, func, field, units, dtype in (
        ('imu_valid', imu_valid, 'valid', '1', 'bool'),
        ('imu_age', imu_age, 'age', 's', 'float32'),
    ):
        result[name] = TermRegistration(func, 'observation', 'role', EmptyConfig,
            tuple(dict(path=['sensor_outputs', group, signal, field], phase='post_physics',
                       tensor=tensor_spec(['B', 'N_group'], units, dtype=dtype))
                  for signal in ('gyro', 'specific_force')),
            tensor_spec(['B', 'N_role', 2], units))
    for name, func, field, units, dtype in (
        ('rangefinder_distance', rangefinder_distance, 'data', 'm', 'float32'),
        ('rangefinder_valid', rangefinder_valid, 'valid', '1', 'bool'),
        ('rangefinder_age', rangefinder_age, 'age', 's', 'float32'),
    ):
        result[name] = TermRegistration(func, 'observation', 'role', EmptyConfig,
            (dict(path=['sensor_outputs', group, 'range', field], phase='post_physics',
                  tensor=tensor_spec(['B', 'N_group', *([1] if field == 'data' else [])],
                                     units, dtype=dtype)),),
            tensor_spec(['B', 'N_role', 1], units))
    from .wireless_terms import registrations as wireless_registrations
    result.update(wireless_registrations())
    return result
