"""单一批量速度/姿态控制器；输入机体FLU速度，输出物理总推力/力矩。"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F
from mjlab.utils.lab_api.math import matrix_from_quat


@dataclass(frozen=True)
class VelocityControllerConfig:
    velocity_kp_s_inv: tuple[float, float, float]
    velocity_ki_s_inv2: tuple[float, float, float]
    attitude_kp_nm_per_rad: tuple[float, float, float]
    rate_kd_nm_s_per_rad: tuple[float, float, float]
    integral_limit_m: tuple[float, float, float]
    tilt_limit_rad: float


class DiffDriveController:
    """原car固定tendon传动：轮速反馈与已知关节阻尼前馈。"""
    def __init__(self, config):
        self.radius = config['wheel_radius_m']
        self.separation = config['wheel_separation_m']
        self.damping = config['wheel_damping_nm_s']
        self.kp = config['wheel_velocity_kp_nm_s']

    @torch.no_grad()
    def compute(self, setpoint, wheel_velocity):
        forward, yaw = setpoint.unbind(-1)
        target = torch.stack((forward - self.separation * yaw / 2,
                              forward + self.separation * yaw / 2), -1) / self.radius
        torque = self.damping * target + self.kp * (target - wheel_velocity)
        return torch.stack((torque.sum(-1), torque[..., 1] - torque[..., 0]), -1)


class VelocityController:
    def __init__(self, config, model_parameters, batch_shape, device):
        self.config = config
        self.mass = model_parameters['mass_kg']
        self.gravity = model_parameters['gravity_mps2']
        self.velocity_kp = torch.tensor(config.velocity_kp_s_inv, device=device)
        self.velocity_ki = torch.tensor(config.velocity_ki_s_inv2, device=device)
        self.attitude_kp = torch.tensor(config.attitude_kp_nm_per_rad, device=device)
        self.rate_kd = torch.tensor(config.rate_kd_nm_s_per_rad, device=device)
        self.integral_limit = torch.tensor(config.integral_limit_m, device=device)
        self.velocity_integral = torch.zeros((*batch_shape, 3), device=device)
        self.yaw_target = torch.zeros(batch_shape, device=device)

    @torch.no_grad()
    def reset(self, env_ids, orientation_wxyz):
        rotation = matrix_from_quat(orientation_wxyz)
        self.velocity_integral[env_ids] = 0
        self.yaw_target[env_ids] = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])

    @torch.no_grad()
    def compute(self, velocity_body, yaw_rate, orientation_wxyz, velocity_world,
                angular_velocity_body, dt):
        rotation = matrix_from_quat(orientation_wxyz)
        target_world = (rotation @ velocity_body.unsqueeze(-1)).squeeze(-1)
        error = target_world - velocity_world
        self.velocity_integral.add_(error * dt)
        self.velocity_integral.clamp_(-self.integral_limit, self.integral_limit)
        acceleration = self.velocity_kp * error + self.velocity_ki * self.velocity_integral
        acceleration[..., 2] += self.gravity

        # 在上半球限制倾角；总推力/力矩的物理饱和由actuator执行层处理。
        acceleration[..., 2].clamp_(min=self.gravity * .1)
        lateral = acceleration[..., :2]
        lateral_norm = torch.linalg.vector_norm(lateral, dim=-1, keepdim=True)
        lateral_limit = acceleration[..., 2:3] * math.tan(self.config.tilt_limit_rad)
        lateral.mul_((lateral_limit / lateral_norm.clamp_min(1e-9)).clamp(max=1))
        desired_z = F.normalize(acceleration, dim=-1)

        self.yaw_target.add_(yaw_rate * dt)
        self.yaw_target.copy_((self.yaw_target + math.pi).remainder(2 * math.pi) - math.pi)
        heading = torch.stack((self.yaw_target.cos(), self.yaw_target.sin(),
                               torch.zeros_like(self.yaw_target)), dim=-1)
        desired_y = F.normalize(torch.linalg.cross(desired_z, heading), dim=-1)
        desired_x = torch.linalg.cross(desired_y, desired_z)
        desired = torch.stack((desired_x, desired_y, desired_z), dim=-1)
        skew = desired.transpose(-1, -2) @ rotation - rotation.transpose(-1, -2) @ desired
        attitude_error = .5 * torch.stack((skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), dim=-1)
        target_rate_body = rotation[..., 2, :] * yaw_rate.unsqueeze(-1)
        torque = -self.attitude_kp * attitude_error - self.rate_kd * (angular_velocity_body - target_rate_body)
        thrust = self.mass * (acceleration * rotation[..., :, 2]).sum(-1, keepdim=True)
        return torch.cat((thrust, torque), dim=-1)
