import pytest
import torch


@pytest.mark.parametrize('sign', [1., -1.])
def test_diff_drive_tendon_control_matches_wheel_torque(sign):
    from muse_builtins.controllers import DiffDriveController
    controller = DiffDriveController(dict(wheel_radius_m=.03, wheel_separation_m=.12,
        wheel_damping_nm_s=.03, wheel_velocity_kp_nm_s=.01))
    target = torch.tensor([[[.1, .8], [0., 0.]]]) * sign
    stationary = torch.zeros(1, 2, 2)
    control = controller.compute(target, stationary)
    torch.testing.assert_close(control, torch.tensor([[[.26666667, .128], [0., 0.]]]) * sign)
    wheel_speed = torch.tensor([[[1., 2.], [3., -2.]]]) * sign
    expected = torch.tensor([[[.23666667, .118], [-.01, .05]]]) * sign
    torch.testing.assert_close(controller.compute(target, wheel_speed), expected)
