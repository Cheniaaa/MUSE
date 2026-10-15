import math

import pytest
import torch


@pytest.fixture
def controller():
    from muse_builtins.controllers import VelocityController, VelocityControllerConfig

    config = VelocityControllerConfig(
        velocity_kp_s_inv=(.8, .8, 2.), velocity_ki_s_inv2=(.05, .05, .1),
        attitude_kp_nm_per_rad=(.00038, .00038, .00052),
        rate_kd_nm_s_per_rad=(.00019, .00019, .00026),
        integral_limit_m=(.2, .2, .2), tilt_limit_rad=.2,
    )
    return VelocityController(config, {'mass_kg': .027, 'gravity_mps2': 9.81}, (2, 2), 'cpu')


def test_level_zero_error_produces_weight(controller):
    q = torch.tensor([1., 0, 0, 0]).expand(2, 2, 4)
    zero = torch.zeros(2, 2, 3)
    controller.reset(torch.tensor([0, 1]), q)
    wrench = controller.compute(zero, zero[..., 0], q, zero, zero, .002)
    torch.testing.assert_close(wrench[..., 0], torch.full((2, 2), .26487))
    torch.testing.assert_close(wrench[..., 1:], zero)


def test_body_forward_at_yaw_quarter_turn_accelerates_world_y(controller):
    q = torch.tensor([2**-.5, 0, 0, 2**-.5]).expand(2, 2, 4)
    zero = torch.zeros(2, 2, 3)
    controller.reset(torch.tensor([0, 1]), q)
    wrench = controller.compute(torch.tensor([.1, 0, 0]).expand_as(zero), zero[..., 0], q, zero, zero, .002)
    assert (wrench[..., 2] > 0).all()
    torch.testing.assert_close(wrench[..., 1], torch.zeros(2, 2), atol=1e-8, rtol=0)
    assert (controller.velocity_integral[..., 1] > 0).all()
    torch.testing.assert_close(controller.velocity_integral[..., 0], torch.zeros(2, 2), atol=1e-8, rtol=0)


def test_yaw_wrap_uses_short_error(controller):
    angle = math.pi - .0001
    q = torch.tensor([math.cos(angle / 2), 0, 0, math.sin(angle / 2)]).expand(2, 2, 4)
    zero = torch.zeros(2, 2, 3)
    controller.reset(torch.tensor([0, 1]), q)
    wrench = controller.compute(zero, torch.full((2, 2), .1), q, zero, zero, .002)
    assert (wrench[..., 3] > 0).all()
    assert (wrench[..., 3] < 3e-5).all()
    assert (controller.yaw_target < 0).all()


def test_subset_reset_preserves_other_world(controller):
    q = torch.tensor([1., 0, 0, 0]).expand(2, 2, 4)
    zero = torch.zeros(2, 2, 3)
    controller.reset(torch.tensor([0, 1]), q)
    controller.compute(torch.full_like(zero, .1), torch.full((2, 2), .1), q, zero, zero, .002)
    other_integral = controller.velocity_integral[1].clone()
    other_yaw = controller.yaw_target[1].clone()
    controller.reset(torch.tensor([0]), q[:1])
    assert torch.equal(controller.velocity_integral[1], other_integral)
    assert torch.equal(controller.yaw_target[1], other_yaw)
    assert torch.count_nonzero(controller.velocity_integral[0]) == 0
    assert torch.count_nonzero(controller.yaw_target[0]) == 0


def test_integral_is_bounded(controller):
    q = torch.tensor([1., 0, 0, 0]).expand(2, 2, 4)
    zero = torch.zeros(2, 2, 3)
    controller.reset(torch.tensor([0, 1]), q)
    controller.compute(torch.ones_like(zero), zero[..., 0], q, zero, zero, 10.)
    torch.testing.assert_close(controller.velocity_integral, torch.full_like(zero, .2))
