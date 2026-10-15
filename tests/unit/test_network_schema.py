"""无线声明边界保持CPU-only，拒绝不明确的参数。"""
import subprocess
import sys

import pytest
from pydantic import ValidationError


def test_network_schema_is_cpu_only():
    code = "import sys; import muse_sdk.network; import muse_sdk.experiment; assert not any(x in sys.modules for x in ('torch', 'mujoco', 'warp', 'mjlab'))"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def selection(**updates):
    from muse_sdk.network import NetworkSelection
    data = dict(resources=[dict(resource_id='band', channel_id=7, center_frequency_hz=1e9,
        bandwidth_hz=1e6, sharing='reuse')], epoch_max_ticks=4, endpoint_limits={},
        role_controls={}, background_emitters=[], passive_endpoints=[], communication_allocations=[], services=[])
    return NetworkSelection.model_validate({**data, **updates})


@pytest.mark.parametrize('ticks', [True, 4.0, 0, -1])
def test_epoch_ticks_are_strict_positive_integers(ticks):
    with pytest.raises(ValidationError):
        selection(epoch_max_ticks=ticks)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.])
def test_service_bits_are_finite_nonnegative(value):
    with pytest.raises(ValidationError):
        selection(services=[dict(service_id='payload', link_id='up', resource_ids=['band'], initial_bits=value)])


def test_zero_bits_and_unknown_parameters():
    assert selection(services=[dict(service_id='payload', link_id='up', resource_ids=['band'], initial_bits=0.)]).services[0].initial_bits == 0
    with pytest.raises(ValidationError):
        selection(extra_mode=True)


def test_markov_matrix_and_initial_state_are_explicit():
    state = dict(active=[True], power_w=[1.], channel_id=[7])
    emitter = dict(emitter_id='noise', endpoints=['tx'], behavior='markov', config=dict(
        states=[state, state], transition_matrix=[[0., 1.], [1., 0.]], dwell_ticks=5, initial_state=0))
    assert selection(background_emitters=[emitter]).background_emitters[0].config.initial_state == 0
    for update in (dict(transition_matrix=[[0., .5], [1., 0.]]), dict(initial_distribution=[.5, .5]),
                   dict(initial_state=None), dict(dwell_ticks=True)):
        with pytest.raises(ValidationError):
            selection(background_emitters=[{**emitter, 'config': {**emitter['config'], **update}}])


def test_nonfinite_power_and_wrong_behavior_parameters_are_rejected():
    from wireless_helpers import resource
    emitter = dict(emitter_id='noise', endpoints=['tx'], behavior='fixed',
        config=dict(active=[True], power_w=[float('inf')], channel_id=[7]))
    with pytest.raises(ValidationError):
        selection(resources=[resource()], background_emitters=[emitter])
    emitter['config'].update(power_w=[1.], dwell_ticks=4)
    with pytest.raises(ValidationError):
        selection(background_emitters=[emitter])


def test_opaque_resource_and_emitter_ids_are_not_public_path_segments():
    from wireless_helpers import resource
    emitter = dict(emitter_id='noise-process', endpoints=['tx'], behavior='fixed',
        config=dict(active=[False], power_w=[0.], channel_id=[7]))
    assert selection(resources=[resource('research-band')], background_emitters=[emitter]).resources[0].resource_id == 'research-band'
