import pytest

from muse_sdk.run_protocol import RunTimeouts


@pytest.mark.parametrize('timeout,interval', [(.1, 1.), (1., 1.)])
def test_heartbeat_deadline_must_allow_a_scheduled_heartbeat(timeout, interval):
    with pytest.raises(ValueError, match='heartbeat'):
        RunTimeouts(heartbeat=timeout, heartbeat_interval=interval)


def test_heartbeat_interval_can_be_tuned_with_timeout():
    limits = RunTimeouts(heartbeat=.5, heartbeat_interval=.05)
    assert limits.heartbeat == .5
    assert limits.heartbeat_interval == .05
