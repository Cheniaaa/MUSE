import pytest
from pydantic import ValidationError

from muse_sdk.run_protocol import RunRequest


def test_recording_request_is_explicit_and_optional():
    data = dict(experiment_id='exp', mode='EVALUATE', max_decision_steps=8)
    assert RunRequest(**data).model_dump()['recording'] is None
    config = dict(world_ids=[1, 0], every_n_decisions=3, max_data_bytes=65536)
    assert RunRequest(**data, recording=config).model_dump()['recording'] == config


@pytest.mark.parametrize('patch', [
    {'world_ids': []}, {'world_ids': [0, 0]}, {'world_ids': [-1]},
    {'world_ids': [True]}, {'world_ids': ['0']}, {'every_n_decisions': 0},
    {'every_n_decisions': True}, {'max_data_bytes': 0}, {'max_data_bytes': '300'},
])
def test_recording_input_rejects_invalid_config(patch):
    from muse_sdk.recording_protocol import RecordingConfig
    config = dict(world_ids=[0], every_n_decisions=3, max_data_bytes=65536)
    with pytest.raises(ValidationError):
        RecordingConfig(**(config | patch))
