import pytest
from pydantic import ValidationError

from muse_sdk.run_protocol import RunRequest


def test_checkpoint_interval_is_explicit():
    request = RunRequest(experiment_id='exp', mode='TRAIN', max_decision_steps=8,
                         checkpoint_interval_updates=2)
    assert request.checkpoint_interval_updates == 2
    assert RunRequest(experiment_id='exp', mode='TRAIN', max_decision_steps=8).checkpoint_interval_updates is None


@pytest.mark.parametrize('value', [0, -1, True, '2'])
def test_invalid_checkpoint_interval_rejected(value):
    with pytest.raises(ValidationError):
        RunRequest(experiment_id='exp', mode='TRAIN', max_decision_steps=8,
                   checkpoint_interval_updates=value)


def test_evaluate_cannot_request_periodic_saving():
    with pytest.raises(ValidationError):
        RunRequest(experiment_id='exp', mode='EVALUATE', max_decision_steps=3,
                   checkpoint_interval_updates=2)


def test_request_accepts_artifact_identity_not_a_file():
    request = RunRequest(experiment_id='exp', mode='TRAIN', max_decision_steps=8,
                         checkpoint_artifact_id='source:checkpoint:abc')
    assert request.checkpoint_artifact_id == 'source:checkpoint:abc'
    with pytest.raises(ValidationError):
        RunRequest(**request.model_dump(), path='/tmp/checkpoint.pt')
