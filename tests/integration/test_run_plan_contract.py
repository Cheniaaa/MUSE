import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from muse_sdk.experiment import RunPlan
from muse_sdk.recording_protocol import RecordingConfig
from muse_sdk.run_protocol import RunRequest, RunTimeouts
from muse_server.resolver import resolve_experiment


def test_catalog_freezes_typed_request_budget_and_recording(pipeline):
    catalog, source = pipeline
    experiment = resolve_experiment(source, catalog.snapshot()).resolved_experiment
    identity = catalog.save_experiment(experiment, source=source)
    request = RunRequest(experiment_id=identity, mode='EVALUATE', max_decision_steps=7,
        recording=RecordingConfig(world_ids=[0], every_n_decisions=2, max_data_bytes=65536))
    limits = RunTimeouts(startup=11.)

    plan = catalog.create_run_plan(request, device='cpu', timeouts=limits)
    loaded = RunPlan.model_validate_json(Path(plan.artifact_directory, 'resolved_run_plan.json').read_text())

    assert loaded.experiment_id == identity
    assert loaded.experiment == experiment
    assert loaded.max_decision_steps == 7
    assert loaded.recording.world_ids == [0]
    assert loaded.recording.every_n_decisions == 2
    assert loaded.timeouts.startup == 11.


@pytest.mark.parametrize('entry_point', ['constructor', 'json'])
def test_frozen_plan_requires_an_explicit_decision_budget(checkpoint_plan, entry_point):
    payload = checkpoint_plan.model_dump()
    del payload['max_decision_steps']

    with pytest.raises(ValidationError) as error:
        if entry_point == 'constructor':
            RunPlan(**payload)
        else:
            RunPlan.model_validate_json(json.dumps(payload))

    assert any(item['loc'] == ('max_decision_steps',) and item['type'] == 'missing'
               for item in error.value.errors())


@pytest.mark.parametrize('budget', [0, -1, True, 1.5, '8'])
def test_frozen_plan_rejects_invalid_serialized_decision_budget(checkpoint_plan, budget):
    payload = json.loads(checkpoint_plan.model_dump_json())
    payload['max_decision_steps'] = budget

    with pytest.raises(ValidationError) as error:
        RunPlan.model_validate_json(json.dumps(payload))

    assert any(item['loc'] == ('max_decision_steps',) for item in error.value.errors())


def test_catalog_rejects_recording_world_outside_frozen_topology(pipeline):
    catalog, source = pipeline
    experiment = resolve_experiment(source, catalog.snapshot()).resolved_experiment
    identity = catalog.save_experiment(experiment, source=source)
    request = RunRequest(experiment_id=identity, mode='EVALUATE', max_decision_steps=3,
        recording=RecordingConfig(world_ids=[experiment.topology.world_count],
            every_n_decisions=1, max_data_bytes=65536))

    with pytest.raises(ValueError, match='recording.world_ids'):
        catalog.create_run_plan(request, device='cpu')
