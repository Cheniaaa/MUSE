"""可上传IMU示例与正式HTTP训练、学习checkpoint恢复。"""
import json
import os
from pathlib import Path

import pytest

from flight_helpers import archive
from test_runs_api import real_http_client, zip_body
from test_supervisor import wait_for


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / 'examples/imu_flight'


def test_imu_example_resolves_real_actor_inputs(tmp_path):
    from muse_builtins.metadata import export_metadata
    from muse_server.catalog import Catalog
    from muse_server.resolver import resolve_experiment
    payload = json.loads((EXAMPLE / 'request_train.json').read_text())
    metadata = tmp_path / 'metadata.json'
    metadata.write_text(export_metadata(ROOT / 'examples/basic_flight/controller.json').model_dump_json())
    catalog = Catalog(tmp_path / 'catalog', metadata)
    for kind, source in [('environment', ROOT / 'examples/basic_flight/environment'),
                         ('training_package', EXAMPLE / 'training_package')]:
        resource = catalog.register_archive(kind, archive(source, tmp_path / (kind + '.zip')))
        payload[kind + '_id'], payload[kind + '_revision'] = resource.resource_id, resource.revision
    report = resolve_experiment(payload, catalog.snapshot())
    assert report.valid, report.errors
    exp = report.resolved_experiment
    for role, fields in exp.algorithm_inputs.actors.items():
        assert sum(field.tensor.shape[-1] for field in fields) == 24
        assert {'imu_gyro', 'imu_specific_force', 'imu_valid', 'imu_age'} <= {f.path[-1] for f in fields}
        group = exp.bindings[role].group_id
        assert all(output['period_ticks'] == 5 for output in exp.sensors[group]['outputs'].values())
        assert 'sensor:' + group in exp.activation


@pytest.mark.skipif(os.environ.get('MUSE_TEST_GPU') != '1', reason='Explicit serial GPU acceptance')
def test_imu_real_http_train_and_restore(real_http_client):
    client = real_http_client
    payload = json.loads((EXAMPLE / 'request_train.json').read_text())
    for kind, route, source in [('environment', 'environments', ROOT / 'examples/basic_flight/environment'),
                               ('training_package', 'training-packages', EXAMPLE / 'training_package')]:
        response = client.post('/catalog/' + route, content=zip_body(source),
                               headers={'Content-Type': 'application/zip'})
        assert response.status_code == 201, response.text
        record = response.json()
        payload[kind + '_id'], payload[kind + '_revision'] = record['resource_id'], record['revision']
    payload['algorithm_config'].update(collection_steps=4, num_learning_epochs=1, num_mini_batches=1)
    saved = client.post('/experiments', json=payload)
    assert saved.status_code == 201, saved.text
    identity = saved.json()['experiment_id']
    checkpoint = None
    for mode, steps, updates in [('TRAIN', 8, 2), ('EVALUATE', 2, 0)]:
        request = dict(experiment_id=identity, mode=mode, max_decision_steps=steps)
        if checkpoint is not None:
            request['checkpoint_artifact_id'] = checkpoint['id']
        created = client.post('/runs', json=request)
        assert created.status_code == 201, created.text
        run_id = created.json()['run_id']
        result = wait_for(lambda: (r if (r := client.get(f'/runs/{run_id}').json())['state']
            in ('COMPLETED', 'FAILED', 'LOST', 'CANCELLED') else None), timeout=180)
        root = Path(result['artifact_directory'])
        assert result['state'] == 'COMPLETED', (result, (root / 'logs/worker.log').read_text())
        assert result['exit_code'] == 0
        assert result['metrics']['steps'] == steps and result['metrics']['updates'] == updates
        frozen = json.loads((root / 'resolved_run_plan.json').read_text())
        assert all(output['period_ticks'] == 5 for profile in frozen['experiment']['sensors'].values()
                   for output in profile['outputs'].values())
        if checkpoint is None:
            checkpoint = next(a for a in result['artifacts'] if a['kind'] == 'checkpoint')
        else:
            assert frozen['checkpoint']['artifact_id'] == checkpoint['id']
