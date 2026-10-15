"""真实Worker训练发布checkpoint后，恢复的新episode在首次step前发布初始IMU。"""
import json
from pathlib import Path
import sqlite3

import torch

from flight_helpers import archive


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / 'examples/imu_flight'


class ProtocolCollector:
    def __init__(self):
        self.events = []

    def emit(self, kind, payload=None, *, sim_time=0.):
        self.events.append((kind, payload))

    def read_control(self):
        return None

    def take_preview_controls(self):
        return ()

    def take_command(self):
        return None


def execute_with_preview(plan, monkeypatch):
    from muse_runtime.worker import execute
    from muse_sdk.run_protocol import preview_socketpair

    protocol = ProtocolCollector()
    sender, receiver = preview_socketpair()
    with sender, receiver:
        # Worker接管并关闭此fd；receiver在整个真实执行期间保持存活。
        monkeypatch.setenv('MUSE_PREVIEW_FD', str(sender.detach()))
        result = execute(plan, protocol)
    assert not any(kind == 'preview_error' for kind, _ in protocol.events), protocol.events
    assert any(kind == 'preview_model' for kind, _ in protocol.events)
    return result, protocol


def test_worker_checkpoint_restore_has_initial_imu_before_first_step(tmp_path, monkeypatch):
    from muse_builtins.metadata import export_metadata
    from muse_runtime import checkpoint as checkpoint_module, driver
    from muse_server.catalog import Catalog
    from muse_server.resolver import resolve_experiment

    metadata = tmp_path / 'builtin.json'
    metadata.write_text(export_metadata(ROOT / 'examples/basic_flight/controller.json').model_dump_json())
    catalog = Catalog(tmp_path / 'catalog', metadata)
    request = json.loads((EXAMPLE / 'request_train.json').read_text())
    for kind, source in [('environment', ROOT / 'examples/basic_flight/environment'),
                         ('training_package', EXAMPLE / 'training_package')]:
        resource = catalog.register_archive(kind, archive(source, tmp_path / (kind + '.zip')))
        request[kind + '_id'], request[kind + '_revision'] = resource.resource_id, resource.revision
    request['algorithm_config'].update(collection_steps=4, num_learning_epochs=1, num_mini_batches=1)
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment)
    source = catalog.create_run_plan(identity, mode='TRAIN', device='cuda:0', max_decision_steps=4)

    original_run, original_load = driver.run, checkpoint_module.load_checkpoint
    loaded, published_ticks, restored = [], [], []

    def load_spy(plan, algorithm):
        original_load(plan, algorithm)
        loaded.append(plan.checkpoint.artifact_id)

    def run_spy(env, algorithm, *args, **kwargs):
        if kwargs['mode'] == 'TRAIN':
            original_sink = kwargs['checkpoint_sink']

            def checkpoint_sink(*, steps, updates):
                ticks = [sample.sample_tick.clone() for outputs in env.sensor_outputs.values()
                         for sample in outputs.values()]
                assert ticks and all((tick > 0).all() for tick in ticks)
                published_ticks.append(ticks)
                return original_sink(steps=steps, updates=updates)

            kwargs['checkpoint_sink'] = checkpoint_sink
        else:
            # 入口位于真实load_checkpoint之后、原driver首次act/step之前；不手动reset。
            assert kwargs['mode'] == 'EVALUATE'
            assert loaded == [artifact['id']]
            assert published_ticks
            assert env.episode_length_buf.count_nonzero() == 0
            assert set(env.sensor_outputs) == {'scout_group', 'guard_group'}
            for group_id, outputs in env.sensor_outputs.items():
                assert set(outputs) == {'gyro', 'specific_force'}
                for signal, native in [('gyro', 'body_gyro'), ('specific_force', 'body_linacc')]:
                    sample = outputs[signal]
                    assert sample.valid.all()
                    assert sample.age.count_nonzero() == sample.sample_tick.count_nonzero() == 0
                    # 初始空中静止、零推力应为自由落体读数，而不是人为填入+g。
                    torch.testing.assert_close(sample.data, torch.zeros_like(sample.data), atol=1e-5, rtol=0)
                    for slot in range(sample.data.shape[1]):
                        torch.testing.assert_close(sample.data[:, slot],
                            env.scene[f'{group_id}_{slot}/{native}'].data, atol=1e-5, rtol=0)
            restored.append(True)
        return original_run(env, algorithm, *args, **kwargs)

    monkeypatch.setattr(checkpoint_module, 'load_checkpoint', load_spy)
    monkeypatch.setattr(driver, 'run', run_spy)
    result, protocol = execute_with_preview(source, monkeypatch)
    assert result == {'state': 'COMPLETED', 'steps': 4, 'updates': 1}
    assert published_ticks
    artifact, = [payload for kind, payload in protocol.events if kind == 'checkpoint']
    assert (Path(source.artifact_directory) / artifact['path']).is_file()

    # 仅登记真实Worker发布的产物，Catalog仍执行正式的引用与兼容性检查。
    record = {'run_id': source.run_id, 'artifact_directory': source.artifact_directory,
              'plan_path': str(Path(source.artifact_directory) / 'resolved_run_plan.json'),
              'artifacts': [artifact]}
    with sqlite3.connect(catalog.database) as db:
        db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, record TEXT NOT NULL)')
        db.execute('INSERT INTO runs VALUES (?, ?)', (source.run_id, json.dumps(record)))
    target = catalog.create_run_plan(identity, mode='EVALUATE', device='cuda:0',
                                    checkpoint_artifact_id=artifact['id'], max_decision_steps=1)
    assert target.checkpoint.artifact_id == artifact['id']
    result, _ = execute_with_preview(target, monkeypatch)
    assert result == {'state': 'COMPLETED', 'steps': 1, 'updates': 0}
    assert restored == [True]
