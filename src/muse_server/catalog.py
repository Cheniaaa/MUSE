"""CPU资源目录，SQLite直接持久化，用户代码仅在one-shot子进程执行。"""
import json
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import subprocess
import sys
from uuid import uuid4

from muse_sdk.config import ConfigError
from muse_sdk.geometry import GeometryAssetSummary
from muse_sdk.descriptor import parse_descriptor_json
from muse_sdk.run_protocol import CheckpointArtifact, CheckpointReference, RunTimeouts
from muse_sdk.experiment import (
    BuiltinMetadata, CatalogSnapshot, EnvironmentBundle, ExperimentDefinition, ResourceRecord, RunPlan,
    learning_contract,
)

from .storage import extract_archive, read_document, write_json_new, validate_bundled_assets, verify_checkpoint_file


class Catalog:
    def __init__(self, directory: Path, builtin_metadata: Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.builtin_path = self.directory / 'builtin-metadata.json'
        incoming = BuiltinMetadata.model_validate_json(Path(builtin_metadata).read_text())
        parse_descriptor_json(incoming.descriptor.model_dump_json())
        if self.builtin_path.exists():
            if self.builtin_path.read_text() != incoming.model_dump_json():
                raise ConfigError('platform_revision', 'Catalog already pins different builtin metadata')
        else:
            write_json_new(self.builtin_path, incoming.model_dump_json())
        self.builtins = incoming
        for name, dto in (('experiment', ExperimentDefinition), ('run-plan', RunPlan)):
            path = self.directory / f'{name}.schema.json'
            if not path.exists():
                write_json_new(path, json.dumps(dto.model_json_schema(), indent=2))
        for name in ('resources', 'runs'):
            (self.directory / name).mkdir(exist_ok=True)
        self.database = self.directory / 'catalog.sqlite3'
        with self._connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS resources (
                    id TEXT NOT NULL, revision INTEGER NOT NULL, kind TEXT NOT NULL,
                    record TEXT NOT NULL, PRIMARY KEY (id, revision));
                CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, definition TEXT NOT NULL);
            ''')

    def _connect(self):
        return sqlite3.connect(self.database)

    def register_archive(self, kind: str, archive: Path, *, resource_id: str | None = None) -> ResourceRecord:
        if kind not in ('environment', 'training_package'):
            raise ConfigError('kind', 'Unsupported upload kind')
        with self._connect() as db:
            if resource_id is None:
                resource_id, revision = uuid4().hex, 1
            else:
                row = db.execute('SELECT MAX(revision) FROM resources WHERE id=? AND kind=?',
                                 (resource_id, kind)).fetchone()
                if row[0] is None:
                    raise ConfigError('resource_id', 'Unknown resource to revise')
                revision = row[0] + 1
            root = self.directory / 'resources' / f'{resource_id}-{revision}'
            root.mkdir()
            content = root / 'content'
            try:
                extract_archive(Path(archive), content)
                if kind == 'training_package':
                    output = root / 'validation'
                    result = subprocess.run([sys.executable, '-m', 'muse_runtime.validation',
                        '--package', str(content), '--output-dir', str(output),
                        '--builtin-metadata', str(self.builtin_path)], capture_output=True, text=True)
                    report_path = output / 'validation-report.json'
                    if result.returncode or not report_path.exists():
                        detail = report_path.read_text() if report_path.exists() else result.stderr
                        raise ConfigError('training_package', f'Validation Worker failed: {detail}')
                    report = json.loads(report_path.read_text())
                    if not report['valid']:
                        raise ConfigError('training_package', str(report['errors']))
                    descriptor = parse_descriptor_json((output / 'descriptor.json').read_text(), self.builtins.descriptor)
                    metadata, digest = descriptor.model_dump(mode='json'), descriptor.package_hash
                else:
                    bundle = EnvironmentBundle.model_validate(read_document(content / 'environment.yaml'))
                    metadata = bundle.model_dump(mode='json')
                    scene = bundle.scene
                    if (not isinstance(scene, str) or '\\' in scene or ':' in scene
                            or PurePosixPath(scene).is_absolute() or '..' in PurePosixPath(scene).parts
                            or not (content / scene).is_file()):
                        raise ConfigError('environment.scene', 'Expected a bundled relative scene path')
                    validate_bundled_assets(content)
                    output = root / 'geometry-inspection.json'
                    result = subprocess.run([sys.executable, '-m', 'muse_runtime.resource_inspection',
                        '--root', str(content), '--scene', scene, '--output', str(output)],
                        capture_output=True, text=True)
                    if result.returncode:
                        raise ConfigError('environment.scene', result.stderr)
                    inspected = json.loads(output.read_text())
                    metadata['geometry_summary'] = GeometryAssetSummary.model_validate(
                        inspected['geometry_summary']).model_dump(mode='json')
                    digest = inspected['sha256']
                record = ResourceRecord(resource_id=resource_id, revision=revision, sha256=digest,
                                        kind=kind, path=str(content), metadata=metadata)
                db.execute('INSERT INTO resources VALUES (?, ?, ?, ?)',
                           (resource_id, revision, kind, record.model_dump_json()))
            except BaseException:
                shutil.rmtree(root)
                raise
        return record

    def snapshot(self) -> CatalogSnapshot:
        with self._connect() as db:
            records = [ResourceRecord.model_validate_json(row[0]) for row in
                       db.execute('SELECT record FROM resources ORDER BY id, revision')]
        return CatalogSnapshot(resources=records, builtins=self.builtins.model_copy(deep=True))

    def save_experiment(self, experiment: ExperimentDefinition) -> str:
        identity = uuid4().hex
        with self._connect() as db:
            db.execute('INSERT INTO experiments VALUES (?, ?)', (identity, experiment.model_dump_json()))
        return identity

    def get_experiment(self, identity: str) -> ExperimentDefinition:
        with self._connect() as db:
            row = db.execute('SELECT definition FROM experiments WHERE id=?', (identity,)).fetchone()
        if row is None:
            raise ConfigError('experiment_id', 'Unknown experiment')
        return ExperimentDefinition.model_validate_json(row[0])

    def resolve_checkpoint(self, artifact_id: str, experiment: ExperimentDefinition) -> CheckpointReference:
        run_id = artifact_id.split(':', 1)[0]
        with self._connect() as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='runs'").fetchone() is None:
                raise ConfigError('checkpoint_artifact_id', 'Unknown checkpoint artifact')
            row = db.execute('SELECT record FROM runs WHERE id=?', (run_id,)).fetchone()
        if row is None:
            raise ConfigError('checkpoint_artifact_id', 'Unknown checkpoint artifact')
        record = json.loads(row[0])
        item = next((a for a in record['artifacts'] if a['id'] == artifact_id), None)
        if item is None or item['kind'] != 'checkpoint':
            raise ConfigError('checkpoint_artifact_id', 'Expected a registered checkpoint artifact')
        try:
            artifact = CheckpointArtifact.model_validate(item)
            path = verify_checkpoint_file(artifact, run_id, record['artifact_directory'])
            source = RunPlan.model_validate_json(Path(record['plan_path']).read_text())
            if record['run_id'] != run_id or source.run_id != run_id:
                raise ValueError('Checkpoint source RunPlan identity mismatch')
            if learning_contract(source.experiment) != learning_contract(experiment):
                raise ValueError('Incompatible checkpoint learning contract')
            for field in ('environment', 'environment_scene', 'environment_metadata', 'entity_models',
                          'scenario', 'controllers', 'sensors'):
                if getattr(source.experiment, field) != getattr(experiment, field):
                    raise ValueError(f'Incompatible checkpoint {field}')
        except (ValueError, OSError) as exc:
            raise ConfigError('checkpoint_artifact_id', str(exc)) from exc
        return CheckpointReference(artifact_id=artifact.id, path=str(path), size=artifact.size, sha256=artifact.sha256)

    def create_run_plan(self, experiment_id: str, *, mode: str, device: str,
                        checkpoint_artifact_id: str | None = None, max_decision_steps: int = 24000,
                        checkpoint_interval_updates: int | None = None,
                        timeouts: RunTimeouts | None = None) -> RunPlan:
        experiment = self.get_experiment(experiment_id)
        if type(max_decision_steps) is not int or max_decision_steps <= 0:
            raise ConfigError('max_decision_steps', 'Expected a positive integer decision budget')
        if mode not in ('TRAIN', 'EVALUATE', 'INTERACTIVE'):
            raise ConfigError('mode', 'Unknown Run mode')
        if checkpoint_interval_updates is not None and (
                mode != 'TRAIN' or type(checkpoint_interval_updates) is not int or checkpoint_interval_updates <= 0):
            raise ConfigError('checkpoint_interval_updates', 'Expected a positive integer interval for TRAIN')
        if mode == 'TRAIN' and experiment.training_schedule is None:
            raise ConfigError('mode', 'TRAIN requires a learning Algorithm')
        if mode in ('EVALUATE', 'INTERACTIVE'):
            if experiment.training_schedule is not None and checkpoint_artifact_id is None:
                raise ConfigError('checkpoint_artifact_id', f'Learning {mode} requires a checkpoint')
            if experiment.training_schedule is None and checkpoint_artifact_id is not None:
                raise ConfigError('checkpoint_artifact_id', f'Rule {mode} cannot load a checkpoint')
        checkpoint = self.resolve_checkpoint(checkpoint_artifact_id, experiment) if checkpoint_artifact_id is not None else None
        if experiment.platform_revision != self.builtins.platform_revision:
            raise ConfigError('platform_revision', 'Experiment platform revision is unavailable')
        snapshot = self.snapshot()
        paths = {'builtin_metadata': str(self.builtin_path)}
        refs = {'environment': experiment.environment, 'training_package': experiment.training_package}
        for key, ref in refs.items():
            record = next((r for r in snapshot.resources if (r.resource_id, r.revision, r.sha256)
                           == (ref.resource_id, ref.revision, ref.sha256)), None)
            if record is None:
                raise ConfigError(key, 'Frozen resource is unavailable')
            paths[key] = record.path
        for model, ref in experiment.entity_models.items():
            record = self.builtins.entity_models[model]
            if record.sha256 != ref.sha256 or record.revision != ref.revision:
                raise ConfigError(model, 'Frozen entity model is unavailable')
            paths['entity_model:' + model] = record.path
        run_id = uuid4().hex
        directory = self.directory / 'runs' / run_id
        directory.mkdir()
        plan = RunPlan(run_id=run_id, experiment_id=experiment_id, mode=mode, device=device,
                       platform_revision=experiment.platform_revision, artifact_directory=str(directory),
                       resource_paths=paths, experiment=experiment, max_decision_steps=max_decision_steps,
                       checkpoint=checkpoint, checkpoint_interval_updates=checkpoint_interval_updates,
                       timeouts=timeouts or RunTimeouts())
        write_json_new(directory / 'resolved_run_plan.json', plan.model_dump_json(indent=2))
        return plan
