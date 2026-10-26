"""固定诊断字段的有界NPY写入与只读历史；不导入物理或研究实现。"""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from uuid import uuid4
import zipfile

import numpy as np

from .experiment import RunPlan
from .recording_protocol import TrajectoryArtifact, TrajectoryField, TrajectoryManifest
from .run_protocol import SceneObject


PHASES = ('initial', 'step', 'reset_before', 'reset_after')
RESET_SOURCES = ('none', 'automatic', 'command')


def build_trajectory_fields(experiment, nbody):
    fields, offsets = [], dict.fromkeys(('float', 'integer', 'boolean'), 0)

    def add(name, shape, dtype='<f4', role=None, cost=None):
        group = 'boolean' if dtype == '|b1' else 'integer' if dtype in ('<i4', '<i8') else 'float'
        fields.append(TrajectoryField(name=name, role_id=role, cost_id=cost, group=group,
            dtype=dtype, shape=list(shape), offset=offsets[group]))
        offsets[group] += math.prod(shape) * np.dtype(dtype).itemsize

    add('xpos', (nbody, 3))
    add('xmat', (nbody, 9))
    for role, binding in experiment.bindings.items():
        slots = binding.role_slot_count
        for name in ('velocity_world', 'angular_velocity_body'):
            add(name, (slots, 3), role=role)
        for name in ('present', 'alive', 'controllable', 'crashed', 'learning_done'):
            add(name, (slots,), '|b1', role)
        spec = experiment.action_specs[role]
        add('action_motion', spec.shape[1:], role=role)
        if spec.radio is not None:
            add('action_power_w', (slots,), role=role)
            add('action_channel', (slots,), '<i4', role)
        add('reward', (slots,), role=role)
        for cost in experiment.costs.roles.get(role, {}):
            add('role_cost', (slots,), role=role, cost=cost)
    if any(s.enabled and s.term_kind == 'reward' and s.binding_role_id is None for s in experiment.selections):
        add('world_reward', ())
    for cost in experiment.costs.world:
        add('world_cost', (), cost=cost)
    add('world_terminated', (), '|b1')
    add('world_truncated', (), '|b1')
    add('world_end_reason', (), '<i8')
    return fields


def trajectory_row_bytes(fields):
    return 56 + sum(math.prod(f.shape) * np.dtype(f.dtype).itemsize for f in fields)


def _array_layout(fields):
    floats = sum(math.prod(f.shape) for f in fields if f.group == 'float')
    booleans = sum(math.prod(f.shape) for f in fields if f.group == 'boolean')
    integer = np.dtype([(f'field_{i}', f.dtype, tuple(f.shape)) for i, f in enumerate(fields)
                        if f.group == 'integer'])
    return {'index': (np.dtype('<i8'), (7,)), 'float': (np.dtype('<f4'), (floats,)),
            'integer': (integer, ()), 'boolean': (np.dtype('|b1'), (booleans,))}


def _value(arrays, field, field_index, row):
    if field.group == 'integer':
        return arrays['integer'][f'field_{field_index}'][row]
    start = field.offset // np.dtype(field.dtype).itemsize
    return arrays[field.group][row, start:start + math.prod(field.shape)].reshape(field.shape)


def _ranges(arrays, fields):
    index = arrays['index']
    count = len(index)
    last = index[-1]
    summary = dict(sample_count=count, last_run_decision=int(last[1]),
                   last_episode=int(last[2]), last_decision=int(last[3]))
    ends = [i for i, f in enumerate(fields) if f.name in ('world_terminated', 'world_truncated')]
    starts = np.r_[0, np.flatnonzero(np.diff(index[:, 2])) + 1, count]
    episodes = []
    for first, stop in zip(starts[:-1], starts[1:]):
        final = stop - 1
        ended = index[final, 5] == 2 or (index[final, 5] == 1 and
            any(bool(_value(arrays, fields[i], i, final)) for i in ends))
        episodes.append(dict(episode=int(index[first, 2]), first_sample_index=int(first),
            last_sample_index=int(final), complete=bool(index[first, 5] in (0, 3) and ended)))
    return summary, episodes


class TrajectoryWriter:
    def __init__(self, plan, model):
        self.plan, self.model = plan, model
        self.fields = build_trajectory_fields(plan.experiment, model.nbody)
        self.row_bytes = trajectory_row_bytes(self.fields)
        self.rows_per_world = plan.recording.max_data_bytes // (len(plan.recording.world_ids) * self.row_bytes)
        if self.rows_per_world < 3:
            raise ValueError('Recording budget must fit at least 3 rows per world')
        self.directory = Path(plan.artifact_directory)
        self._temporary = tempfile.TemporaryDirectory(prefix='trajectory-', dir=self.directory)
        self.arrays = {}
        self.counts = dict.fromkeys(plan.recording.world_ids, 0)
        self._static = {'resolved_run_plan.json': plan.model_dump_json(indent=2).encode(),
            'objects.json': json.dumps([obj.model_dump() for obj in model.objects],
                                      ensure_ascii=False, allow_nan=False).encode()}
        self.static_bytes = sum(map(len, self._static.values())) + (self.directory / model.path).stat().st_size
        try:
            for world in self.counts:
                root = Path(self._temporary.name) / str(world)
                root.mkdir()
                self.arrays[world] = {name: np.lib.format.open_memmap(root / (name + '.npy'), mode='w+',
                    dtype=dtype, shape=(self.rows_per_world, *shape))
                    for name, (dtype, shape) in _array_layout(self.fields).items()}
        except BaseException:
            self.close()
            raise

    def can_append(self, world_ids, rows):
        return all(self.counts[world] + rows <= self.rows_per_world for world in world_ids)

    def append(self, world_id, index, values):
        row = self.counts[world_id]
        arrays = self.arrays[world_id]
        arrays['index'][row] = index
        for i, (field, value) in enumerate(zip(self.fields, values, strict=True)):
            if field.group == 'integer':
                arrays['integer'][f'field_{i}'][row] = value
            else:
                start = field.offset // np.dtype(field.dtype).itemsize
                arrays[field.group][row, start:start + math.prod(field.shape)] = value.reshape(-1)
        self.counts[world_id] += 1

    def seal(self, reason, run_decision):
        worlds, episodes = {}, {}
        for world, count in self.counts.items():
            arrays = {name: array[:count] for name, array in self.arrays[world].items()}
            worlds[str(world)], episodes[str(world)] = _ranges(arrays, self.fields)
        summary = dict(reason=reason, sealed_run_decision=run_decision, worlds=worlds)
        scene = self.plan.experiment.scenario.scene
        manifest = TrajectoryManifest(run_id=self.plan.run_id, experiment_id=self.plan.experiment_id,
            recording=self.plan.recording, physics_dt=scene.physics_dt, decimation=scene.control_substeps,
            nbody=self.model.nbody, model_sha256=self.model.sha256, fields=self.fields,
            summary=summary, episodes=episodes)
        token = uuid4().hex
        final = self.directory / 'trajectories' / (token + '.zip')
        final.parent.mkdir(exist_ok=True)
        temporary = Path(self._temporary.name) / (token + '.zip')
        with temporary.open('xb') as stream:
            with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
                archive.writestr('manifest.json', manifest.model_dump_json())
                for name, content in self._static.items():
                    archive.writestr(name, content)
                archive.write(self.directory / self.model.path, 'model.mjb')
                for world, count in self.counts.items():
                    for name, array in self.arrays[world].items():
                        with archive.open(f'worlds/{world}/{name}.npy', 'w', force_zip64=True) as member:
                            np.save(member, array[:count], allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, final)
        with final.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        return TrajectoryArtifact(id=f'{self.plan.run_id}:trajectory:{token}', run=self.plan.run_id,
            path=final.relative_to(self.directory).as_posix(), size=final.stat().st_size,
            sha256=digest, summary=summary)

    def close(self):
        self.arrays.clear()
        self._temporary.cleanup()


class TrajectoryReader:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = TrajectoryManifest.model_validate_json((self.directory / 'manifest.json').read_bytes())
        self.plan = RunPlan.model_validate_json((self.directory / 'resolved_run_plan.json').read_bytes())
        self.objects = [SceneObject.model_validate(obj) for obj in json.loads(
            (self.directory / 'objects.json').read_bytes())]
        self.arrays = {}
        try:
            self._load()
        except BaseException:
            self.close()
            raise

    def _load(self):
        manifest, plan = self.manifest, self.plan
        scene = plan.experiment.scenario.scene
        if (manifest.run_id != plan.run_id or manifest.experiment_id != plan.experiment_id
                or manifest.recording != plan.recording or manifest.physics_dt != scene.physics_dt
                or manifest.decimation != scene.control_substeps
                or manifest.fields != build_trajectory_fields(plan.experiment, manifest.nbody)):
            raise ValueError('Trajectory manifest does not match frozen Run layout')
        with (self.directory / 'model.mjb').open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != manifest.model_sha256:
                raise ValueError('Trajectory model hash mismatch')
        for obj in self.objects:
            if obj.body_id is None or not 0 <= obj.body_id < manifest.nbody:
                raise ValueError('Trajectory object body outside model')
        keys = {str(world) for world in manifest.recording.world_ids}
        if set(manifest.summary['worlds']) != keys or set(manifest.episodes) != keys:
            raise ValueError('Trajectory world ranges differ from recording config')
        capacity = plan.recording.max_data_bytes // (len(keys) * trajectory_row_bytes(manifest.fields))
        for world in manifest.recording.world_ids:
            summary = manifest.summary['worlds'][str(world)]
            count = summary['sample_count']
            if not 1 <= count <= capacity:
                raise ValueError('Trajectory sample count outside capacity')
            arrays = {}
            self.arrays[world] = arrays
            for name, (dtype, shape) in _array_layout(manifest.fields).items():
                array = np.load(self.directory / f'worlds/{world}/{name}.npy', mmap_mode='r', allow_pickle=False)
                if array.dtype != dtype or array.shape != (count, *shape):
                    raise ValueError(f'Trajectory {name} dtype/shape mismatch')
                arrays[name] = array
            index = arrays['index']
            if (not np.array_equal(index[:, 0], np.arange(count)) or np.any(index[:, :5] < 0)
                    or np.any(np.diff(index[:, 1]) < 0) or np.any(np.diff(index[:, 2]) < 0)
                    or np.any(index[:, 4] != index[:, 3] * manifest.decimation)
                    or np.any((index[:, 5] < 0) | (index[:, 5] > 3))
                    or np.any((index[:, 6] < 0) | (index[:, 6] > 2))
                    or tuple(index[0]) != (0, 0, 0, 0, 0, 0, 0)
                    or index[-1, 1] > manifest.summary['sealed_run_decision']):
                raise ValueError('Trajectory sample order/clock is invalid')
            actual_summary, episodes = _ranges(arrays, manifest.fields)
            if summary != actual_summary or manifest.episodes[str(world)] != episodes:
                raise ValueError('Trajectory summary/episode ranges mismatch')

    def episode_samples(self, world_id, episode):
        if world_id not in self.arrays:
            raise ValueError('Unknown recorded world')
        samples = np.flatnonzero(self.arrays[world_id]['index'][:, 2] == episode)
        if not len(samples):
            raise ValueError('Unknown recorded episode')
        return samples

    def frame(self, world_id, sample_index):
        if world_id not in self.arrays or not 0 <= sample_index < len(self.arrays[world_id]['index']):
            raise ValueError('Sample outside recorded range')
        arrays = self.arrays[world_id]
        index = arrays['index'][sample_index]
        diagnostics = dict(zip(('sample_index', 'run_decision', 'episode', 'decision', 'tick'), map(int, index[:5])))
        diagnostics.update(world_id=world_id, sim_time=int(index[4]) * self.manifest.physics_dt,
            phase=PHASES[index[5]], reset_source=RESET_SOURCES[index[6]], world={'costs': {}}, roles={})
        step = index[5] == 1
        for role in self.plan.experiment.bindings:
            diagnostics['roles'][role] = dict(action={} if step else None, reward=None, costs={} if step else None)
        poses = {}
        for i, field in enumerate(self.manifest.fields):
            value = _value(arrays, field, i, sample_index)
            if field.name in ('xpos', 'xmat'):
                poses[field.name] = value.copy()
                continue
            target = diagnostics['world'] if field.role_id is None else diagnostics['roles'][field.role_id]
            encoded = np.asarray(value).tolist()
            if field.name.startswith('action_'):
                if step:
                    if field.name == 'action_motion':
                        target['action']['motion'] = encoded
                    else:
                        target['action'].setdefault('radio', {})[field.name.removeprefix('action_')] = encoded
            elif field.cost_id is not None:
                if step:
                    target['costs'][field.cost_id] = encoded
            else:
                result = field.name in ('reward', 'world_reward', 'world_terminated', 'world_truncated',
                                       'world_end_reason', 'learning_done')
                target[field.name] = encoded if step or not result else None
        if not step:
            diagnostics['world']['costs'] = None
        objects = {}
        for obj in self.objects:
            position = poses['xpos'][obj.body_id]
            rotation = poses['xmat'][obj.body_id].reshape(3, 3)
            if obj.kind == 'environment':
                position = position + rotation @ np.asarray(obj.local_position)
                rotation = rotation @ np.asarray(obj.local_orientation_matrix).reshape(3, 3)
            slot = None if obj.role_id is None else self.plan.experiment.bindings[obj.role_id].role_slot_to_group_slot.index(obj.group_slot)
            objects[obj.object_id] = dict(position=position.tolist(), orientation_matrix=rotation.reshape(9).tolist(),
                                         role_id=obj.role_id, role_slot=slot)
        diagnostics['objects'] = objects
        return poses['xpos'], poses['xmat'], diagnostics

    def close(self):
        self.arrays.clear()
