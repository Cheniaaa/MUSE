"""独立学习快照与同步原子发布；登记由Server消费Worker事件完成。"""
import hashlib
import os
from pathlib import Path
from uuid import uuid4

import torch

from muse_sdk.experiment import learning_contract
from muse_sdk.run_protocol import CheckpointArtifact


def snapshot_learning_state(value):
    """仅复制tensor/primitive容器，不序列化模块或其他活对象。"""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            raise TypeError('Learning state dictionary keys must be str or int')
        return {key: snapshot_learning_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [snapshot_learning_state(item) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    raise TypeError(f'Unsupported learning state type: {type(value).__name__}')


def publish_checkpoint(plan, algorithm, *, steps, updates) -> CheckpointArtifact:
    envelope = dict(format_revision=1, run_id=plan.run_id, experiment_id=plan.experiment_id,
                    learning_contract=learning_contract(plan.experiment), steps=steps, updates=updates,
                    learning_state=snapshot_learning_state(algorithm.learning_state_dict()))
    identity = uuid4().hex
    relative = Path('checkpoints') / f'{identity}.pt'
    final = Path(plan.artifact_directory) / relative
    final.parent.mkdir(exist_ok=True)
    temporary = final.with_suffix('.tmp')
    try:
        with temporary.open('xb') as stream:
            torch.save(envelope, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, final)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    with final.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return CheckpointArtifact(id=f'{plan.run_id}:checkpoint:{identity}', run=plan.run_id,
                              path=relative.as_posix(), size=final.stat().st_size, sha256=digest)


def load_checkpoint(plan, algorithm) -> None:
    reference = plan.checkpoint
    with Path(reference.path).open('rb') as stream:
        if (os.fstat(stream.fileno()).st_size != reference.size
                or hashlib.file_digest(stream, 'sha256').hexdigest() != reference.sha256):
            raise ValueError('Checkpoint file integrity mismatch')
        stream.seek(0)
        envelope = torch.load(stream, weights_only=True, map_location='cpu')
    if envelope['format_revision'] != 1:
        raise ValueError('Unsupported checkpoint format revision')
    if envelope['run_id'] != reference.artifact_id.split(':', 1)[0]:
        raise ValueError('Checkpoint source run mismatch')
    if envelope['learning_contract'] != learning_contract(plan.experiment):
        raise ValueError('Checkpoint learning contract mismatch')
    algorithm.load_learning_state_dict(envelope['learning_state'])
