import hashlib
from pathlib import Path

import pytest
import torch


class LearningState:
    def __init__(self):
        self.state = {'weight': torch.tensor([1.]), 'iteration': 2, 'optimizer': {0: (None, True)}}

    def learning_state_dict(self):
        return self.state

    def load_learning_state_dict(self, state):
        self.state = state


def reference(plan, artifact):
    from muse_sdk.run_protocol import CheckpointReference
    return CheckpointReference(artifact_id=artifact.id,
        path=str(Path(plan.artifact_directory) / artifact.path), size=artifact.size, sha256=artifact.sha256)


def test_snapshot_detaches_borrowed_learning_tensor():
    from muse_runtime.checkpoint import snapshot_learning_state
    weight = torch.tensor([1.], requires_grad=True)
    saved = snapshot_learning_state({'weight': weight, 'iteration': 3})
    with torch.no_grad():
        weight.add_(9.)
    assert saved['weight'].item() == 1.
    assert saved['weight'].device.type == 'cpu'
    assert not saved['weight'].requires_grad


@pytest.mark.parametrize('state', [object(), torch.nn.Linear(1, 1), {('bad',): 1}])
def test_snapshot_rejects_objects_and_nonprimitive_keys(state):
    from muse_runtime.checkpoint import snapshot_learning_state
    with pytest.raises(TypeError):
        snapshot_learning_state(state)


def test_publish_and_load_independent_checkpoints(checkpoint_plan):
    from muse_runtime.checkpoint import load_checkpoint, publish_checkpoint
    source = LearningState()
    first = publish_checkpoint(checkpoint_plan, source, steps=8, updates=2)
    source.state['weight'].add_(5.)
    second = publish_checkpoint(checkpoint_plan, source, steps=12, updates=3)
    assert first.id != second.id and first.path != second.path
    path = Path(checkpoint_plan.artifact_directory) / first.path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first.sha256
    assert path.stat().st_size == first.size
    envelope = torch.load(path, weights_only=True, map_location='cpu')
    assert envelope['steps'] == 8 and envelope['updates'] == 2
    assert not list(path.parent.glob('*.tmp'))
    target = LearningState()
    target.state = None
    checkpoint_plan.checkpoint = reference(checkpoint_plan, first)
    load_checkpoint(checkpoint_plan, target)
    assert target.state['weight'].item() == 1.
    assert target.state['iteration'] == 2


@pytest.mark.parametrize('operation', ['save', 'fsync', 'replace'])
def test_failed_write_does_not_publish(checkpoint_plan, monkeypatch, operation):
    from muse_runtime import checkpoint
    def fail(*args, **kwargs):
        raise OSError('injected write failure')
    owner = checkpoint.torch if operation == 'save' else checkpoint.os
    monkeypatch.setattr(owner, operation, fail)
    with pytest.raises(OSError, match='injected'):
        checkpoint.publish_checkpoint(checkpoint_plan, LearningState(), steps=8, updates=2)
    assert not list(Path(checkpoint_plan.artifact_directory).rglob('*.pt'))
    assert not list(Path(checkpoint_plan.artifact_directory).rglob('*.tmp'))


@pytest.mark.parametrize('mutation', ['size', 'hash', 'format', 'contract', 'identity'])
def test_load_rejects_corrupt_or_incompatible_checkpoint(checkpoint_plan, mutation):
    from muse_runtime.checkpoint import load_checkpoint, publish_checkpoint
    artifact = publish_checkpoint(checkpoint_plan, LearningState(), steps=8, updates=2)
    ref = reference(checkpoint_plan, artifact)
    if mutation == 'size':
        ref.size += 1
    elif mutation == 'hash':
        ref.sha256 = '0' * 64
    else:
        path = Path(ref.path)
        data = torch.load(path, weights_only=True)
        if mutation == 'format':
            data['format_revision'] = 99
        elif mutation == 'identity':
            data['run_id'] = 'other'
        else:
            data['learning_contract']['algorithm_id'] = 'other'
        torch.save(data, path)
        ref.size = path.stat().st_size
        ref.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint_plan.checkpoint = ref
    target = LearningState()
    with pytest.raises(ValueError):
        load_checkpoint(checkpoint_plan, target)
    assert target.state['weight'].item() == 1.
