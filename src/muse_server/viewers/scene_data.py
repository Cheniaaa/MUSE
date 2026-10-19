"""两个显示进程共用的冻结模型与刚体快照投影。"""
import hashlib
from pathlib import Path

import mujoco
import numpy as np


def load_model(path: Path, sha256: str, nbody: int) -> mujoco.MjModel:
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != sha256:
        raise ValueError('Preview model hash mismatch')
    model = mujoco.MjModel.from_binary_path(str(path))
    if model.nbody != nbody:
        raise ValueError('Preview model nbody mismatch')
    return model


def geom_rotations(model) -> np.ndarray:
    rotations = np.empty((model.ngeom, 3, 3))
    for index in range(model.ngeom):
        mujoco.mju_quat2Mat(rotations[index].reshape(9), model.geom_quat[index])
    return rotations


def apply_body_poses(model, data, xpos, xmat, local_rotations) -> None:
    data.xpos[:] = xpos
    data.xmat[:] = np.asarray(xmat).reshape(-1, 9)
    rotations = data.xmat.reshape(-1, 3, 3)[model.geom_bodyid]
    data.geom_xpos[:] = data.xpos[model.geom_bodyid] + np.einsum(
        'nij,nj->ni', rotations, model.geom_pos)
    data.geom_xmat[:] = (rotations @ local_rotations).reshape(-1, 9)
