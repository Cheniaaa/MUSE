"""固定端点与UAV/UGV共用Scene、reset及只读preview。"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flight_helpers import add_fixed_anchors, archive


@pytest.fixture(scope='module')
def fixed_plan(pipeline, tmp_path_factory):
    from muse_server.resolver import resolve_experiment
    catalog, _ = pipeline
    root = tmp_path_factory.mktemp('fixed-entities')
    example = Path(__file__).resolve().parents[2] / 'examples/heterogeneous'
    request = json.loads((example / 'request_train.json').read_text())
    for kind in ('environment', 'training_package'):
        resource = catalog.register_archive(kind, archive(example / kind, root / f'{kind}.zip'))
        request[kind + '_id'], request[kind + '_revision'] = resource.resource_id, resource.revision
    add_fixed_anchors(request, index=1)
    request['term_config'] = {'world': {'termination': {'episode_limit': {'config': {'episode_max_steps': 500}}}}}
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment, source=request)
    return catalog.create_run_plan(identity, mode='TRAIN', device='cuda:0', max_decision_steps=8)


@pytest.fixture(scope='module')
def fixed_env(fixed_plan, gpu_stream):
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    env = MuseEnv(fixed_plan.experiment, RuntimeContext('cuda:0', fixed_plan.resource_paths, {}),
                  WorldRng(7, 2, 'cuda:0'))
    yield env
    env.close()


def commands(env):
    from muse_sdk.runtime_types import MotionAction
    return {'scout': MotionAction(torch.zeros(2, 1, 4, device=env.device)),
            'rover': MotionAction(torch.tensor([.1, 0.], device=env.device).expand(2, 1, 2).clone())}


def test_fixed_endpoints_allow_motion_and_preserve_subset_reset(fixed_env):
    env = fixed_env
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))
    model = env.sim.mj_model
    bodies = [model.body(f'anchor_group_{slot}/station').id for slot in range(2)]
    markers = [model.geom(f'anchor_group_{slot}/marker').id for slot in range(2)]
    expected = torch.tensor([[0., 1., 1.], [3., 2., 1.5]], device=env.device).expand(2, 2, 3)
    torch.testing.assert_close(env.sim.data.xpos[:, bodies], expected)
    np.testing.assert_array_equal(model.body_jntnum[bodies], [0, 0])
    start = env.role_positions_world['rover'].clone()
    # 与scout重叠的marker若意外成为碰撞体，本段会触发world reset。
    for _ in range(24):
        output = env.step(commands(env))
        assert not output.transition.world_reset.any()
    assert (env.role_positions_world['rover'][..., 0] - start[..., 0] > .05).all()
    torch.testing.assert_close(env.sim.data.xpos[:, bodies], expected)
    assert env.action_manager.action.shape == (2, 6)
    contact = env.sim.data.contact
    active = torch.arange(contact.worldid.shape[0], device=env.device) < env.sim.data.nacon[0]
    assert not torch.isin(contact.geom[active], torch.tensor(markers, device=env.device)).any()
    state = {key: getattr(env.sim.data, key)[1].clone() for key in ('qpos', 'qvel', 'ctrl')}
    action = env.action_manager.action[1].clone()
    env.reset(torch.tensor([True, False], device=env.device))
    for key, value in state.items():
        assert torch.equal(getattr(env.sim.data, key)[1], value)
    assert torch.equal(env.action_manager.action[1], action)
    assert env.action_manager.action[0].count_nonzero() == 0
    torch.testing.assert_close(env.sim.data.xpos[:, bodies], expected)


def test_fixed_endpoints_survive_timeout_and_real_learning(fixed_env):
    from muse_builtins.algorithms.ippo import IPPO
    from muse_runtime.driver import run
    env = fixed_env
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))
    timeout = next(cfg for cfg in env.termination_manager.cfg.values() if cfg.time_out)
    old = timeout.params['episode_max_steps']
    timeout.params['episode_max_steps'] = 4
    try:
        for _ in range(4):
            output = env.step(commands(env))
        assert output.transition.world_truncated.tolist() == [True, True]
        algorithm = IPPO(env.experiment.algorithm_config)
        algorithm.initialize(env.experiment, SimpleNamespace(device=env.device), env.rng)
        result = run(env, algorithm, lambda: None, lambda *args: None,
                     mode='TRAIN', collection_steps=4, max_decision_steps=8)
        assert result == dict(state='COMPLETED', steps=8, updates=2)
        assert set(algorithm.children) == {'scout', 'rover'}
    finally:
        timeout.params['episode_max_steps'] = old


def test_fixed_endpoints_in_exported_model_and_world_preview(fixed_env, tmp_path):
    import mujoco
    from muse_runtime.preview import WorkerPreview
    from muse_sdk.run_protocol import drain_preview, preview_socketpair
    from preview_helpers import preview_control, preview_model
    env = fixed_env
    sender, receiver = preview_socketpair()
    try:
        preview = WorkerPreview('fixed', env, sender, lambda *args: None, model=preview_model(env, tmp_path))
        assert preview.error is None
        preview.apply(preview_control('fixed', 1, world=1))
        preview.sample(force=True)
        header, raw = drain_preview(receiver, run_id='fixed', selections=preview.selections,
                                    scene_hash=preview.model.sha256)['viser']
        model = mujoco.MjModel.from_binary_path(str(tmp_path / preview.model.path))
        pos = np.frombuffer(raw, dtype='<f4')[:model.nbody*3].reshape(model.nbody, 3)
        mat = np.frombuffer(raw, dtype='<f4')[model.nbody*3:].reshape(model.nbody, 3, 3)
        ids = [model.body(f'anchor_group_{slot}/station').id for slot in range(2)]
        assert header.world_id == 1
        np.testing.assert_allclose(pos[ids], [[0., 1., 1.], [3., 2., 1.5]])
        np.testing.assert_allclose(mat[ids], [np.eye(3), np.diag([-1., -1., 1.])], atol=1e-6)
        for slot in range(2):
            geom = model.geom(f'anchor_group_{slot}/marker').id
            assert model.geom_rgba[geom, 3] > 0
            assert model.geom_bodyid[geom] == ids[slot]
    finally:
        sender.close()
        receiver.close()
