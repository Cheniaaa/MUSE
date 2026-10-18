import numpy as np
import pytest
import torch

from test_flight_env import flight_env, flight_plan, actions


def test_final_three_vehicle_model_snapshot_mapping_and_disabled(flight_env, tmp_path):
    import mujoco
    from muse_runtime.preview import WorkerPreview
    from muse_sdk.run_protocol import WorkerControl, DisplaySettings, preview_socketpair, drain_preview
    env = flight_env
    events = []
    sender, receiver = preview_socketpair()
    preview = WorkerPreview('r', env, tmp_path, sender, lambda kind, payload: events.append((kind, payload)))
    try:
        preview.sample()
        with pytest.raises(BlockingIOError):
            receiver.recv(100)
        model = mujoco.MjModel.from_binary_path(str(tmp_path / preview.model.path))
        assert model.nmesh == env.sim.mj_model.nmesh > 0
        np.testing.assert_array_equal(model.mesh_vert, env.sim.mj_model.mesh_vert)
        # 每架实体写不同非零 yaw，验证原模型 body 顺序，不按目录重新装配。
        for index in range(model.njnt):
            if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE:
                address = model.jnt_qposadr[index]
                yaw = (index + 1) * .3
                env.sim.data.qpos[1, address:address+3] = torch.tensor([index * 2., 2., 1.5], device=env.device)
                env.sim.data.qpos[1, address+3:address+7] = torch.tensor([np.cos(yaw/2), 0., 0., np.sin(yaw/2)], device=env.device)
        env.sim.forward()
        original = env.sim.data.qpos.clone()
        preview.apply(WorkerControl(run_id='r', kind='preview', selection_seq=1,
                      display=DisplaySettings(enabled=True, world_id=1, rate_hz=30.)))
        preview.sample(force=True)
        header, raw = drain_preview(receiver, run_id='r', selection_seq=1, world_id=1, scene_hash=preview.model.sha256)
        pos = np.frombuffer(raw, dtype='<f4')[:model.nbody*3].reshape(1, model.nbody, 3)
        mat = np.frombuffer(raw, dtype='<f4')[model.nbody*3:].reshape(1, model.nbody, 3, 3)
        np.testing.assert_allclose(pos, env.sim.data.xpos[1:2].cpu().numpy())
        np.testing.assert_allclose(mat, env.sim.data.xmat[1:2].cpu().numpy().reshape(mat.shape))
        for i in range(model.njnt):
            body = model.jnt_bodyid[i]
            np.testing.assert_allclose(pos[0, body], [i * 2., 2., 1.5], atol=1e-6)
            yaw = (i + 1) * .3
            np.testing.assert_allclose(mat[0, body], [[np.cos(yaw),-np.sin(yaw),0],
                [np.sin(yaw),np.cos(yaw),0],[0,0,1]], atol=1e-6)
        assert torch.equal(original, env.sim.data.qpos)
        import viser
        from muse_server.viewers.scene_interaction import SceneInteraction
        server = viser.ViserServer(host='127.0.0.1', port=0, verbose=False)
        interaction = None
        try:
            interaction = SceneInteraction(server, model, preview.model.objects, emit=lambda _: None)
            interaction.update_frame(pos[0], mat[0], header.model_dump(mode='json'))
            rendered_geoms, entity_bodies, fixed_geoms, mesh_geoms = set(), set(), set(), set()
            for index, obj in enumerate(preview.model.objects):
                if obj.kind == 'entity':
                    entity_bodies.add(obj.body_id)
                for geom in obj.geom_ids:
                    rendered_geoms.add(geom)
                    handle = server.scene.get_handle_by_name(f'/muse/objects/{index}/geoms/{geom}')
                    assert handle is not None and handle.visible
                    body = model.geom_bodyid[geom]
                    np.testing.assert_allclose(handle.position, pos[0, body], atol=1e-6)
                    rotation = np.empty(9)
                    mujoco.mju_quat2Mat(rotation, handle.wxyz)
                    np.testing.assert_allclose(rotation.reshape(3, 3), mat[0, body], atol=1e-6)
                    if body == 0:
                        fixed_geoms.add(geom)
                        np.testing.assert_array_equal(handle.position, [0., 0., 0.])
                    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                        mesh_geoms.add(geom)
                        mesh = model.geom_dataid[geom]
                        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                        local_rotation = np.empty(9)
                        mujoco.mju_quat2Mat(local_rotation, model.geom_quat[geom])
                        expected = model.mesh_vert[start:start + count] @ local_rotation.reshape(3, 3).T + model.geom_pos[geom]
                        actual = interaction._meshes[geom].vertices
                        np.testing.assert_allclose(actual.min(0), expected.min(0), atol=1e-6)
                        np.testing.assert_allclose(actual.max(0), expected.max(0), atol=1e-6)
            assert rendered_geoms == set(range(model.ngeom))
            assert entity_bodies == set(model.jnt_bodyid)
            assert fixed_geoms and mesh_geoms
        finally:
            if interaction is not None:
                interaction.close()
            server.stop()
        assert events[-1][0] == 'preview_applied'
        preview.apply(WorkerControl(run_id='r', kind='preview', selection_seq=2,
                      display=DisplaySettings(enabled=False, world_id=1, rate_hz=30.)))
        # disabled 在提取前返回，即使 CPU copy 被禁止也不访问。
        from unittest.mock import patch
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('disabled copied to CPU')):
            preview.sample(force=True)
    finally:
        sender.close()
        receiver.close()
        env.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))


def test_preview_export_failure_isolated_but_gpu_copy_error_propagates(flight_env, tmp_path, monkeypatch):
    from muse_runtime import preview as module
    from muse_sdk.run_protocol import WorkerControl, DisplaySettings, preview_socketpair
    env = flight_env
    sender, receiver = preview_socketpair()
    events = []
    original = module.export_model
    try:
        def broken_export(*args, **kwargs):
            raise OSError('export denied')
        monkeypatch.setattr(module, 'export_model', broken_export)
        preview = module.WorkerPreview('r', env, tmp_path, sender, lambda *args: events.append(args))
        assert preview.error == 'export denied'
        assert events[-1][0] == 'preview_error'
        env.step(actions(env, 0.))
        preview.after_step()
        preview.sample(force=True)
        monkeypatch.setattr(module, 'export_model', original)
        preview = module.WorkerPreview('r', env, tmp_path, sender, lambda *args: None)
        preview.apply(WorkerControl(run_id='r', kind='preview', selection_seq=1,
                      display=DisplaySettings(enabled=True, world_id=0, rate_hz=30.)))
        def gpu_error(*args):
            raise RuntimeError('CUDA failure')
        monkeypatch.setattr(torch.Tensor, 'cpu', gpu_error)
        with pytest.raises(RuntimeError, match='CUDA failure'):
            preview.sample(force=True)
    finally:
        sender.close()
        receiver.close()


def test_episode_reset_zero_pairs_with_new_episode(flight_env, tmp_path):
    from muse_runtime.preview import WorkerPreview
    from muse_sdk.run_protocol import WorkerControl, DisplaySettings, preview_socketpair, drain_preview
    env = flight_env
    sender, receiver = preview_socketpair()
    preview = WorkerPreview('r', env, tmp_path, sender, lambda *args: None)
    try:
        preview.apply(WorkerControl(run_id='r', kind='preview', selection_seq=1,
                      display=DisplaySettings(enabled=True, world_id=0, rate_hz=30.)))
        env.step(actions(env, 0.))
        preview.after_step()
        preview.sample(force=True)
        old, _ = drain_preview(receiver, run_id='r', selection_seq=1, world_id=0, scene_hash=preview.model.sha256)
        # 下一成功 step 触发 world0 crash/reset；world1 不 reset。
        address = int(env.sim.mj_model.jnt_qposadr[0])
        env.sim.data.qpos[0, address+3:address+7] = torch.tensor([0., 1., 0., 0.], device=env.device)
        env.sim.forward()
        env.step(actions(env, 0.))
        preview.after_step()
        preview.sample(force=True)
        new, _ = drain_preview(receiver, run_id='r', selection_seq=1, world_id=0, scene_hash=preview.model.sha256)
        assert new.episode == old.episode + 1
        assert new.decision == new.tick == new.sim_time == 0
        assert preview.episodes[1] == 0
    finally:
        sender.close()
        receiver.close()
        env.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
