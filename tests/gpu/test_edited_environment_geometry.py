"""B9 串行执行：真实派生包经生产装配进入 CUDA contact 与 RF 查询。"""
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
import yaml

from muse_sdk.content import hash_package
from muse_sdk.experiment import EnvironmentBundle, Pose, ResourceRef, ScenarioDefinition, Topology
from muse_sdk.geometry import GeometryPoint, PropagationPath, PropagationPathLayout, ResolvedGeometry
from muse_sdk.scene_edit import BoxEdit, EnvironmentEdit


def test_moved_wall_changes_runtime_contact_and_los_with_identical_cpu_geometry(tmp_path, gpu_stream):
    from mjlab.scene import Scene
    from mjlab.sim import Simulation
    from muse_runtime.environment_edit import derive_environment
    from muse_runtime.network.geometry import GeometryQuery
    from muse_runtime.scene import assemble_scene

    base = tmp_path / 'base'
    base.mkdir()
    (base / 'environment.yaml').write_text('scene: scene.xml\nrf_geometry: {rf_blocking: [wall]}\n')
    (base / 'scene.xml').write_text('''<mujoco><worldbody>
      <geom name="wall" type="box" size=".1 1 1" pos="2 0 1"/>
    </worldbody></mujoco>''')
    reference = ResourceRef(resource_id='wall', revision=1, sha256=hash_package(base))
    derived = tmp_path / 'derived'
    derive_environment(base, EnvironmentEdit(base=reference, removed=[], upserts={'wall': BoxEdit(
        label='移开的墙', pose=Pose(position=[2., 3., 1.], orientation_wxyz=[1., 0., 0., 0.]),
        size_m=[.2, 2., 2.], collision=True, rf_blocking=True)}), derived)
    # 两个自由球体充当原生 contact probe 和真实 RF endpoint，不创建学习器。
    asset = tmp_path / 'probe'
    asset.mkdir()
    (asset / 'probe.xml').write_text('''<mujoco><worldbody><body name="probe"><freejoint/>
      <geom name="probe_geom" type="sphere" size=".1" mass="1"/>
    </body></worldbody></mujoco>''')
    scenario = ScenarioDefinition.model_validate(dict(
        entity_groups=[dict(group_id='probe_group', model_id='probe', model_revision=1, count=2,
                            placement_id='positions', controller_id='none', sensor_profile_id='none')],
        placements={'positions': [dict(position=position, orientation_wxyz=[1., 0., 0., 0.])
                                  for position in ([1.85, 0., 1.], [4., 0., 1.])]},
        scene=dict(physics_dt=.002, control_substeps=1, integrator='RK4', density=0.,
                   viscosity=0., precision='float32'), command_targets_world={}))
    experiment = SimpleNamespace(scenario=scenario, topology=Topology(world_count=1),
        entity_models={'probe': SimpleNamespace(physics_asset='probe.xml', root_body='probe')},
        sensors={}, activation=[], environment_scene='scene.xml')
    layout = PropagationPathLayout(points=tuple(GeometryPoint(group_id='probe_group', slot=slot,
        device_id='radio', body='probe', position_m=(0., 0., 0.), assumption='test contact probe')
        for slot in range(2)), paths=(PropagationPath(path_id='p0', tx=0, rx=1),),
        desired=(), sensing=(), interference=())
    contacts, visibility = [], []
    for environment in (base, derived):
        cfg, simulation_cfg = assemble_scene(experiment, {'environment': str(environment),
                                                          'entity_model:probe': str(asset)})
        cpu_scene = Scene(cfg, device='cpu')
        cpu_model = cpu_scene.compile()
        cpu_data = mujoco.MjData(cpu_model)
        mujoco.mj_forward(cpu_model, cpu_data)
        sim = Simulation(1, simulation_cfg, spec=cpu_scene.spec, device='cuda:0')
        sim.forward()
        for field in ('geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_contype', 'geom_conaffinity'):
            np.testing.assert_allclose(getattr(cpu_model, field), getattr(sim.mj_model, field), atol=2e-6)
        np.testing.assert_allclose(cpu_data.geom_xpos, sim.data.geom_xpos[0].cpu().numpy(), atol=2e-6)
        np.testing.assert_allclose(cpu_data.geom_xmat, sim.data.geom_xmat[0].cpu().numpy().reshape(-1, 9), atol=2e-6)
        wall, probe = (sim.mj_model.geom(name).id for name in ('environment/wall', 'probe_group_0/probe_geom'))
        pairs = sim.data.contact.geom[:int(sim.data.nacon[0])]
        contact = bool(((pairs == wall).any(-1) & (pairs == probe).any(-1)).any().item())
        cpu_contact = any({int(c.geom1), int(c.geom2)} == {wall, probe} for c in cpu_data.contact)
        assert cpu_contact == contact
        contacts.append(contact)
        bundle = EnvironmentBundle.model_validate(yaml.safe_load((environment / 'environment.yaml').read_text()))
        query = GeometryQuery(ResolvedGeometry(layout=layout, endpoint_epsilon_m=1e-4,
            blockers=tuple(bundle.rf_geometry.rf_blocking)), sim, physics_dt=.002, device='cuda:0')
        view = query.refresh(torch.ones(1, dtype=torch.bool, device='cuda:0'),
            sample_tick=torch.zeros(1, dtype=torch.int64, device='cuda:0'),
            present=torch.ones((1, 2), dtype=torch.bool, device='cuda:0'))
        assert view.valid.all()
        visibility.append(bool(view.los[0, 0].item()))
    assert contacts == [True, False]
    assert visibility == [False, True]
    assert hash_package(base) == reference.sha256
