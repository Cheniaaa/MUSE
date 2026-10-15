from pathlib import Path

import pytest


@pytest.fixture(scope='module')
def gpu_stream():
    import torch
    import warp as wp

    wp.init()
    previous = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(previous)
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        yield stream
        stream.synchronize()


@pytest.fixture(scope='module')
def cf2_scene_factory(gpu_stream):
    import mujoco
    from mjlab.entity import EntityCfg
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import Simulation, SimulationCfg
    from mjlab.sim.sim import MujocoCfg

    asset = Path(__file__).parents[2] / 'src/muse_builtins/assets/bitcraze_crazyflie_2/cf2.xml'
    original = mujoco.MjModel.from_xml_path(str(asset))

    class OriginalOptions(MujocoCfg):
        def apply(self, model):
            super().apply(model)
            # mjlab的配置枚举未列RK4，但固定版Warp支持；显式保留原资产选项。
            model.opt.integrator = original.opt.integrator
            model.opt.density = original.opt.density
            model.opt.viscosity = original.opt.viscosity

    def scenery(spec):
        spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, .1])

    def build(worlds, count=1):
        entities = {
            'cf2' if count == 1 else f'cf2_{slot}': EntityCfg(
                spec_fn=lambda: mujoco.MjSpec.from_file(str(asset)),
                init_state=EntityCfg.InitialStateCfg(pos=(0., 2. * slot, 1.)))
            for slot in range(count)
        }
        scene = Scene(SceneCfg(num_envs=worlds, entities=entities, spec_fn=scenery), device='cuda:0')
        # init_state不修改自由体编译qpos0，编译前也必须应用场景placement。
        for name, cfg in entities.items():
            scene.spec.body(f'{name}/cf2').pos = cfg.init_state.pos
        sim = Simulation(worlds, SimulationCfg(nconmax=128, njmax=512, mujoco=OriginalOptions(timestep=.002)),
                         spec=scene.spec, device='cuda:0')
        scene.initialize(sim.mj_model, sim.model, sim.data)
        sim.set_sensor_context(scene.sensor_context)
        return scene, sim

    return build


@pytest.fixture(scope='module')
def cf2_scene_sim(cf2_scene_factory):
    return cf2_scene_factory(1)


@pytest.fixture
def cf2_sim(cf2_scene_sim):
    import torch

    scene, sim = cf2_scene_sim
    sim.reset()
    scene.reset()
    entity = scene['cf2']
    entity.write_root_state_to_sim(entity.data.default_root_state)
    sim.data.ctrl[:] = 0
    scene.write_data_to_sim()
    sim.forward()
    yield scene, sim
    assert torch.isfinite(sim.data.qpos[:]).all()
    assert torch.isfinite(sim.data.qvel[:]).all()
    assert int(sim.data.nacon[0]) <= sim.data.naconmax
    assert (sim.data.nefc[:] <= sim.data.njmax).all()
