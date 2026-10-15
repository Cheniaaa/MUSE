from dataclasses import dataclass
from pathlib import Path
import os

import mujoco
from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import MujocoCfg


@dataclass(kw_only=True)
class FlightOptions(MujocoCfg):
    flight_integrator: str
    density: float
    viscosity: float

    def apply(self, model):
        super().apply(model)
        model.opt.integrator = getattr(mujoco.mjtIntegrator, 'mjINT_' + self.flight_integrator)
        model.opt.density = self.density
        model.opt.viscosity = self.viscosity


def entity_name(group, slot):
    return f'{group}_{slot}'


def assemble_scene(experiment, resource_paths):
    entities = {}
    placements = {}
    for group in experiment.scenario.entity_groups:
        asset = Path(resource_paths['entity_model:' + group.model_id]) / experiment.entity_models[group.model_id].physics_asset
        root_body = experiment.entity_models[group.model_id].root_body
        range_outputs = [output for output in experiment.sensors[group.group_id]['outputs'].values()
                         if output['sensor_type'] == 'rangefinder'] if 'sensor:' + group.group_id in experiment.activation else []
        for slot, pose in enumerate(experiment.scenario.placements[group.placement_id]):
            name = entity_name(group.group_id, slot)
            def load(path=asset, root=root_body, outputs=range_outputs):
                spec = mujoco.MjSpec.from_file(str(path))
                for key in list(spec.keys):
                    spec.delete(key)
                for output in outputs:
                    spec.body(root).add_site(name=output['mount'],
                        pos=output['position'], quat=output['orientation_wxyz'])
                    sensor = spec.add_sensor(name=output['sensor_name'],
                        type=mujoco.mjtSensor.mjSENS_RANGEFINDER, objtype=mujoco.mjtObj.mjOBJ_SITE,
                        objname=output['mount'])
                    # 3.11要求显式dist；在Entity装配时添加，Scene才会自动包装BuiltinSensor。
                    sensor.intprm[0] = 1 << int(mujoco.mjtRayDataField.mjRAYDATA_DIST)
                return spec
            entities[name] = EntityCfg(spec_fn=load, init_state=EntityCfg.InitialStateCfg(
                pos=tuple(pose.position), rot=tuple(pose.orientation_wxyz)))
            placements[name] = (pose, root_body)
    def configure(spec):
        root = Path(resource_paths['environment'])
        scene_path = root / experiment.environment_scene
        assets = {os.path.relpath(path, scene_path.parent): path.read_bytes()
                  for path in root.rglob('*') if path.is_file()}
        environment = mujoco.MjSpec.from_string(scene_path.read_text(), include=assets, assets=assets)
        spec.attach(environment, prefix='environment/', frame=spec.worldbody.add_frame())
        for name, (pose, root_body) in placements.items():
            body = spec.body(f'{name}/{root_body}')
            # 固定体的placement已由mjlab写入mocap根；自由体还需同步编译初态。
            if any(joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in body.joints):
                body.pos = pose.position
                body.quat = pose.orientation_wxyz
    scene = SceneCfg(num_envs=experiment.topology.world_count, entities=entities, spec_fn=configure)
    cfg = experiment.scenario.scene
    simulation = SimulationCfg(nconmax=128, njmax=512, mujoco=FlightOptions(
        timestep=cfg.physics_dt, flight_integrator=cfg.integrator,
        density=cfg.density, viscosity=cfg.viscosity))
    return scene, simulation
