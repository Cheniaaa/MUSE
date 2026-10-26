"""协议测试的完整双槽控制；默认只启用本用例关注的Viser。"""
from muse_sdk.run_protocol import DisplaySettings, PreviewSelection, PreviewSelections, WorkerControl


def preview_model(env, directory):
    from muse_runtime.scene_snapshot import export_model, scene_objects
    return export_model(env.sim.mj_model, directory,
                        objects=scene_objects(env.experiment, env.scene, env.sim.mj_model))


def preview_control(run_id, seq, *, world=0, rate=10., enabled=True):
    return WorkerControl(run_id=run_id, kind='preview', preview=PreviewSelections(
        viser=PreviewSelection(selection_seq=seq,
            display=DisplaySettings(enabled=enabled, world_id=world, rate_hz=rate)),
        native=PreviewSelection(selection_seq=0,
            display=DisplaySettings(enabled=False, world_id=0, rate_hz=10.))))


def native_snapshot_case():
    """独立参考状态：覆盖非零act/mocap/外力/equality及原生sensor来源。"""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_string('''
    <mujoco model="MUSE snapshot fixture">
      <size nuserdata="1"/>
      <worldbody>
        <body name="slider" pos=".2 .1 .3">
          <joint name="slide" type="slide" axis="1 0 0"/>
          <geom type="box" size=".1 .1 .1" mass="1"/>
          <site name="tip" pos=".1 .2 .3" size=".02"/>
        </body>
        <body name="target" mocap="true"><geom type="sphere" size=".05"/></body>
      </worldbody>
      <actuator><general name="motor" joint="slide" dyntype="filter" dynprm=".1"
          ctrllimited="true" ctrlrange="-1 1"/></actuator>
      <equality><joint joint1="slide" polycoef=".45 0 0 0 0"/></equality>
      <sensor><jointpos joint="slide"/></sensor>
    </mujoco>''')
    data = mujoco.MjData(model)
    data.time = 7.5
    data.qpos[:] = .45
    data.qvel[:] = .12
    data.act[:] = .35
    data.ctrl[:] = .2
    data.qfrc_applied[:] = .3
    data.xfrc_applied[1] = [.1, .2, .3, .4, .5, .6]
    data.mocap_pos[:] = [.7, .8, .9]
    data.mocap_quat[:] = [.5, .5, .5, .5]
    data.eq_active[:] = 1
    mujoco.mj_forward(model, data)
    data.sensordata[:] = 123.
    raw = np.concatenate([data.qpos, data.qvel, data.act, data.ctrl, data.qfrc_applied,
        data.xfrc_applied.ravel(), data.mocap_pos.ravel(), data.mocap_quat.ravel(),
        data.eq_active, data.sensordata]).astype('<f4').tobytes()
    return model, data, raw
