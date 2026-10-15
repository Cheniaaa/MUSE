"""真实物理epoch的左端快照与有限服务积分。"""
import torch

from test_wireless_env import wireless_env


def test_real_env_epoch_geometry_matches_left_ticks(wireless_env, monkeypatch):
    env, requests = wireless_env
    # 初态给出可辨识的真实平移速度；小速度setpoint在8ms内尚未形成可见位移。
    motion = env.action_manager.get_term('scout_group')
    env.sim.data.qvel[:, motion.qvel_ids[0, 0]] = 1.
    env.sim.forward()
    env.geometry_query.refresh(torch.ones(env.num_envs, dtype=torch.bool, device=env.device),
        sample_tick=torch.zeros(env.num_envs, dtype=torch.int64, device=env.device), present=env._geometry_present)
    counts = {}
    def count(obj, name, label):
        original = getattr(obj, name)
        def wrapped(*args, **kwargs):
            counts[label] = counts.get(label, 0) + 1
            return original(*args, **kwargs)
        monkeypatch.setattr(obj, name, wrapped)
    count(env.sim, 'step', 'physics')
    count(env.action_manager, 'process_action', 'process')
    count(env.action_manager, 'apply_action', 'apply')
    for name in ('observation_manager', 'reward_manager', 'termination_manager', 'command_manager'):
        count(getattr(env, name), 'compute', name)
    before = env.geometry_query.view.distance_m.clone()
    out = env.step(requests)
    records = env.wireless.models['radio']._model.evaluations
    assert [tick.tolist() for tick, _, _ in records] == [[0, 0], [4, 4], [8, 8]]
    assert [interval[0].item() for _, interval, _ in records] == [.004, .004, .002]
    torch.testing.assert_close(records[0][2], before)
    assert not torch.equal(records[0][2], records[2][2])
    assert counts == dict(physics=10, process=1, apply=10,
        observation_manager=1, reward_manager=1, termination_manager=1, command_manager=1)
    assert out.transition.wireless.radio_sample_tick.tolist() == [8, 8]
    assert env.physics_outputs.sim_time.tolist() == [.01, .01]
    torch.testing.assert_close(out.transition.wireless.sensing['fim'].data,
        .03*torch.eye(2, dtype=torch.float64, device=env.device).expand(2, 1, 2, 2))
