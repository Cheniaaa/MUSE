"""独立 CPU MAPPO 契约：不依赖示例包、resolver 或 GPU。"""
import copy
import importlib.util
from types import SimpleNamespace as NS

import pytest
import torch

from muse_sdk.runtime_types import Observations, RoleMask, TransitionView
from muse_runtime.checkpoint import snapshot_learning_state


def make_algorithm(batch=2, costs=None, **overrides):
    from muse_sdk.experiment import CostLayout
    assert importlib.util.find_spec('muse_builtins.algorithms.mappo') is not None, 'MAPPO implementation missing'
    from muse_builtins.algorithms.mappo import MAPPO, MAPPOConfig
    config = MAPPOConfig(collection_steps=4, hidden_dims=[12], num_learning_epochs=2,
                         num_mini_batches=3).model_dump()
    config.update(overrides)
    specs = {r: NS(shape=[batch, n, width], minimum=[-2.] * width, maximum=[3.] * width)
             for r, n, width in [('scout', 2, 4), ('guard', 1, 2)]}
    actors = {r: [NS(path=['observations', 'actors', r, 'target'],
                     tensor=NS(shape=[batch, s.shape[1], width]))]
              for (r, s), width in zip(specs.items(), [3, 5])}
    critics = {'team': [NS(path=['observations', 'critic_inputs', 'team', 'target'],
                          tensor=NS(shape=[batch, 3, 2]))]}
    experiment = NS(root_seed=17, action_specs=specs, algorithm_inputs=NS(actors=actors, critics=critics),
                    training_schedule=NS(collection_steps=config['collection_steps']),
                    costs=CostLayout.model_validate(costs or {'roles': {}, 'world': {}}))
    algorithm = MAPPO(config)
    algorithm.initialize(experiment, NS(device='cpu'), None)
    obs = Observations({r: {'target': torch.arange(batch * s.shape[1] * width).float().reshape(
        batch, s.shape[1], width) / 10} for (r, s), width in zip(specs.items(), [3, 5])},
        {'team': {'target': torch.arange(batch * 6).float().reshape(batch, 3, 2) / 7}})
    masks = {r: RoleMask(*(torch.ones(s.shape[:2], dtype=torch.bool) for _ in range(3)))
             for r, s in specs.items()}
    return algorithm, obs, masks


def transition(obs, masks, step=0, costs=None):
    batch = obs.critic_inputs['team']['target'].shape[0]
    return TransitionView(masks, obs, {r: torch.full(m.present.shape, -999.) for r, m in masks.items()},
                          torch.arange(batch).float() + step + 1,
                          torch.zeros(batch, dtype=torch.bool), torch.zeros(batch, dtype=torch.bool),
                          torch.zeros(batch, dtype=torch.long),
                          {r: torch.zeros_like(m.present) for r, m in masks.items()},
                          torch.zeros(batch, dtype=torch.bool),
                          costs if costs is not None else {'roles': {}, 'world': {}}, wireless=None)


def collect(algorithm, obs, masks):
    for step in range(algorithm.collection_steps):
        result = algorithm.act(obs, masks, None, False)
        algorithm.process_env_step(transition(obs, masks, step), result.act_aux)


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_registration_and_descriptor_are_cpu_lightweight():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c', '''
import sys
from muse_builtins.algorithms import registrations
assert 'mappo' in registrations(), 'MAPPO registration missing'
desc = registrations()['mappo'].describe()
assert desc['actor_observation'] == 'all_local'
assert desc['critic_inputs'] == 'all_centralized'
assert desc['reward_inputs'] == ['world']
assert not any(x in sys.modules for x in ('torch', 'rsl_rl', 'mjlab', 'warp'))
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('sharing', [True, False])
def test_owner_partition_central_isolation_ratio_and_world_reward(sharing):
    algorithm, obs, masks = make_algorithm(share_actor_parameters=sharing)
    assert len(algorithm.children) == (2 if sharing else 3)
    assert len({id(c.optimizer) for c in algorithm.children.values()}) == len(algorithm.children)
    all_ids = []
    for child in algorithm.children.values():
        ids = {id(p) for group in child.optimizer.param_groups for p in group['params']}
        expected = {id(p) for model in (child.actor, child.critic) for p in model.parameters()}
        assert ids == expected
        all_ids.extend(ids)
    assert len(all_ids) == len(set(all_ids))
    changed = copy.deepcopy(obs)
    changed.critic_inputs['team']['target'].add_(10)
    for owner, child in algorithm.children.items():
        original = algorithm._observations(obs, owner)
        central = algorithm._observations(changed, owner)
        assert_tree_equal(child.actor(original), child.actor(central))
        assert not torch.equal(child.critic(original), child.critic(central))
    result = algorithm.act(obs, masks, None, False)
    assert result.role_actions['scout'].motion.shape == (2, 2, 4)
    assert result.role_actions['guard'].motion.shape == (2, 1, 2)
    for owner, child in algorithm.children.items():
        role, slot = algorithm.owners[owner]
        pending = child.storage.pending
        raw = pending.actions
        assert pending.actions_log_prob.shape == (raw.shape[0],)
        child.actor(pending.observations, stochastic_output=True)
        torch.testing.assert_close((child.actor.get_output_log_prob(raw) - pending.actions_log_prob).exp(),
                                   torch.ones(raw.shape[0]))
        physical = result.role_actions[role].motion
        if slot is not None:
            physical = physical[:, slot:slot + 1]
        torch.testing.assert_close(physical.flatten(0, 1), -2 + 5 * raw)
    algorithm.process_env_step(transition(obs, masks), result.act_aux)
    for owner, child in algorithm.children.items():
        _, slot = algorithm.owners[owner]
        want = [1., 1., 2., 2.] if owner == 'scout' and slot is None else [1., 2.]
        torch.testing.assert_close(child.storage.rewards[0, :, 0], torch.tensor(want))


@pytest.mark.parametrize('sharing', [True, False])
@pytest.mark.parametrize('recurrent', [None, 'gru', 'lstm'])
def test_real_updates_masks_and_frozen_adam_actor(sharing, recurrent):
    algorithm, obs, masks = make_algorithm(share_actor_parameters=sharing,
        recurrent=None if recurrent is None else {'type': recurrent, 'hidden_size': 7})
    before = snapshot_learning_state(algorithm.learning_state_dict())
    collect(algorithm, obs, masks)
    metrics = algorithm.update()
    assert metrics.keys() == algorithm.children.keys()
    for owner, child in algorithm.children.items():
        for kind in ('actor', 'critic'):
            assert any(not torch.equal(v, getattr(child, kind).state_dict()[k])
                       for k, v in before['roles'][owner][kind].items())
    before = snapshot_learning_state(algorithm.learning_state_dict())
    for mask in masks.values():
        mask.controllable.zero_()
    collect(algorithm, obs, masks)
    metrics = algorithm.update()
    for owner, child in algorithm.children.items():
        assert metrics[owner]['actor_samples'] == 0
        assert metrics[owner]['critic_samples'] > 0
        assert_tree_equal(before['roles'][owner]['actor'], child.actor.state_dict())
        assert any(not torch.equal(v, child.critic.state_dict()[k])
                   for k, v in before['roles'][owner]['critic'].items())
    for mask in masks.values():
        mask.present.zero_()
    before = snapshot_learning_state(algorithm.learning_state_dict())
    collect(algorithm, obs, masks)
    algorithm.update()
    for owner, child in algorithm.children.items():
        assert_tree_equal(before['roles'][owner]['critic'], child.critic.state_dict())


@pytest.mark.parametrize('sharing', [True, False])
def test_checkpoint_rng_lazy_adam_batch_change_and_frozen_eval(sharing):
    global_before = torch.random.get_rng_state().clone()
    algorithm, obs, masks = make_algorithm(share_actor_parameters=sharing)
    collect(algorithm, obs, masks)
    algorithm.update()
    assert torch.equal(global_before, torch.random.get_rng_state())
    state = snapshot_learning_state(algorithm.learning_state_dict())
    fresh, fresh_obs, fresh_masks = make_algorithm(batch=3, share_actor_parameters=sharing)
    fresh.load_learning_state_dict(state)
    assert_tree_equal(state, fresh.learning_state_dict())
    assert all(child.optimizer.state for child in fresh.children.values())
    fresh.set_mode('EVALUATE')
    before = snapshot_learning_state(fresh.learning_state_dict())
    for _ in range(3):
        result = fresh.act(fresh_obs, fresh_masks, None, True)
        fresh.process_env_step(transition(fresh_obs, fresh_masks), result.act_aux)
    assert_tree_equal(before, fresh.learning_state_dict())
    assert all(c.storage.step == 0 for c in fresh.children.values())
    with pytest.raises(RuntimeError, match='full rollout'):
        fresh.update()
    restored, _, _ = make_algorithm(share_actor_parameters=sharing)
    restored.load_learning_state_dict(state)
    a = algorithm.act(obs, masks, None, False)
    b = restored.act(obs, masks, None, False)
    assert_tree_equal(a.role_actions, b.role_actions)
    algorithm.discard_pending()
    restored.discard_pending()
    collect(algorithm, obs, masks)
    collect(restored, obs, masks)
    algorithm.update()
    restored.update()
    assert_tree_equal(algorithm.learning_state_dict(), restored.learning_state_dict())


def test_missing_world_reward_rejected_without_consuming_pending():
    algorithm, obs, masks = make_algorithm()
    result = algorithm.act(obs, masks, None, False)
    tr = transition(obs, masks)._replace(world_reward=None)
    with pytest.raises(ValueError, match='world_reward'):
        algorithm.process_env_step(tr, result.act_aux)
    assert all(c.storage.pending is not None and c.storage.step == 0 for c in algorithm.children.values())


def test_learning_state_rejects_owner_configuration():
    algorithm, _, _ = make_algorithm()
    incompatible, _, _ = make_algorithm(share_actor_parameters=False)
    with pytest.raises(ValueError, match='layout|signature|configuration'):
        incompatible.load_learning_state_dict(algorithm.learning_state_dict())


def hidden_tensors(model):
    state = model.get_hidden_state()
    return state if isinstance(state, tuple) else (state,)


@pytest.mark.parametrize('recurrent', ['gru', 'lstm'])
@pytest.mark.parametrize('sharing', [True, False])
@pytest.mark.parametrize('deterministic', [True, False])
def test_recurrent_zero_state_single_act_final_probe_and_boundaries(recurrent, sharing, deterministic):
    algorithm, obs, masks = make_algorithm(recurrent={'type': recurrent, 'hidden_size': 7},
                                           share_actor_parameters=sharing)
    actor_oracles, critic_oracles = {}, {}
    for owner, child in algorithm.children.items():
        assert child.actor.is_recurrent and child.critic.is_recurrent
        for model in (child.actor, child.critic):
            assert all(h is not None and torch.count_nonzero(h) == 0 for h in hidden_tensors(model))
        actor_oracles[owner] = copy.deepcopy(child.actor)
        critic_oracles[owner] = copy.deepcopy(child.critic)
    for step in range(algorithm.collection_steps):
        with torch.no_grad():
            for owner in algorithm.children:
                td = algorithm._observations(obs, owner)
                actor_oracles[owner](td)
                critic_oracles[owner](td)
        result = algorithm.act(obs, masks, None, deterministic)
        final = copy.deepcopy(obs)
        final.critic_inputs['team']['target'].add_(step + 4)
        tr = transition(final, masks, step)
        if step == 1:
            tr.learning_done['scout'][0, 1] = True
        if step == 2:
            tr.world_terminated[1] = True
        if step == 3:
            tr.world_truncated[0] = True
            tr.learning_done['guard'][1, 0] = True
        expected_values = {}
        for owner, child in algorithm.children.items():
            assert_tree_equal(child.actor.get_hidden_state(), actor_oracles[owner].get_hidden_state())
            assert_tree_equal(child.critic.get_hidden_state(), critic_oracles[owner].get_hidden_state())
            with torch.no_grad():
                probe = copy.deepcopy(critic_oracles[owner])
                expected_values[owner] = probe(algorithm._observations(final, owner))
            if step == 0:
                for hidden in child.storage.pending.hidden_states:
                    parts = hidden if isinstance(hidden, tuple) else (hidden,)
                    assert all(torch.count_nonzero(part) == 0 for part in parts)
        algorithm.process_env_step(tr, result.act_aux)
        for owner, child in algorithm.children.items():
            role, slot = algorithm.owners[owner]
            boundary = (tr.world_terminated | tr.world_truncated)[:, None] | tr.learning_done[role]
            terminal = tr.world_terminated[:, None] | tr.learning_done[role]
            if slot is not None:
                boundary = boundary[:, slot:slot + 1]
                terminal = terminal[:, slot:slot + 1]
            actor_oracles[owner].reset(boundary.flatten())
            critic_oracles[owner].reset(boundary.flatten())
            assert_tree_equal(child.actor.get_hidden_state(), actor_oracles[owner].get_hidden_state())
            assert_tree_equal(child.critic.get_hidden_state(), critic_oracles[owner].get_hidden_state())
            torch.testing.assert_close(child.storage.final_values[step], expected_values[owner])
            torch.testing.assert_close(child.storage.terminated[step, :, 0], terminal.flatten())
            assert_tree_equal(child.storage.dones[step, :, 0].bool(), boundary.flatten())
            if step == 0:
                for saved in (child.storage.saved_hidden_state_a, child.storage.saved_hidden_state_c):
                    assert saved is not None
                    assert all(torch.count_nonzero(part[0]) == 0 for part in saved)
    # 独立逐步 oracle 验证 H 末 timeout bootstrap 与 terminal 优先，不读 reset observation。
    from muse_builtins.algorithms.ippo import compute_returns
    for owner, child in algorithm.children.items():
        st = child.storage
        compute_returns(st, child.gamma, child.lam)
        for agent in range(st.num_envs):
            for start in range(algorithm.collection_steps):
                expected = st.values[start, agent, 0].item()
                discount = 1.
                for end in range(start, algorithm.collection_steps):
                    value = st.final_values[end, agent, 0].item() if not st.terminated[end, agent, 0] else 0.
                    expected += discount * (st.rewards[end, agent, 0].item() + child.gamma * value
                                            - st.values[end, agent, 0].item())
                    if st.dones[end, agent, 0]:
                        break
                    discount *= child.gamma * child.lam
                assert st.returns[start, agent, 0].item() == pytest.approx(expected, rel=1e-5)
        with torch.no_grad(), algorithm._learner_rng():
            for batch in st.recurrent_mini_batch_generator(3, 1):
                child.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0],
                            stochastic_output=True)
                log_prob = child.actor.get_output_log_prob(batch.actions)
                ratio = (log_prob - batch.old_actions_log_prob.squeeze(-1)).exp()
                torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=1e-5, atol=1e-5)
    before = snapshot_learning_state(algorithm.learning_state_dict())
    live = {owner: (copy.deepcopy(c.actor.get_hidden_state()), copy.deepcopy(c.critic.get_hidden_state()))
            for owner, c in algorithm.children.items()}
    algorithm.update()
    for owner, child in algorithm.children.items():
        assert_tree_equal(live[owner], (child.actor.get_hidden_state(), child.critic.get_hidden_state()))
        for kind in ('actor', 'critic'):
            assert any(not torch.equal(v, getattr(child, kind).state_dict()[k])
                       for k, v in before['roles'][owner][kind].items() if k.startswith('rnn.'))


@pytest.mark.parametrize('recurrent', ['gru', 'lstm'])
def test_recurrent_checkpoint_new_episode_different_batch_and_eval_reset(recurrent):
    algorithm, obs, masks = make_algorithm(recurrent={'type': recurrent, 'hidden_size': 7})
    assert all(c.actor.is_recurrent for c in algorithm.children.values())
    collect(algorithm, obs, masks)
    algorithm.update()
    state = snapshot_learning_state(algorithm.learning_state_dict())
    import io
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True)
    fresh, obs, masks = make_algorithm(batch=3, recurrent={'type': recurrent, 'hidden_size': 7})
    result = fresh.act(obs, masks, None, False)
    fresh.process_env_step(transition(obs, masks), result.act_aux)
    fresh.load_learning_state_dict(state)
    assert_tree_equal(state, fresh.learning_state_dict())
    for child in fresh.children.values():
        assert child.storage.step == 0 and child.storage.pending is None
        for model in (child.actor, child.critic):
            assert all(h is not None and torch.count_nonzero(h) == 0 for h in hidden_tensors(model))
    fresh.set_mode('EVALUATE')
    before = snapshot_learning_state(fresh.learning_state_dict())
    first = fresh.act(obs, masks, None, True)
    tr = transition(obs, masks)
    tr.world_truncated[0] = True
    tr.learning_done['scout'][1, 1] = True
    fresh.process_env_step(tr, first.act_aux)
    for owner, child in fresh.children.items():
        role, slot = fresh.owners[owner]
        boundary = tr.world_truncated[:, None] | tr.learning_done[role]
        if slot is not None:
            boundary = boundary[:, slot:slot + 1]
        for hidden in hidden_tensors(child.actor):
            assert torch.count_nonzero(hidden[:, boundary.flatten()]) == 0
            assert torch.count_nonzero(hidden[:, ~boundary.flatten()]) > 0
    second = fresh.act(obs, masks, None, False)
    # EVALUATE 冻结 RNG，即使调用方传 False 也使用确定性动作。
    for role in first.role_actions:
        torch.testing.assert_close(first.role_actions[role].motion[0], second.role_actions[role].motion[0])
    assert_tree_equal(before, fresh.learning_state_dict())
    incompatible, _, _ = make_algorithm(recurrent={'type': 'lstm' if recurrent == 'gru' else 'gru', 'hidden_size': 7})
    with pytest.raises(ValueError, match='signature'):
        incompatible.load_learning_state_dict(state)


@pytest.mark.parametrize('sharing', [True, False])
@pytest.mark.parametrize('kind', ['present', 'alive', 'controllable'])
def test_masked_agent_reward_cannot_change_actor_gradient(sharing, kind):
    left, obs, masks = make_algorithm(share_actor_parameters=sharing)
    right, _, _ = make_algorithm(share_actor_parameters=sharing)
    getattr(masks['scout'], kind)[0, 1] = False
    collect(left, obs, masks)
    collect(right, obs, masks)
    for owner, child in right.children.items():
        role, slot = right.owners[owner]
        if role == 'scout' and slot in (None, 1):
            child.storage.rewards[:, 1 if slot is None else 0] = 10000.
    left.update()
    right.update()
    for owner in left.children:
        assert_tree_equal(left.children[owner].actor.state_dict(), right.children[owner].actor.state_dict())
        if kind == 'present':
            assert_tree_equal(left.children[owner].critic.state_dict(), right.children[owner].critic.state_dict())


@pytest.mark.parametrize('override', [{'recurrent': {'type': 'rnn'}}, {'recurrent': {'type': 'gru', 'hidden_size': 0}},
                                      {'recurrent': {'type': 'lstm', 'hidden_size': -1}}])
def test_recurrent_configuration_rejects_invalid_values(override):
    from muse_builtins.algorithms.mappo import MAPPOConfig
    from muse_sdk.config import ConfigError, export_config_schema, resolve_config
    with pytest.raises(ConfigError):
        resolve_config(export_config_schema(MAPPOConfig), override, {}, require_complete=True)


@pytest.mark.parametrize('recurrent', ['gru', 'lstm'])
@pytest.mark.parametrize('sharing', [True, False])
def test_explicit_world_reset_clears_subset_hidden_and_freezes_learning(recurrent, sharing):
    algorithm, obs, masks = make_algorithm(recurrent={'type': recurrent, 'hidden_size': 7},
                                           share_actor_parameters=sharing)
    collect(algorithm, obs, masks)
    algorithm.update()
    algorithm.set_mode('EVALUATE')
    before = snapshot_learning_state(algorithm.learning_state_dict())
    result = algorithm.act(obs, masks, None, True)
    algorithm.process_env_step(transition(obs, masks), result.act_aux)
    hidden = {owner: tuple(h.clone() for h in hidden_tensors(c.actor))
              for owner, c in algorithm.children.items()}
    assert hasattr(algorithm, 'reset_worlds'), 'Algorithm reset_worlds hook missing'
    algorithm.reset_worlds(torch.tensor([True, False]))
    for owner, child in algorithm.children.items():
        width = 2 if owner == 'scout' and sharing else 1
        for old, new in zip(hidden[owner], hidden_tensors(child.actor)):
            assert torch.count_nonzero(old[:, :width]) > 0
            assert torch.count_nonzero(new[:, :width]) == 0
            torch.testing.assert_close(new[:, width:], old[:, width:], rtol=0, atol=0)
        assert child.storage.step == 0 and child.storage.pending is None
    algorithm.act(obs, masks, None, True)
    assert_tree_equal(before, algorithm.learning_state_dict())
