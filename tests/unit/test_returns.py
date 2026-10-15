import copy
from types import SimpleNamespace as NS

import pytest
import torch

from muse_sdk.runtime_types import Observations, PolicyInput, RoleMask, StepOutput, TransitionView
from muse_runtime.checkpoint import snapshot_learning_state


def make_algorithm(device='cpu', costs=None, **overrides):
    from muse_sdk.experiment import ActionSpec, CostLayout
    from muse_builtins.algorithms.ippo import IPPO, IPPOConfig
    config = IPPOConfig(collection_steps=4, hidden_dims=[16], num_learning_epochs=2,
                        num_mini_batches=1).model_dump()
    config.update(overrides)
    specs = {r: ActionSpec(interface='body_velocity', dtype='float32', shape=[2, n, 4],
                          fields=['vx', 'vy', 'vz', 'yaw_rate'], units=['m/s']*3 + ['rad/s'],
                          frames=['body_flu']*4, minimum=[-.1]*4, maximum=[.1]*4)
             for r, n in [('scout', 1), ('guard', 2)]}
    inputs = {r: [NS(path=['observations', 'actors', r, 'target'], tensor=NS(shape=[2, s.shape[1], 3]))]
              for r, s in specs.items()}
    experiment = NS(root_seed=17, action_specs=specs, algorithm_inputs=NS(actors=inputs),
                    training_schedule=NS(collection_steps=config['collection_steps']),
                    costs=CostLayout.model_validate(costs or {'roles': {}, 'world': {}}))
    algorithm = IPPO(config)
    algorithm.initialize(experiment, NS(device=device), None)
    obs = Observations({r: {'target': torch.arange(2*s.shape[1]*3, device=device,
                                                   dtype=torch.float32).reshape(2, s.shape[1], 3)/10}
                        for r, s in specs.items()}, {})
    masks = {r: RoleMask(*(torch.ones(s.shape[:2], dtype=torch.bool, device=device)
                          for _ in range(3))) for r, s in specs.items()}
    return algorithm, obs, masks


def transition(obs, masks, step=0, costs=None):
    device = next(iter(obs.actors.values()))['target'].device
    terminated = torch.tensor([False, step == 1], device=device)
    truncated = torch.tensor([step == 3, False], device=device)
    return TransitionView(masks, obs,
        {r: torch.arange(m.alive.numel(), device=device).reshape(m.alive.shape).float() + .1 + step
         for r, m in masks.items()}, None, terminated, truncated,
        torch.zeros(2, dtype=torch.long, device=device),
         {r: torch.zeros_like(m.alive) for r, m in masks.items()}, terminated | truncated,
         costs if costs is not None else {'roles': {}, 'world': {}}, wireless=None)


def collect(algorithm, obs, masks):
    for step in range(algorithm.collection_steps):
        result = algorithm.act(obs, masks, None, False)
        algorithm.process_env_step(transition(obs, masks, step), result.act_aux)


@pytest.mark.parametrize('boundary', ['continue', 'terminated', 'truncated', 'learning_done', 'both'])
@pytest.mark.parametrize('index', [1, 3])
def test_returns_against_scalar_discounted_delta_oracle(boundary, index):
    from muse_builtins.algorithms.ippo import compute_returns
    rewards, values, next_values = [1., -2., 3., 4.], [.2, .4, .6, .8], [2., 7., 11., 13.]
    terminal = [False]*4
    truncated = [False]*4
    terminal[index] = boundary in ('terminated', 'learning_done', 'both')
    truncated[index] = boundary in ('truncated', 'both')
    expected = []
    for start in range(4):
        total, discount = values[start], 1.
        for end in range(start, 4):
            total += discount * (rewards[end] + (.9*next_values[end] if not terminal[end] else 0) - values[end])
            if terminal[end] or truncated[end]:
                break
            discount *= .9*.8
        expected.append(total)
    st = NS(rewards=torch.tensor(rewards).reshape(4, 1, 1), values=torch.tensor(values).reshape(4, 1, 1),
            final_values=torch.tensor(next_values).reshape(4, 1, 1),
            terminated=torch.tensor(terminal).reshape(4, 1, 1),
            truncated=torch.tensor(truncated).reshape(4, 1, 1), returns=torch.empty(4, 1, 1),
            advantages=torch.empty(4, 1, 1), actor_valid=torch.ones(4, 1, 1, dtype=torch.bool))
    compute_returns(st, .9, .8)
    torch.testing.assert_close(st.returns.flatten(), torch.tensor(expected))


def test_raw_beta_logprob_and_agent_axes():
    algorithm, obs, masks = make_algorithm()
    result = algorithm.act(obs, masks, None, False)
    for role, child in algorithm.children.items():
        pending = child.storage.pending
        raw = pending.actions
        assert raw.shape == (masks[role].alive.numel(), 4)
        assert pending.actions_log_prob.shape == (raw.shape[0],)
        torch.testing.assert_close(pending.actions_log_prob, child.actor.get_output_log_prob(raw))
        torch.testing.assert_close(result.role_actions[role].motion.flatten(0, 1), -.1 + .2*raw)
    algorithm.process_env_step(transition(obs, masks), result.act_aux)
    with pytest.raises(RuntimeError, match='pending|aux'):
        algorithm.process_env_step(transition(obs, masks), result.act_aux)


def test_independent_rng_and_state_snapshot_lazy_adam():
    global_before = torch.random.get_rng_state().clone()
    algorithm, obs, masks = make_algorithm()
    collect(algorithm, obs, masks)
    algorithm.update()
    assert torch.equal(global_before, torch.random.get_rng_state())
    state = snapshot_learning_state(algorithm.learning_state_dict())
    import io
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True, map_location='cpu')
    fresh, _, _ = make_algorithm()
    assert not fresh.children['scout'].optimizer.state
    fresh.load_learning_state_dict(state)
    assert fresh.update_count == 1
    for role in algorithm.children:
        assert fresh.children[role].optimizer.state
    a = algorithm.act(obs, masks, None, False)
    b = fresh.act(obs, masks, None, False)
    for role in a.role_actions:
        torch.testing.assert_close(a.role_actions[role].motion, b.role_actions[role].motion, rtol=0, atol=0)
    algorithm.discard_pending()
    fresh.discard_pending()
    collect(algorithm, obs, masks)
    collect(fresh, obs, masks)
    algorithm.update()
    fresh.update()
    for role in algorithm.children:
        for left, right in zip(algorithm.children[role].actor.parameters(), fresh.children[role].actor.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert state['updates'] == 1
    assert not torch.equal(state['roles']['scout']['actor']['mlp.0.weight'],
                           algorithm.children['scout'].actor.state_dict()['mlp.0.weight'].cpu())


@pytest.mark.parametrize('corrupt', ['name', 'shape', 'dtype', 'optimizer_name', 'optimizer_shape'])
def test_learning_state_rejects_incompatible_names_and_shapes(corrupt):
    algorithm, obs, masks = make_algorithm()
    collect(algorithm, obs, masks)
    algorithm.update()
    state = snapshot_learning_state(algorithm.learning_state_dict())
    role = state['roles']['scout']
    name = next(iter(role['actor']))
    if corrupt == 'name':
        role['actor']['unknown'] = role['actor'].pop(name)
    elif corrupt == 'shape':
        role['actor'][name] = torch.zeros(1)
    elif corrupt == 'dtype':
        role['actor'][name] = role['actor'][name].double()
    elif corrupt == 'optimizer_name':
        role['parameter_names'][0] = 'wrong'
    else:
        next(iter(role['optimizer']['state'].values()))['exp_avg'] = torch.zeros(1)
    fresh, _, _ = make_algorithm()
    with pytest.raises(ValueError, match='name|shape|dtype|parameter'):
        fresh.load_learning_state_dict(state)


def test_masks_keep_inactive_critic_and_freeze_actor_with_adam_momentum():
    algorithm, obs, masks = make_algorithm()
    collect(algorithm, obs, masks)
    algorithm.update()
    before_actor = {r: copy.deepcopy(c.actor.state_dict()) for r, c in algorithm.children.items()}
    before_critic = {r: copy.deepcopy(c.critic.state_dict()) for r, c in algorithm.children.items()}
    for m in masks.values():
        m.controllable.zero_()
    collect(algorithm, obs, masks)
    algorithm.update()
    for role, child in algorithm.children.items():
        assert all(torch.equal(v, child.actor.state_dict()[k]) for k, v in before_actor[role].items())
        assert any(not torch.equal(v, child.critic.state_dict()[k]) for k, v in before_critic[role].items())
    before = snapshot_learning_state(algorithm.learning_state_dict())
    for m in masks.values():
        m.present.zero_()
    collect(algorithm, obs, masks)
    algorithm.update()
    for role, child in algorithm.children.items():
        for name, value in before['roles'][role]['critic'].items():
            assert torch.equal(value, child.critic.state_dict()[name])


def test_learning_done_saved_and_pending_observations_are_owned():
    algorithm, obs, masks = make_algorithm()
    result = algorithm.act(obs, masks, None, False)
    old = obs.actors['guard']['target'].clone()
    obs.actors['guard']['target'].add_(100)
    tr = transition(obs, masks)
    tr.learning_done['guard'][0, 1] = True
    algorithm.process_env_step(tr, result.act_aux)
    st = algorithm.children['guard'].storage
    torch.testing.assert_close(st.observations[0]['target'], old.flatten(0, 1))
    assert st.terminated[0, 1]
    assert not st.terminated[0, 0]
    torch.testing.assert_close(st.final_values[0], algorithm.children['guard'].critic(st.observations[0] + 100))


def test_driver_failed_step_discards_pending_and_cancel_does_not_update():
    from muse_runtime.driver import run
    algorithm, obs, masks = make_algorithm()
    def fail(actions):
        raise RuntimeError('physics failed')
    env = NS(policy_input=PolicyInput(obs, masks), rng=None, step=fail)
    with pytest.raises(RuntimeError, match='physics failed'):
        run(env, algorithm, lambda: None, lambda *args: None, mode='TRAIN', collection_steps=4, max_decision_steps=8)
    assert all(c.storage.pending is None and c.storage.step == 0 for c in algorithm.children.values())
    count = 0
    def step(actions):
        nonlocal count
        count += 1
        return StepOutput(obs, masks, transition(obs, masks), {})
    env.step = step
    result = run(env, algorithm, lambda: 'cancel' if count == 2 else None, lambda *args: None,
                 mode='TRAIN', collection_steps=4, max_decision_steps=8)
    assert result == dict(state='CANCELLED', steps=2, updates=0)
    assert algorithm.update_count == 0


def test_ippo_metadata_export_is_cpu_only():
    import subprocess
    import sys
    script = '''
import sys
from pathlib import Path
from muse_builtins.metadata import export_metadata
result = export_metadata(Path('examples/basic_flight/controller.json'))
assert 'ippo' in result.descriptor.algorithms
assert not any(name in sys.modules for name in ('torch', 'rsl_rl', 'mjlab', 'warp'))
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('kind', ['present', 'alive', 'controllable'])
def test_masked_samples_cannot_change_actor_gradient(kind):
    left, obs, masks = make_algorithm()
    right, _, _ = make_algorithm()
    for m in masks.values():
        getattr(m, kind)[0, 0] = False
    collect(left, obs, masks)
    collect(right, obs, masks)
    for child in right.children.values():
        child.storage.rewards[:, 0].fill_(10000)
    left.update()
    right.update()
    for role in left.children:
        for a, b in zip(left.children[role].actor.parameters(), right.children[role].actor.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        if kind == 'present':
            for a, b in zip(left.children[role].critic.parameters(), right.children[role].critic.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_eval_freezes_learning_and_full_block_is_required():
    algorithm, obs, masks = make_algorithm()
    with pytest.raises(RuntimeError, match='full rollout'):
        algorithm.update()
    algorithm.set_mode('EVALUATE')
    before = snapshot_learning_state(algorithm.learning_state_dict())
    first = algorithm.act(obs, masks, None, True)
    algorithm.process_env_step(transition(obs, masks), first.act_aux)
    second = algorithm.act(obs, masks, None, True)
    for role in first.role_actions:
        assert torch.equal(first.role_actions[role].motion, second.role_actions[role].motion)
        assert algorithm.children[role].storage.step == 0
    assert torch.equal(before['cpu_rng'], algorithm.learning_state_dict()['cpu_rng'])


@pytest.mark.parametrize('dims', [[0], [-1, 16]])
def test_hidden_dimensions_are_positive_at_config_boundary(dims):
    from muse_builtins.algorithms.ippo import IPPOConfig
    from muse_sdk.config import ConfigError, export_config_schema, resolve_config
    with pytest.raises(ConfigError):
        resolve_config(export_config_schema(IPPOConfig), {'hidden_dims': dims}, {}, require_complete=True)


def test_advantage_normalization_excludes_masked_outlier():
    from muse_builtins.algorithms.ippo import compute_returns
    st = NS(rewards=torch.tensor([[[1.], [3.], [10000.]]]), values=torch.zeros(1, 3, 1),
            final_values=torch.zeros(1, 3, 1), terminated=torch.ones(1, 3, 1, dtype=torch.bool),
            truncated=torch.zeros(1, 3, 1, dtype=torch.bool), returns=torch.empty(1, 3, 1),
            advantages=torch.empty(1, 3, 1), actor_valid=torch.tensor([[[True], [True], [False]]]))
    compute_returns(st, .9, .8)
    torch.testing.assert_close(st.advantages.flatten(), torch.tensor([-1., 1., 0.]))


def test_failed_rng_scope_restores_global_state():
    algorithm, obs, masks = make_algorithm()
    before = torch.random.get_rng_state().clone()
    del obs.actors['scout']['target']
    with pytest.raises(KeyError):
        algorithm.act(obs, masks, None, False)
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize('corrupt', ['rng_shape', 'adam_step_shape', 'revision'])
def test_state_boundary_rejects_malformed_rng_and_adam_step(corrupt):
    algorithm, obs, masks = make_algorithm()
    collect(algorithm, obs, masks)
    algorithm.update()
    state = snapshot_learning_state(algorithm.learning_state_dict())
    if corrupt == 'rng_shape':
        state['cpu_rng'] = torch.zeros(1, dtype=torch.uint8)
    elif corrupt == 'adam_step_shape':
        next(iter(state['roles']['scout']['optimizer']['state'].values()))['step'] = torch.ones(2)
    else:
        state['revision'] += 1
    fresh, _, _ = make_algorithm()
    with pytest.raises(ValueError, match='shape|revision|RNG'):
        fresh.load_learning_state_dict(state)
