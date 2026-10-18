"""参考任务真实物理、规则可行性、可见性、成本梯度与有限终止。"""
import json

import pytest
import torch
import yaml

from reference_scenario_helpers import reference_catalog, reference_env, ReferenceRule, rule_evaluation


def test_ordinary_motion_only_64_decisions_two_updates(pipeline, tmp_path, gpu_stream):
    from flight_helpers import EXAMPLE, archive
    from muse_builtins.algorithms.ippo import IPPO
    from muse_runtime.driver import run
    catalog, _ = pipeline
    package = catalog.register_archive('training_package', archive(EXAMPLE / 'training_package_ippo', tmp_path / 'ippo.zip'))
    request = json.loads((EXAMPLE / 'request_train.json').read_text())
    environment = next(item for item in catalog.snapshot().resources if item.kind == 'environment')
    request.update(environment_id=environment.resource_id, environment_revision=environment.revision,
                   training_package_id=package.resource_id, training_package_revision=package.revision)
    request['topology']['world_count'] = 2
    request['algorithm_config'].update(collection_steps=32, num_learning_epochs=1, num_mini_batches=1)
    with reference_env(catalog, request) as env:
        learner = IPPO(env.experiment.algorithm_config)
        learner.initialize(env.experiment, env.context, env.rng)
        result = run(env, learner, lambda: None, lambda *args: None, mode='TRAIN', collection_steps=32,
                     max_decision_steps=64)
        assert result == dict(state='COMPLETED', steps=64, updates=2)
        assert env.wireless is None and not learner.cost_state
        assert 'costs' not in learner.learning_state_dict()
        (tmp_path / 'ordinary-result.json').write_text(json.dumps(result))


def test_isac_hover_feasible_and_motion_changes_service_information(tmp_path, gpu_stream):
    catalog, request = reference_catalog(tmp_path / 'catalog', scenario='isac_cost', algorithm='mappo',
                                         costs=True, history_length=0, worlds=2)
    results, bounds = [], []
    for name, velocity in [('hover', 0.), ('move', .08)]:
        with reference_env(catalog, request) as env:
            episodes, terminal = rule_evaluation(env, tmp_path / name, velocity_y=velocity)
            assert all(item['complete'] and not item['failed'] and item['length'] == 500 for item in episodes)
            assert all(item['service_valid'] and item['discounted_mission_cost'] == 0 for item in episodes)
            assert all(item['rf_energy_j'] == pytest.approx(.01, rel=1e-5) for item in episodes)
            assert terminal.world_terminated.all() and not terminal.world_truncated.any()
            view = terminal.wireless.sensing['fim']
            assert view.valid.all()
            information = view.data.sum(dim=(1, 2))
            bound = torch.linalg.eigvalsh(information).reciprocal().sum(-1).sqrt()
            assert (bound <= .01).all()
            results.append(sum(item['served_bits'] for item in episodes))
            bounds.append(bound)
            (tmp_path / name / 'bound.json').write_text(json.dumps(bound.tolist()))
    assert results[1] > results[0]
    assert (bounds[1] < bounds[0]).all()


@pytest.mark.parametrize('power,same_channel', [(0., False), (.0001, False), (.001, True)])
def test_isac_rf_controls_change_violation_and_service(tmp_path, gpu_stream, power, same_channel):
    catalog, request = reference_catalog(tmp_path / 'catalog', scenario='isac_cost', algorithm='mappo',
                                         costs=True, history_length=0, worlds=2)
    with reference_env(catalog, request) as env:
        rule = ReferenceRule(env, power=power, same_channel=same_channel)
        for _ in range(4):
            action = rule.act(env.observations, env.role_masks, env.rng, True)
            out = env.step(action.role_actions)
            diagnostics = out.diagnostics['task']
            assert diagnostics['position_violation'].eq(1).all()
            assert out.transition.costs['world']['mission_position_violation'].eq(1).all()
            assert not out.transition.world_reset.any()
            torch.testing.assert_close(diagnostics['rf_energy_j'], torch.full((2,), 2*power*.04, device=env.device))
            if power == 0:
                assert not diagnostics['served_bits'].any()
                assert all(not fields['feedback'].any() for fields in out.policy_observations.actors.values())
            else:
                assert diagnostics['served_bits'].gt(0).all()


@pytest.mark.parametrize('scenario', ['isac_cost', 'hybrid_history_markov'])
@pytest.mark.parametrize('seed', [101, 211])
def test_fixed_rule_evaluation_eight_worlds_per_scenario(tmp_path, gpu_stream, scenario, seed):
    sensing = scenario == 'isac_cost'
    catalog, request = reference_catalog(tmp_path / 'catalog', scenario=scenario,
        algorithm='mappo' if sensing else 'ippo', costs=sensing, history_length=4 if not sensing else 0, worlds=4)
    request['root_seed'] = seed
    with reference_env(catalog, request) as env:
        episodes, _ = rule_evaluation(env, tmp_path / f'{scenario}-{seed}')
        assert all(item['complete'] and not item['failed'] and item['length'] == 500
                   and item['service_valid'] for item in episodes)
        if sensing:
            assert all(item['discounted_mission_cost'] == 0 for item in episodes)


@pytest.mark.parametrize('length', [1, 4])
def test_history_actual_feedback_alias_and_hidden_markov_state(tmp_path, gpu_stream, monkeypatch, length):
    catalog, request = reference_catalog(tmp_path, scenario='hybrid_history_markov', algorithm='ippo',
                                         costs=False, history_length=length, worlds=2)
    with reference_env(catalog, request) as env:
        emitter, = env.wireless.emitters
        emitter.state_index.copy_(torch.tensor([1, 2], device=env.device))
        emitter._publish(torch.arange(2, device=env.device))
        # 两条合法self-transition的确定性路径；采样只在测试边界控制。
        monkeypatch.setattr(emitter, '_sample', lambda worlds, probabilities, purpose: emitter.state_index[worlds].clone())
        rule = ReferenceRule(env)
        first = rule.act(env.observations, env.role_masks, env.rng, True)
        first.role_actions['pilot'].radio.channel[:, 0] = torch.tensor([-3, 19], device=env.device)
        env.step(first.role_actions)
        second = rule.act(env.observations, env.role_masks, env.rng, True)
        output = env.step(second.role_actions)
        feedback = output.policy_observations.actors['pilot']['feedback']
        torch.testing.assert_close(feedback[0, 0, -1], feedback[1, 0, -1])
        assert feedback[0, 0, -1, 0, 1] == 1
        assert not feedback[:, :, -1, 1:].any()
        if length == 4:
            assert not torch.equal(feedback[0, 0, -2], feedback[1, 0, -2])
        else:
            torch.testing.assert_close(feedback[0], feedback[1])
        assert output.transition.costs == {'roles': {}, 'world': {}}


def test_markov_subset_reset_preserves_other_world_future(tmp_path, gpu_stream):
    catalog, request = reference_catalog(tmp_path, scenario='hybrid_history_markov', algorithm='ippo',
                                         costs=False, history_length=4, worlds=2)
    reference = None
    for reset in (False, True):
        with reference_env(catalog, request) as env:
            rule = ReferenceRule(env)
            for _ in range(3):
                env.step(rule.act(env.observations, env.role_masks, env.rng, True).role_actions)
            if reset:
                before = env.observations.actors['pilot']['feedback'][1].clone()
                env.reset(torch.tensor([True, False], device=env.device))
                torch.testing.assert_close(env.observations.actors['pilot']['feedback'][1], before, rtol=0, atol=0)
                assert not env.observations.actors['pilot']['feedback'][0].any()
            future = []
            for _ in range(5):
                env.step(rule.act(env.observations, env.role_masks, env.rng, True).role_actions)
                future.append(env.observations.actors['pilot']['feedback'][1].clone())
            if reset:
                for value, expected in zip(future, reference, strict=True):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
            else:
                reference = future


@pytest.mark.parametrize('boundary', ['deadline', 'timeout', 'failed', 'both'])
def test_finite_task_and_admin_timeout_keep_terminal_history(tmp_path, gpu_stream, boundary):
    from flight_helpers import archive
    catalog, request = reference_catalog(tmp_path / 'catalog', scenario='isac_cost', algorithm='mappo',
                                         costs=True, history_length=0, worlds=2)
    request['task_config']['horizon_decisions'] = 3
    request['term_config'] = {'roles': {role: {'observation': {'feedback': {'history_length': 4}}}
                                      for role in ('alpha', 'beta')}}
    if boundary in ('failed', 'both'):
        request['task_config']['workspace_by_role'] = {
            'alpha': {'minimum_m': [-2., -2., .5], 'maximum_m': [2.5, 3., .99]},
            'beta': {'minimum_m': [5.5, -2., .5], 'maximum_m': [10., 3., 1.5]}}
    if boundary in ('timeout', 'both'):
        import shutil
        from reference_scenario_helpers import ISAC
        package = tmp_path / 'catalog/package'
        shutil.copytree(ISAC / 'training_package', package)
        package_config = yaml.safe_load((package / 'training_package.yaml').read_text())
        composition = package_config['compositions'][request['composition_id']]
        composition['world_termination_terms'].append(dict(selection_id='admin_timeout', term='builtin:timeout',
            config={'episode_max_steps': 2 if boundary == 'timeout' else 1}))
        (package / 'training_package.yaml').write_text(yaml.safe_dump(package_config, sort_keys=False))
        registered = catalog.register_archive('training_package', archive(package, tmp_path / 'timeout.zip'))
        request.update(training_package_id=registered.resource_id, training_package_revision=registered.revision)
    with reference_env(catalog, request) as env:
        from muse_builtins.algorithms.mappo import MAPPO
        learner = MAPPO(env.experiment.algorithm_config)
        learner.initialize(env.experiment, env.context, env.rng)
        steps = 3 if boundary == 'deadline' else 2 if boundary == 'timeout' else 1
        for _ in range(steps):
            action = learner.act(env.observations, env.role_masks, env.rng, True)
            output = env.step(action.role_actions)
            learner.process_env_step(output.transition, action.act_aux)
        assert output.transition.world_reset.all()
        assert output.transition.world_truncated.all().item() == (boundary == 'timeout')
        assert output.transition.world_terminated.all().item() == (boundary != 'timeout')
        final = output.transition.pre_reset_next_observations.actors['alpha']['feedback']
        assert final[:, :, -1].any()
        assert not output.policy_observations.actors['alpha']['feedback'].any()
        for owner, child in learner.children.items():
            with torch.no_grad():
                expected = child.critic(learner._observations(output.transition.pre_reset_next_observations, owner))
            torch.testing.assert_close(child.storage.final_values[steps-1], expected)
            assert child.storage.terminated[steps-1].all().item() == (boundary != 'timeout')
            assert child.storage.truncated[steps-1].all().item() == (boundary == 'timeout')
        if boundary in ('failed', 'both'):
            torch.testing.assert_close(output.transition.world_reward,
                torch.full((2,), -2.9701, device=env.device), atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(output.transition.costs['world']['mission_position_violation'],
                torch.full((2,), 2.9701, device=env.device), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('scenario', ['isac_cost', 'hybrid_history_markov'])
def test_actual_task_policy_gradients_and_cost_contribution(tmp_path, gpu_stream, monkeypatch, scenario):
    from muse_builtins.algorithms import registrations
    from muse_runtime.driver import run
    sensing = scenario == 'isac_cost'
    catalog, request = reference_catalog(tmp_path, scenario=scenario, algorithm='mappo' if sensing else 'ippo',
                                         costs=sensing, history_length=0 if sensing else 4, worlds=2)
    request['algorithm_config'].update(collection_steps=8, hidden_dims=[16], num_learning_epochs=1,
                                        num_mini_batches=1, entropy_coef=0.)
    results = []
    for multiplier in ([0., .7] if sensing else [0.]):
        with reference_env(catalog, request) as env, monkeypatch.context() as patch:
            learner = registrations()[env.experiment.algorithm_id.removeprefix('builtin:')](env.experiment.algorithm_config)
            learner.initialize(env.experiment, env.context, env.rng)
            for cost in learner.cost_state.values():
                cost.multiplier.fill_(multiplier)
            gradients, before, task_rewards = {}, {}, []
            for owner, child in learner.children.items():
                head = child.actor.mlp[-1]
                before[owner] = head.weight.detach().clone()
                original = child.optimizer.step
                def observed_step(*args, owner=owner, head=head, original=original, storage=child.storage, **kwargs):
                    if storage.sample_kind == 'actor':
                        gradients[owner] = head.weight.grad.detach().clone()
                    return original(*args, **kwargs)
                patch.setattr(child.optimizer, 'step', observed_step)
            def receive(kind, payload):
                if kind == 'step':
                    task_rewards.append(payload['diagnostics']['task']['task_reward'].clone())
            result = run(env, learner, lambda: None, receive, mode='TRAIN', collection_steps=8, max_decision_steps=8)
            assert result == dict(state='COMPLETED', steps=8, updates=1)
            for owner, gradient in gradients.items():
                assert gradient[:4].abs().sum() + gradient[5:9].abs().sum() > 0
                assert gradient[4].abs().sum() + gradient[9].abs().sum() > 0
                assert gradient[10:].abs().sum() > 0
                assert not torch.equal(learner.children[owner].actor.mlp[-1].weight, before[owner])
            assert set(gradients) == set(learner.children)
            results.append((gradients, torch.stack(task_rewards)))
    if sensing:
        torch.testing.assert_close(results[0][1], results[1][1])
        assert all(not torch.allclose(results[0][0][owner], results[1][0][owner]) for owner in results[0][0])
    (tmp_path / 'gradient-norms.json').write_text(json.dumps([
        {owner: float(value.norm()) for owner, value in gradients.items()} for gradients, _ in results], indent=2))


@pytest.mark.parametrize('kind', ['mappo', 'ippo'])
def test_isac_cost_off_uses_same_task_reward_and_no_cost_state(tmp_path, gpu_stream, monkeypatch, kind):
    from muse_builtins.algorithms import registrations
    from muse_runtime.driver import run
    catalog, request = reference_catalog(tmp_path, scenario='isac_cost', algorithm=kind,
                                         costs=False, history_length=0, worlds=2)
    request['algorithm_config'].update(collection_steps=4, hidden_dims=[16], num_learning_epochs=1, num_mini_batches=1)
    with reference_env(catalog, request) as env:
        learner = registrations()[kind](env.experiment.algorithm_config)
        learner.initialize(env.experiment, env.context, env.rng)
        step = env.step
        def observed(actions):
            out = step(actions)
            assert out.transition.costs == {'roles': {}, 'world': {}}
            assert 'mission_cost' in out.diagnostics['task']
            expected = out.diagnostics['task']['task_reward']
            if kind == 'mappo':
                torch.testing.assert_close(out.transition.world_reward, expected)
            else:
                assert out.transition.world_reward is None
                for reward in out.transition.rewards.values():
                    torch.testing.assert_close(reward, expected[:, None].expand_as(reward))
            return out
        monkeypatch.setattr(env, 'step', observed)
        result = run(env, learner, lambda: None, lambda *args: None, mode='TRAIN',
                     collection_steps=4, max_decision_steps=4)
        assert result['updates'] == 1 and 'costs' not in learner.learning_state_dict()
