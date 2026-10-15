from types import SimpleNamespace as NS

import torch
import pytest

from muse_sdk.runtime_types import Observations, RoleMask, TransitionView


def _transition(costs):
    masks = {'scout': RoleMask(*(torch.ones((2, 1), dtype=torch.bool) for _ in range(3)))}
    obs = Observations({'scout': {}}, {})
    return TransitionView(masks, obs, {'scout': torch.zeros(2, 1)}, None,
                          torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool),
                          torch.zeros(2, dtype=torch.long), {'scout': torch.zeros(2, 1, dtype=torch.bool)},
                          torch.zeros(2, dtype=torch.bool), costs, wireless=None)


def test_transition_keeps_named_costs_in_role_and_world_layouts():
    costs = {
        'roles': {'scout': {'energy': torch.ones(2, 1)}},
        'world': {'team_energy': torch.full((2,), 2.)},
    }
    transition = _transition(costs)
    assert transition.costs['roles']['scout']['energy'].shape == (2, 1)
    assert transition.costs['world']['team_energy'].shape == (2,)


def test_transition_requires_explicit_cost_layout():
    transition = _transition({'roles': {}, 'world': {}})
    assert transition.costs == {'roles': {}, 'world': {}}
    with pytest.raises(TypeError, match='costs'):
        TransitionView(**{name: value for name, value in transition._asdict().items() if name != 'costs'})


def test_cost_cfg_reuses_native_default_params():
    from mjlab.managers.manager_base import ManagerTermBaseCfg
    from muse_runtime.managers import CostTermCfg, WorldCostManager

    cfg = CostTermCfg(func=lambda env: torch.ones(env.num_envs), cost_id='energy')
    assert isinstance(cfg, ManagerTermBaseCfg)
    manager = WorldCostManager({'energy': cfg}, NS(num_envs=2, device='cpu'))
    torch.testing.assert_close(manager.compute()['energy'], torch.ones(2))


def test_cost_manager_evaluates_each_named_cost_once():
    from muse_runtime.managers import CostTermCfg, RoleCostManager, WorldCostManager

    calls = []
    env = NS(num_envs=2, device=torch.device('cpu'),
            role_positions_world={'scout': torch.zeros(2, 1, 3)})

    def role_cost(env, *, value):
        calls.append(('role', value))
        return torch.full((2, 1), value, device=env.device)

    def world_cost(env, *, value):
        calls.append(('world', value))
        return torch.full((2,), value, device=env.device)

    role = RoleCostManager({'energy': CostTermCfg(func=role_cost, params={'value': 1.5}, cost_id='energy')},
                           {'energy': 'scout'}, env)
    world = WorldCostManager({'team_energy': CostTermCfg(func=world_cost, params={'value': 2.5},
                                                         cost_id='team_energy')}, env)
    role_values = role.compute()
    world_values = world.compute()
    torch.testing.assert_close(role_values['scout']['energy'], torch.full((2, 1), 1.5))
    torch.testing.assert_close(world_values['team_energy'], torch.full((2,), 2.5))
    assert calls == [('role', 1.5), ('world', 2.5)]


@pytest.mark.parametrize('scope', ['role', 'world'])
@pytest.mark.parametrize('bad', ['shape', 'dtype', 'device', 'finite'])
def test_cost_manager_rejects_changed_term_output(scope, bad):
    from muse_runtime.managers import CostTermCfg, RoleCostManager, WorldCostManager

    env = NS(num_envs=2, device='cpu', role_positions_world={'scout': torch.zeros(2, 1, 3)})
    shape = (2, 1) if scope == 'role' else (2,)
    value = torch.ones(shape)
    cfg = {'energy': CostTermCfg(func=lambda env: value, cost_id='energy')}
    manager = (RoleCostManager(cfg, {'energy': 'scout'}, env) if scope == 'role'
               else WorldCostManager(cfg, env))
    result = manager.compute()
    torch.testing.assert_close((result['scout'] if scope == 'role' else result)['energy'], value)
    if bad == 'shape':
        value = torch.ones((*shape, 1))
    elif bad == 'dtype':
        value = value.double()
    elif bad == 'device':
        value = torch.empty(shape, device='meta')
    else:
        value[0] = float('nan')
    with pytest.raises(RuntimeError if bad == 'finite' else ValueError, match=bad):
        manager.compute()


def test_class_cost_is_instantiated_and_subset_reset_reaches_instance():
    from muse_runtime.managers import CostTermCfg, RoleCostManager

    class StatefulCost:
        def __init__(self, cfg, env):
            self.value = torch.zeros(2, 1, device=env.device)

        def __call__(self, env, *, increment):
            self.value.add_(increment)
            return self.value

        def reset(self, env_ids):
            self.value[env_ids] = 0

    env = NS(num_envs=2, device=torch.device('cpu'),
            role_positions_world={'scout': torch.zeros(2, 1, 3)})
    manager = RoleCostManager({'energy': CostTermCfg(func=StatefulCost, params={'increment': 1.},
                                                      cost_id='energy')}, {'energy': 'scout'}, env)
    torch.testing.assert_close(manager.compute()['scout']['energy'], torch.ones(2, 1))
    manager.reset(torch.tensor([True, False]))
    torch.testing.assert_close(manager.compute()['scout']['energy'], torch.tensor([[1.], [2.]]))
