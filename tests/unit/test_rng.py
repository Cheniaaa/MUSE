import torch


def test_world_draws_and_instance_streams_are_independent():
    from muse_runtime.rng import WorldRng

    a, b = WorldRng(42, 2, 'cpu'), WorldRng(42, 2, 'cpu')
    a.uniform([0], 'command:scout', 'reset', (3,))
    a.uniform([1], 'command:guard', 'reset', (3,))
    actual = a.uniform([1], 'command:scout', 'reset', (3,))
    expected = b.uniform([1], 'command:scout', 'reset', (3,))
    assert torch.equal(actual, expected)
    assert not torch.equal(a.uniform([0], 'command:scout', 'reset', (3,)), expected)


def test_world_generators_do_not_change_global_torch_rng():
    from muse_runtime.rng import WorldRng

    before = torch.random.get_rng_state().clone()
    rng = WorldRng(12, 2, 'cpu')
    rng.uniform([0, 1], 'initialization', 'position', (4,))
    assert torch.equal(torch.random.get_rng_state(), before)


def test_reordered_world_ids_keep_each_world_sequence():
    from muse_runtime.rng import WorldRng

    a, b = WorldRng(99, 3, 'cpu'), WorldRng(99, 3, 'cpu')
    actual = a.uniform([2, 0], 'instance', 'purpose', (2,))
    expected = b.uniform([0, 2], 'instance', 'purpose', (2,))
    assert torch.equal(actual, expected.flip(0))
