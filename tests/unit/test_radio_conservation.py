"""服务手算账本、物理发射与payload分离。"""
import pytest
import torch

from wireless_runtime_helpers import ConstantRate, epoch, make_runtime, requests, runtime_spec


def test_service_uses_actual_tail_and_single_backlog():
    runtime = make_runtime()
    service = runtime.network.services['data']
    runtime.begin_decision(torch.zeros(2, dtype=torch.int64))
    for left, right, served, remaining in [(0, 4, 4., 3.), (4, 8, 7., 0.), (8, 10, 7., 0.)]:
        epoch(runtime, left, right)
        torch.testing.assert_close(service.served_bits, torch.full((2, 1), served, dtype=torch.float64))
        torch.testing.assert_close(service.remaining_bits, torch.full((2, 1), remaining, dtype=torch.float64))
    assert service.interval_start_tick.tolist() == [0, 0]
    assert service.interval_end_tick.tolist() == [10, 10]
    assert service.valid.all()
    runtime.begin_decision(torch.full((2,), 10, dtype=torch.int64))
    epoch(runtime, 10, 14)
    assert not service.served_bits.any()
    assert runtime.network.simultaneous_transmissions.power_w[:, 0].eq(1.).all()
    epoch(runtime, 14, 18, requests(power=0.))
    assert not runtime.network.simultaneous_transmissions.active.any()


@pytest.mark.parametrize('bits', [0., 4., 7., 20.])
def test_service_scalar_ledger_and_subset_reset(bits):
    runtime = make_runtime(spec=runtime_spec(bits=bits, resources=2, links=2))
    runtime.begin_decision(torch.zeros(2, dtype=torch.int64))
    epoch(runtime, 0, 4, requests(links=2))
    epoch(runtime, 4, 8, requests(links=2, channel=-3))
    service = runtime.network.services['data']
    assert service.served_bits.eq(min(bits, 8.)).all()
    assert (service.served_bits+service.remaining_bits).eq(bits).all()
    before = [tensor.clone() for tensor in service]
    pointers = [tensor.data_ptr() for tensor in service]
    runtime.reset(torch.tensor([True, False]), sample_tick=torch.tensor([0, 8]))
    assert service.remaining_bits[0].eq(bits).all()
    assert not service.served_bits[0].any() and not service.valid[0].any()
    for previous, current, pointer in zip(before, service, pointers):
        assert torch.equal(previous[1], current[1])
        assert pointer == current.data_ptr()


def test_exclusive_checks_all_transmitters_and_reuse_retains_both():
    exclusive = make_runtime(spec=runtime_spec(links=2, sharing='exclusive'))
    with pytest.raises(RuntimeError, match='exclusive'):
        epoch(exclusive, 0, 4, requests(links=2))
    reuse = make_runtime(spec=runtime_spec(links=2))
    epoch(reuse, 0, 4, requests(links=2))
    assert reuse.network.simultaneous_transmissions.active.sum().item() == 4


def test_invalid_nan_rate_contributes_zero_without_poisoning_service():
    class InvalidRate(ConstantRate):
        def result(self, inputs, reset=False):
            result = super().result(inputs, reset)
            if not reset:
                result.output.rate.rate_bps[0] = float('nan')
                result.output.rate.valid[0] = False
            return result
    runtime = make_runtime(radio=InvalidRate)
    runtime.begin_decision(torch.zeros(2, dtype=torch.int64))
    epoch(runtime, 0, 4)
    service = runtime.network.services['data']
    assert service.served_bits.tolist() == [[0.], [4.]]
    assert service.valid.tolist() == [[False], [True]]


def test_no_service_allocates_no_backlog():
    runtime = make_runtime(spec=runtime_spec(service=False))
    epoch(runtime, 0, 4)
    assert runtime.network.services == {}


def test_allocation_rejects_over_budget_at_its_unique_boundary():
    runtime = make_runtime()
    with pytest.raises(RuntimeError, match='power'):
        epoch(runtime, 0, 4, requests(power=3.))


def test_valid_zero_rate_keeps_service_valid_without_consuming_bits():
    class ZeroRate(ConstantRate):
        def result(self, inputs, reset=False):
            result = super().result(inputs, reset)
            result.output.rate.rate_bps.zero_()
            return result

    runtime = make_runtime(radio=ZeroRate)
    runtime.begin_decision(torch.zeros(2, dtype=torch.int64))
    for left, right in ((0, 4), (4, 8), (8, 10)):
        epoch(runtime, left, right)
    service = runtime.network.services['data']
    assert runtime.network.simultaneous_transmissions.active.any()
    assert runtime.network.radio.rate.valid.all()
    assert not runtime.network.radio.rate.rate_bps.any()
    assert service.valid.tolist() == [[True], [True]]
    assert service.served_bits.tolist() == [[0.], [0.]]
    assert service.remaining_bits.tolist() == [[7.], [7.]]
    assert service.interval_start_tick.tolist() == [0, 0]
    assert service.interval_end_tick.tolist() == [10, 10]


def test_no_transmission_is_valid_zero_service_even_with_invalid_radio_rate():
    runtime = make_runtime()
    runtime.begin_decision(torch.zeros(2, dtype=torch.int64))
    for left, right in ((0, 4), (4, 8), (8, 10)):
        epoch(runtime, left, right, requests(power=0.))
    service = runtime.network.services['data']
    assert not runtime.network.simultaneous_transmissions.active.any()
    assert not runtime.network.simultaneous_transmissions.power_w.any()
    assert not runtime.network.radio.rate.valid.any()
    assert service.valid.tolist() == [[True], [True]]
    assert service.served_bits.tolist() == [[0.], [0.]]
    assert service.remaining_bits.tolist() == [[7.], [7.]]
    assert service.interval_end_tick.tolist() == [10, 10]


def test_channel_switch_uses_one_backlog_for_all_service_resources():
    class ResourceRates(ConstantRate):
        def result(self, inputs, reset=False):
            result = super().result(inputs, reset)
            if not reset:
                result.output.rate.rate_bps[..., 0] = 1000.
                result.output.rate.rate_bps[..., 1] = 2000.
                # 未分配槽也可有有效容量估计，但不能作为实际发送预算扣减。
                result.output.rate.valid.fill_(True)
            return result

    runtime = make_runtime(spec=runtime_spec(bits=5., resources=2), radio=ResourceRates)
    service = runtime.network.services['data']
    assert service.remaining_bits.shape == (2, 1)
    runtime.begin_decision(torch.zeros(2, dtype=torch.int64))
    for left, right, channel, total, remaining in ((0, 4, 7, 4., 1.),
                                                  (4, 8, -3, 5., 0.),
                                                  (8, 10, 7, 5., 0.)):
        epoch(runtime, left, right, requests(channel=channel))
        assert runtime.network.allocations.link_active.sum(-1).tolist() == [[1], [1]]
        assert service.served_bits.tolist() == [[total], [total]]
        assert service.remaining_bits.tolist() == [[remaining], [remaining]]
    assert service.valid.all()
    # 手算实际完成量为4、1、0 bit，不是每个F分别拥有5 bit backlog。
    assert (service.served_bits + service.remaining_bits).tolist() == [[5.], [5.]]


def test_large_finite_service_budget_is_capped_by_backlog():
    spec = runtime_spec().model_copy(update={'physics_dt': 4e304})
    runtime = make_runtime(spec=spec)
    runtime.begin_decision(torch.zeros(2, dtype=torch.int64))
    epoch(runtime, 0, 4)
    # rate=1000、interval=1.6e305，预算1.6e308仍有限。
    service = runtime.network.services['data']
    assert service.served_bits.tolist() == [[7.], [7.]]
    assert not service.remaining_bits.any()
    assert service.valid.all()


def test_service_budget_overflow_is_not_hidden_by_backlog_cap():
    spec = runtime_spec().model_copy(update={'physics_dt': 5e304})
    runtime = make_runtime(spec=spec)
    runtime.begin_decision(torch.zeros(2, dtype=torch.int64))
    runtime.begin_epoch(torch.zeros(2, dtype=torch.int64), interval_ticks=4, radio_requests=requests())
    assert torch.isfinite(runtime.network.radio.rate.rate_bps).all()
    assert torch.isfinite(runtime.network.radio.interval_s).all()
    before = [value.clone() for value in runtime.network.services['data']]
    # 有限rate与interval相乘为2e308，不能先min(...,7)掩盖预算溢出。
    with pytest.raises(RuntimeError, match='nonfinite service budget'):
        runtime.end_epoch(torch.full((2,), 4, dtype=torch.int64))
    for old, value in zip(before, runtime.network.services['data']):
        torch.testing.assert_close(value, old, rtol=0, atol=0)


def test_large_finite_service_accumulation_preserves_one_initial_budget():
    runtime = make_runtime(spec=runtime_spec(bits=1.6e308).model_copy(update={'physics_dt': 2e304}))
    runtime.begin_decision(torch.zeros(2, dtype=torch.int64))
    epoch(runtime, 0, 4)
    assert runtime.network.services['data'].served_bits.tolist() == [[8e307], [8e307]]
    epoch(runtime, 4, 8)
    epoch(runtime, 8, 10)
    service = runtime.network.services['data']
    assert service.served_bits.tolist() == [[1.6e308], [1.6e308]]
    assert torch.isfinite(service.served_bits).all()
    assert not service.remaining_bits.any()


def test_service_accumulation_rounding_overflow_is_rejected_before_commit():
    spec = runtime_spec(bits=torch.finfo(torch.float64).max).model_copy(update={'physics_dt': 2e304})
    runtime = make_runtime(spec=spec)
    runtime.begin_decision(torch.zeros(2, dtype=torch.int64))
    epoch(runtime, 0, 4)
    epoch(runtime, 4, 8)
    service = runtime.network.services['data']
    assert service.served_bits.tolist() == [[1.6e308], [1.6e308]]
    assert service.remaining_bits.gt(0).all() and torch.isfinite(service.remaining_bits).all()
    before = [value.clone() for value in service]
    runtime.begin_epoch(torch.full((2,), 8, dtype=torch.int64), interval_ticks=2, radio_requests=requests())
    # 最后预算4e307有限，但累加剩余量时float64舍入会使累计完成量溢出。
    with pytest.raises(RuntimeError, match='nonfinite service budget/accumulation'):
        runtime.end_epoch(torch.full((2,), 10, dtype=torch.int64))
    for old, value in zip(before, service):
        torch.testing.assert_close(value, old, rtol=0, atol=0)
