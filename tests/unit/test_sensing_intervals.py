"""snapshot与increment混合的逐字段时间和NaN-safe部分和。"""
import pytest
import torch

from wireless_runtime_helpers import EmptyConfig, MixedSensing, ConstantRate, epoch, make_runtime, requests, runtime_spec
from muse_runtime.network.runtime import WirelessRuntime
from muse_runtime.rng import WorldRng
from muse_sdk.config import export_config_schema
from muse_sdk.descriptor import InputField, RadioDescriptor, SensingDescriptor
from muse_sdk.models import ModelResult, RadioEvaluationResult, RateMetrics, SensingResult
from muse_sdk.network import BoundModel, network_model_input_fields
from muse_sdk.runtime_types import RadioAction


def test_mixed_sensing_fields_keep_distinct_time_semantics():
    runtime = make_runtime(sensing=MixedSensing)
    runtime.begin_decision(torch.zeros(2, dtype=torch.int64))
    for left, right in [(0, 4), (4, 8), (8, 10)]:
        epoch(runtime, left, right)
    snapshot = runtime.sensing_outputs['measurement']
    increment = runtime.sensing_outputs['fim']
    torch.testing.assert_close(increment.data, 5*torch.eye(2, dtype=torch.float64).expand(2, -1, -1))
    assert not increment.valid.any()
    assert snapshot.data.tolist() == [[8.], [8.]]
    assert snapshot.interval_start_tick.tolist() == [8, 8]
    assert snapshot.interval_end_tick.tolist() == [8, 8]
    assert increment.interval_start_tick.tolist() == [0, 0]
    assert increment.interval_end_tick.tolist() == [10, 10]
    saved = {name: [t.clone() for t in view] for name, view in runtime.sensing_outputs.items()}
    runtime.reset(torch.tensor([True, False]), sample_tick=torch.tensor([0, 10]))
    assert not increment.data[0].any() and not increment.valid[0]
    for name, view in runtime.sensing_outputs.items():
        assert view.interval_start_tick[0] == view.interval_end_tick[0] == 0
        for value, old in zip(view, saved[name]):
            assert torch.equal(value[1], old[1])


def test_increment_overflow_is_not_hidden_by_invalid_completeness():
    class Large(MixedSensing):
        def result(self, inputs, reset=False):
            result = super().result(inputs, reset)
            if not reset:
                result.output.fim[result.output.field_valid['fim']] *= 5e307
            return result
    runtime = make_runtime(spec=runtime_spec(service=False), radio=None, sensing=Large)
    runtime.begin_decision(torch.zeros(2, dtype=torch.int64))
    epoch(runtime, 0, 4)
    epoch(runtime, 4, 8)
    with pytest.raises(RuntimeError, match='nonfinite|overflow'):
        epoch(runtime, 8, 10)


def measurement_runtime(model_type):
    """两个元素mask算例复用真实冻结绑定，不修改已构造模型的descriptor。"""
    spec = runtime_spec(service=False)
    fields = [item for phase in ('reset', 'post_physics') for item in
              network_model_input_fields(spec, geometry=None, batch_size=2, phase=phase)]
    descriptor = SensingDescriptor(model_id='element_increments', config_schema=export_config_schema(EmptyConfig),
        state_scope='world', state_schema={}, diagnostics_schema={}, assumptions={'synthetic': True},
        input_fields=[], output_spec=dict(measurement=dict(shape=['B', 3], dtype='float32', units='m',
            frame='none', rate='radio_epoch', time_semantics='increment')))
    runtime = WirelessRuntime(spec, {'sensing': BoundModel(reference='element_increments', config={},
        descriptor=descriptor, input_fields=fields)}, {'sensing': model_type}, batch_size=2, device='cpu',
        rng=WorldRng(7, 2, 'cpu'), geometry_layout=None, geometry_view=None)
    runtime.reset(torch.ones(2, dtype=torch.bool), sample_tick=torch.zeros(2, dtype=torch.int64))
    return runtime


def test_measurement_increment_masks_each_element_and_preserves_partial_sum():
    class ElementIncrements(MixedSensing):
        def result(self, inputs, reset=False):
            if reset:
                data = torch.zeros((2, 3))
                valid = torch.zeros((2, 3), dtype=torch.bool)
            else:
                nan = float('nan')
                values = torch.tensor([[[2., nan, 1.], [4., 10., nan]],
                                       [[nan, 3., 2.], [1., nan, nan]],
                                       [[5., 7., 3.], [2., 6., nan]]])
                masks = torch.tensor([[[True, False, True], [True, True, False]],
                                      [[False, True, True], [True, False, False]],
                                      [[True, True, True], [True, True, False]]])
                indices = (inputs.sim_time/.001).round().long() // 4
                data, valid = values[indices, torch.arange(2)], masks[indices, torch.arange(2)]
            return ModelResult(SensingResult(data, None, {'measurement': valid},
                                             {'measurement': 'increment'}), (), {})

    runtime = measurement_runtime(ElementIncrements)
    view = runtime.sensing_outputs['measurement']
    pointers = [tensor.data_ptr() for tensor in view]
    runtime.begin_decision(torch.zeros(2, dtype=torch.int64))
    for left, right in ((0, 4), (4, 8), (8, 10)):
        epoch(runtime, left, right)
    assert view.data.tolist() == [[7., 10., 6.], [7., 16., 0.]]
    assert view.valid.tolist() == [[False, False, True], [True, False, False]]
    assert torch.isfinite(view.data).all()
    assert view.interval_start_tick.tolist() == [0, 0]
    assert view.interval_end_tick.tolist() == [10, 10]
    before = [tensor.clone() for tensor in view]
    runtime.reset(torch.tensor([True, False]), sample_tick=torch.tensor([0, 10], dtype=torch.int64))
    assert view.data[0].tolist() == [0., 0., 0.]
    assert not view.valid[0].any()
    for old, current in zip(before, view):
        torch.testing.assert_close(current[1], old[1], rtol=0, atol=0)
    assert [tensor.data_ptr() for tensor in view] == pointers


def test_all_invalid_measurement_increments_publish_zero_and_invalid():
    class AllInvalid(MixedSensing):
        def result(self, inputs, reset=False):
            return ModelResult(SensingResult(torch.full((2, 3), float('nan')), None,
                {'measurement': torch.zeros((2, 3), dtype=torch.bool)}, {'measurement': 'increment'}), (), {})

    runtime = measurement_runtime(AllInvalid)
    runtime.begin_decision(torch.zeros(2, dtype=torch.int64))
    for left, right in ((0, 4), (4, 8), (8, 10)):
        epoch(runtime, left, right)
    view = runtime.sensing_outputs['measurement']
    assert torch.isnan(runtime.models['sensing'].output.measurement).all()
    assert view.data.tolist() == [[0., 0., 0.], [0., 0., 0.]]
    assert not view.valid.any()
    assert view.interval_start_tick.tolist() == [0, 0]
    assert view.interval_end_tick.tolist() == [10, 10]


def test_radio_then_sensing_share_transmissions_and_current_dependency_output():
    spec = runtime_spec(resources=2, links=2, service=False)
    fields = [item for phase in ('reset', 'post_physics') for item in
              network_model_input_fields(spec, geometry=None, batch_size=2, phase=phase)]
    transmission_fields = [item for item in fields if item.path[1] == 'simultaneous_transmissions']
    radio_fields = [InputField(path=['network', 'radio', 'rate', name], phase=phase,
        tensor=dict(shape=['B', 'L', 'F'], dtype=dtype, units=units, frame='none', rate='radio_epoch'))
        for phase in ('reset', 'post_physics') for name, dtype, units in
        (('rate_bps', 'float32', 'bit/s'), ('valid', 'bool', '1'))]
    available_radio_fields = [item.model_copy(update={'tensor': item.tensor.model_copy(update={'shape': [2, 2, 2]})})
                              for item in radio_fields]
    common = dict(config_schema=export_config_schema(EmptyConfig), state_scope='world', state_schema={},
                  diagnostics_schema={}, assumptions={'synthetic': True})
    radio_descriptor = RadioDescriptor(model_id='transmission_rate', input_fields=transmission_fields,
        output_spec=dict(output_blocks=['rate']), **common)
    sensing_descriptor = SensingDescriptor(model_id='dependent_measurement',
        input_fields=[*transmission_fields, *radio_fields], output_spec=dict(measurement=dict(
            shape=['B', 'L', 'F'], dtype='float32', units='bit/s', frame='none', rate='radio_epoch',
            time_semantics='snapshot')), **common)
    events = []

    class TransmissionRate(ConstantRate):
        def result(self, inputs, reset=False):
            power = inputs.simultaneous_transmissions['power_w']
            active = inputs.simultaneous_transmissions['active']
            tx = [binding.tx for binding in self.layout['network'].communication_allocations]
            events.append(('radio', reset, power, active, None))
            rate = torch.zeros_like(power[:, tx]) if reset else power[:, tx] * 1000.
            valid = torch.zeros_like(active[:, tx]) if reset else active[:, tx].clone()
            return ModelResult(RadioEvaluationResult(None, None, RateMetrics(rate, valid)), (), {})

    class DependentMeasurement(MixedSensing):
        def result(self, inputs, reset=False):
            power = inputs.simultaneous_transmissions['power_w']
            active = inputs.simultaneous_transmissions['active']
            tx = [binding.tx for binding in self.layout['network'].communication_allocations]
            radio = inputs.radio_outputs
            events.append(('sensing', reset, power, active, radio))
            return ModelResult(SensingResult(radio.rate.rate_bps + 10.*power[:, tx], None,
                {'measurement': radio.rate.valid.clone()}, {'measurement': 'snapshot'}), (), {})

    bindings = {'radio': BoundModel(reference='transmission_rate', config={}, descriptor=radio_descriptor,
                                   input_fields=fields),
                'sensing': BoundModel(reference='dependent_measurement', config={}, descriptor=sensing_descriptor,
                                     input_fields=[*fields, *available_radio_fields])}
    runtime = WirelessRuntime(spec, bindings, {'radio': TransmissionRate, 'sensing': DependentMeasurement},
        batch_size=2, device='cpu', rng=WorldRng(7, 2, 'cpu'), geometry_layout=None, geometry_view=None)
    runtime.reset(torch.ones(2, dtype=torch.bool), sample_tick=torch.zeros(2, dtype=torch.int64))
    runtime.begin_decision(torch.zeros(2, dtype=torch.int64))
    cases = [(0, 4, [[1., 2.], [2., 1.]], [[7, -3], [-3, 7]],
              [[[1., 0.], [0., 2.], [0., 0.]], [[0., 2.], [1., 0.], [0., 0.]]],
              [[[1010., 0.], [0., 2020.]], [[0., 2020.], [1010., 0.]]]),
             (4, 8, [[0., 1.], [1., 0.]], [[7, 7], [-3, 7]],
              [[[0., 0.], [1., 0.], [0., 0.]], [[0., 1.], [0., 0.], [0., 0.]]],
              [[[0., 0.], [1010., 0.]], [[0., 1010.], [0., 0.]]])]
    for left, right, power, channel, expected_transmissions, expected_measurement in cases:
        action = {'pilot': RadioAction(torch.tensor(power), torch.tensor(channel, dtype=torch.int32))}
        epoch(runtime, left, right, action)
        transmissions = runtime.network.simultaneous_transmissions
        assert transmissions.power_w.tolist() == expected_transmissions
        assert runtime.sensing_outputs['measurement'].data.tolist() == expected_measurement
        expected_valid = torch.tensor(expected_measurement) > 0
        torch.testing.assert_close(runtime.sensing_outputs['measurement'].valid, expected_valid)
        radio_event, sensing_event = events[-2:]
        assert radio_event[2] is sensing_event[2] is transmissions.power_w
        assert radio_event[3] is sensing_event[3] is transmissions.active
        assert sensing_event[4] is runtime.models['radio'].output
    assert [(kind, reset) for kind, reset, *_ in events] == [
        ('radio', True), ('sensing', True), ('radio', False), ('sensing', False),
        ('radio', False), ('sensing', False)]


class IntervalSensing(MixedSensing):
    def result(self, inputs, reset=False):
        result = super().result(inputs, reset)
        result.output.time_semantics['measurement'] = 'interval_sample'
        result.output.field_valid['measurement'].fill_(not reset)
        return result


def interval_runtime(model_type=IntervalSensing):
    spec = runtime_spec(service=False)
    descriptor = SensingDescriptor(model_id='interval_measurement', input_fields=[],
        config_schema=export_config_schema(EmptyConfig), state_scope='world', state_schema={},
        assumptions={'synthetic': True}, output_spec=dict(
            measurement=dict(shape=['B', 1], dtype='float32', units='tick', frame='none',
                rate='radio_epoch', time_semantics='interval_sample'),
            fim=dict(shape=['B', 2, 2], dtype='float64', units='1', frame='none',
                rate='radio_epoch', time_semantics='increment')))
    binding = BoundModel(reference='interval_measurement', config={}, descriptor=descriptor, input_fields=[])
    runtime = WirelessRuntime(spec, {'sensing': binding}, {'sensing': model_type}, batch_size=2,
        device='cpu', rng=WorldRng(7, 2, 'cpu'), geometry_layout=None, geometry_view=None)
    runtime.reset(torch.ones(2, dtype=torch.bool), sample_tick=torch.zeros(2, dtype=torch.int64))
    return runtime


def test_interval_sample_commits_only_completed_window_and_replaces_samples():
    runtime = interval_runtime()
    view = runtime.sensing_outputs['measurement']
    assert not view.valid.any()
    assert view.data.data_ptr() != runtime.models['sensing'].output.measurement.data_ptr()
    runtime.begin_decision(torch.tensor([0, 0]))
    for left, right in ((0, 4), (4, 8), (8, 10)):
        before = [value.clone() for value in view]
        runtime.begin_epoch(torch.tensor([left, left]), interval_ticks=right-left, radio_requests=requests())
        for current, previous in zip(view, before):
            torch.testing.assert_close(current, previous, rtol=0, atol=0)
        runtime.end_epoch(torch.tensor([right, right]))
        assert view.data.tolist() == [[float(left)], [float(left)]]
        assert view.valid.all()
        assert view.interval_start_tick.tolist() == [left, left]
        assert view.interval_end_tick.tolist() == [right, right]
    assert view.data.tolist() == [[8.], [8.]]
    fim = runtime.sensing_outputs['fim']
    torch.testing.assert_close(fim.data, 5*torch.eye(2, dtype=torch.float64).expand(2, 2, 2))
    assert not fim.valid.any()
    assert fim.interval_start_tick.tolist() == [0, 0]
    assert fim.interval_end_tick.tolist() == [10, 10]
    before = [value.clone() for value in view]
    runtime.begin_decision(torch.tensor([10, 10]))
    for current, previous in zip(view, before):
        torch.testing.assert_close(current, previous, rtol=0, atol=0)


def test_interval_sample_missing_observation_replaces_previous_valid_sample():
    class Missing(IntervalSensing):
        def result(self, inputs, reset=False):
            result = super().result(inputs, reset)
            missing = (inputs.sim_time/.001).round().long() == 4
            result.output.measurement[missing] = float('nan')
            result.output.field_valid['measurement'][missing] = False
            return result

    runtime = interval_runtime(Missing)
    runtime.begin_decision(torch.tensor([0, 0]))
    epoch(runtime, 0, 4)
    view = runtime.sensing_outputs['measurement']
    assert view.valid.all()
    epoch(runtime, 4, 8)
    assert torch.isnan(runtime.models['sensing'].output.measurement).all()
    assert not view.valid.any()
    assert view.data.tolist() == [[0.], [0.]]
    assert view.interval_start_tick.tolist() == [4, 4]
    assert view.interval_end_tick.tolist() == [8, 8]
    epoch(runtime, 8, 10)
    assert view.valid.all()
    assert view.data.tolist() == [[8.], [8.]]


def test_interval_sample_subset_reset_preserves_other_world_and_fixed_caches():
    runtime = interval_runtime()
    runtime.begin_decision(torch.tensor([0, 0]))
    epoch(runtime, 8, 10)
    view = runtime.sensing_outputs['measurement']
    pointers = [value.data_ptr() for value in view]
    saved = [value.clone() for value in view]
    runtime.reset(torch.tensor([True, False]), sample_tick=torch.tensor([2, 10]))
    assert view.data[0].tolist() == [0.]
    assert not view.valid[0].any()
    assert view.interval_start_tick[0] == view.interval_end_tick[0] == 2
    for value, previous in zip(view, saved):
        torch.testing.assert_close(value[1], previous[1], rtol=0, atol=0)
    runtime.begin_decision(torch.tensor([2, 10]))
    runtime.begin_epoch(torch.tensor([2, 10]), interval_ticks=4, radio_requests=requests())
    assert view.interval_end_tick.tolist() == [2, 10]
    runtime.end_epoch(torch.tensor([6, 14]))
    assert view.data.tolist() == [[2.], [10.]]
    assert view.interval_start_tick.tolist() == [2, 10]
    assert view.interval_end_tick.tolist() == [6, 14]
    assert [value.data_ptr() for value in view] == pointers
