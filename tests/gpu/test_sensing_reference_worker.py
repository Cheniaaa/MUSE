"""真实Supervisor/Worker装配三能力；科学数值由独立oracle验收。"""
import json
from pathlib import Path
import time

import pytest

from sensing_package_helpers import sensing_plan


def run_sensing_worker(catalog, plan):
    from muse_server.supervisor import Supervisor, TERMINAL_STATES
    with Supervisor(catalog) as supervisor:
        supervisor.submit(plan)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            record = supervisor.get_run(plan.run_id)
            if record['state'] in TERMINAL_STATES:
                break
            time.sleep(.05)
        log = Path(plan.artifact_directory, 'logs/worker.log').read_text()
        assert record['state'] == 'COMPLETED', log[-8000:]
        assert record['exit_code'] == 0
        assert not supervisor.gpu_busy
    receipts = [json.loads(line.split('SENSING_RESULT ', 1)[1]) for line in log.splitlines()
                if 'SENSING_RESULT ' in line]
    assert len(receipts) == plan.max_decision_steps
    return receipts


@pytest.mark.parametrize('model', ['free_space_path_length_measurement',
    'free_space_position_efim_2d', 'free_space_path_length_efim_2d'])
def test_sensing_builtin_worker_capabilities(tmp_path, model):
    catalog, plan = sensing_plan(tmp_path, model=model, steps=2)
    receipt = run_sensing_worker(catalog, plan)[0]
    expected = ({'measurement'} if model == 'free_space_path_length_measurement' else
                {'fim'} if model == 'free_space_position_efim_2d' else {'measurement', 'fim'})
    assert set(receipt['sensing']) == expected


def test_sensing_builtin_worker_full_reference(tmp_path):
    catalog, plan = sensing_plan(tmp_path, steps=200)
    receipts = run_sensing_worker(catalog, plan)
    for index, start, end in [(0, 14, 20), (199, 3994, 4000)]:
        sample = receipts[index]['sensing']['measurement']
        fim = receipts[index]['sensing']['fim']
        assert sample['interval_start_tick'] == [start, start]
        assert sample['interval_end_tick'] == [end, end]
        assert fim['interval_start_tick'] == [end-20, end-20]
        assert fim['interval_end_tick'] == [end, end]
    assert all(not any(row['reset']) for row in receipts)
    for row in receipts[:50]:
        assert not any(value for world in row['sensing']['measurement']['valid'] for relation in world for value in relation)
        assert all(value for world in row['sensing']['fim']['valid'] for relation in world for value in relation)
        assert row['world_cost']['position_bound_violation'] == [1., 1.]
    assert receipts[100]['sensing']['measurement']['valid'][0] == [[True, False], [False, True]]
    assert receipts[199]['sensing']['measurement']['valid'][0] == [[False, True], [False, True]]
    assert receipts[100]['radio']['rate']['rate_bps'][0][0][0] > 0
    assert receipts[199]['radio']['rate']['rate_bps'][0][0][0] == 0
    assert receipts[199]['radio']['rate']['rate_bps'][0][0][1] > 0
    # 下一步收到的是上一步已完成测量，不把同一步新采集的样本倒灌到action前输入。
    for index in (0, 50, 100, 199):
        for relation, role in enumerate(('alpha', 'beta')):
            local = receipts[index]['policy_observations'][role]['measurement']
            if index == 0:
                assert local == [[[0., 0., 0.]], [[0., 0., 0.]]]
            else:
                import numpy as np
                previous = receipts[index-1]['sensing']['measurement']
                for world in range(2):
                    valid = previous['valid'][world][relation]
                    distance = sum(v for v, ok in zip(previous['data'][world][relation], valid) if ok)
                    np.testing.assert_allclose(local[world][0], [distance/100, float(any(valid)), .3], rtol=1e-7, atol=0)
