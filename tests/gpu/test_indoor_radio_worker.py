"""真实Supervisor→builtin Worker；收据只验证执行与终态，不充当科学oracle。"""
import json
from pathlib import Path
import time

import pytest

from indoor_radio_helpers import indoor_radio_plan


def run_worker(catalog, plan):
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
    receipts = [json.loads(line.split('INDOOR_RADIO_RESULT ', 1)[1]) for line in log.splitlines()
                if 'INDOOR_RADIO_RESULT ' in line]
    assert len(receipts) == plan.max_decision_steps
    return receipts


@pytest.mark.parametrize('model', ['indoor_office_radio', 'free_space_radio'])
def test_indoor_builtin_worker_evaluate(tmp_path, model):
    catalog, plan = indoor_radio_plan(tmp_path, model=model)
    receipts = run_worker(catalog, plan)
    assert receipts[0]['sample_tick'] == [14, 14]
    assert receipts[-1]['sample_tick'] == [3994, 3994]
    assert all(not any(x['reset']) and x['services'] == {} for x in receipts)
    assert all(v == 0 for row in receipts[0]['radio']['rate']['rate_bps'] for link in row for v in link)
    assert receipts[100]['radio']['rate']['rate_bps'][0][0][0] > 0
    assert receipts[199]['radio']['rate']['rate_bps'][0][0][0] == 0
    assert receipts[199]['radio']['rate']['rate_bps'][0][0][1] > 0


def test_builtin_service_worker_retains_terminal_snapshot(tmp_path):
    catalog, plan = indoor_radio_plan(tmp_path, service=True)
    receipts = run_worker(catalog, plan)
    index = next(i for i, value in enumerate(receipts) if value['reset'][0])
    terminal = receipts[index]
    assert terminal['reset'] == [True, False]
    assert terminal['services']['upload']['remaining'][0] == [0., 0.]
    assert any(v > 0 for v in terminal['services']['upload']['remaining'][1])
    assert terminal['services']['upload']['end'] == [20*(index+1)]*2
    assert receipts[index+1]['sample_tick'] == [14, 20*(index+1)+14]
    assert receipts[index+1]['services']['upload']['remaining'][0] == [2e6, 2e6]
