"""真实Supervisor→Worker EVALUATE，读取包内终态收据。"""
import json
from pathlib import Path
import time

from wireless_package_helpers import wireless_plan


def test_real_wireless_worker_completes_and_resets(tmp_path):
    from muse_server.supervisor import Supervisor, TERMINAL_STATES
    catalog, plan = wireless_plan(tmp_path)
    with Supervisor(catalog) as supervisor:
        supervisor.submit(plan)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            record = supervisor.get_run(plan.run_id)
            if record['state'] in TERMINAL_STATES:
                break
            time.sleep(.05)
        log = Path(plan.artifact_directory, 'logs/worker.log').read_text()
        assert record['state'] == 'COMPLETED', log[-8000:]
        assert record['exit_code'] == 0
        receipts = [json.loads(line.split('WIRELESS_ORACLE ', 1)[1]) for line in log.splitlines()
                    if 'WIRELESS_ORACLE ' in line]
        assert len(receipts) == 6
        assert receipts[0]['sample_tick'] == [8, 8]
        assert receipts[1]['reset'] == [True, False]
        assert receipts[1]['services']['upload']['remaining'][0] == [0.]
        assert receipts[1]['services']['upload']['end'] == [20, 20]
        assert receipts[2]['sample_tick'] == [8, 28]
        assert not supervisor.gpu_busy
