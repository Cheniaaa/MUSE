"""显式提供真实GUI证据与对应Catalog，独立核对同一请求的科学冻结值。"""
import json
import os
from pathlib import Path

import pytest

from muse_server.catalog import Catalog
from muse_server.examples import import_example
from muse_server.resolver import resolve_experiment


@pytest.fixture(scope='module')
def gui_evidence():
    directory = os.environ.get('MUSE_WORKBENCH_GUI_EVIDENCE')
    catalog = os.environ.get('MUSE_WORKBENCH_CATALOG')
    if not directory or not catalog:
        pytest.skip('Explicit real GUI capture directory and its registered Catalog are required')
    root = Path(catalog)
    return Path(directory), Catalog(root, root / 'builtin-metadata.json')


@pytest.mark.parametrize('scenario', ['basic_flight', 'isac_cost', 'hybrid_history_markov'])
def test_gui_payload_matches_direct_resolve(gui_evidence, scenario):
    directory, catalog = gui_evidence
    capture = json.loads((directory / f'g3-live-{scenario}-edited.json').read_text())
    request = capture['document']['request']
    checked = resolve_experiment(request, catalog.snapshot())
    assert checked.valid, checked.errors
    assert checked.resolved_experiment.model_dump(mode='json') == capture['inspection']['resolved_experiment']
    assert checked.resolved_experiment.model_dump(mode='json') == capture['validation']['resolved_experiment']
    assert 'max_decision_steps' not in request


def test_three_reference_imports_freeze_sources(gui_evidence):
    _, catalog = gui_evidence
    for scenario in ('basic_flight', 'isac_cost', 'hybrid_history_markov'):
        imported = import_example(catalog, scenario)
        request = imported['request']
        checked = resolve_experiment(request, catalog.snapshot())
        assert checked.valid, checked.errors
        identity = catalog.save_experiment(checked.resolved_experiment, source=request, label='GUI contract ' + scenario)
        assert catalog.get_experiment_source(identity) == request
        assert catalog.get_experiment(identity) == checked.resolved_experiment
