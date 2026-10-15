"""室内参考例的真实归档/解析，供CPU与GPU验收共同消费。"""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from flight_helpers import EXAMPLE, archive


INDOOR = EXAMPLE.parent / 'indoor_radio_reference'


def indoor_radio_catalog(root, *, service=False, model='indoor_office_radio', worlds=2):
    from muse_server.catalog import Catalog
    root.mkdir(parents=True, exist_ok=True)
    metadata = root / 'builtin.json'
    subprocess.run([sys.executable, '-m', 'muse_builtins.metadata', '--output', str(metadata),
                    '--controller', str(EXAMPLE / 'controller.json')], check=True)
    catalog = Catalog(root / 'catalog', metadata)
    environment = catalog.register_archive('environment', archive(INDOOR / 'environment', root / 'environment.zip'))
    package = root / 'package'
    shutil.copytree(INDOOR / 'training_package', package)
    composition = yaml.safe_load((package / 'training_package.yaml').read_text())
    composition['model_bindings']['radio'] = 'builtin:' + model
    (package / 'training_package.yaml').write_text(yaml.safe_dump(composition, sort_keys=False))
    record = catalog.register_archive('training_package', archive(package, root / 'package.zip'))
    request = json.loads((INDOOR / ('request_service.json' if service else 'request_evaluate.json')).read_text())
    request.update(environment_id=environment.resource_id, environment_revision=environment.revision,
                   training_package_id=record.resource_id, training_package_revision=record.revision)
    request['topology']['world_count'] = worlds
    request['algorithm_config']['world_power_scales'] = [1 / (i + 1)**2 if service else 1. for i in range(worlds)]
    if model == 'free_space_radio':
        request['model_config']['radio']['minimum_distance_m'] = 1.
    return catalog, request


def indoor_radio_plan(root, *, steps=200, **kwargs):
    from muse_server.resolver import resolve_experiment
    catalog, request = indoor_radio_catalog(root, **kwargs)
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment)
    return catalog, catalog.create_run_plan(identity, mode='EVALUATE', device='cuda:0', max_decision_steps=steps)
