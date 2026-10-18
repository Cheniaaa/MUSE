"""真实感知参考归档/解析；供CPU与主线程GPU验证复用。"""
import json
import shutil
import subprocess
import sys

import yaml

from flight_helpers import EXAMPLE, archive


SENSING = EXAMPLE.parent / 'sensing_reference'


def sensing_catalog(root, *, model='free_space_path_length_efim_2d', worlds=2, jammer=True):
    from muse_server.catalog import Catalog
    root.mkdir(parents=True, exist_ok=True)
    metadata = root / 'builtin.json'
    subprocess.run([sys.executable, '-m', 'muse_builtins.metadata', '--output', str(metadata),
                    '--controller', str(EXAMPLE / 'controller.json')], check=True)
    catalog = Catalog(root / 'catalog', metadata)
    environment = catalog.register_archive('environment', archive(SENSING / 'environment', root / 'environment.zip'))
    package = root / 'package'
    shutil.copytree(SENSING / 'training_package', package)
    package_config = yaml.safe_load((package / 'training_package.yaml').read_text())
    composition = package_config['compositions']['default']
    reference = 'builtin:' + model
    composition['model_bindings']['sensing'] = {'default': reference, 'options': {reference: {}}}
    (package / 'training_package.yaml').write_text(yaml.safe_dump(package_config, sort_keys=False))
    record = catalog.register_archive('training_package', archive(package, root / 'package.zip'))
    request = json.loads((SENSING / 'request_evaluate.json').read_text())
    request.update(environment_id=environment.resource_id, environment_revision=environment.revision,
                   training_package_id=record.resource_id, training_package_revision=record.revision)
    request['topology']['world_count'] = worlds
    if not jammer:
        request['scenario']['network']['background_emitters'][0]['config']['active'] = [False]
    if model == 'free_space_path_length_measurement':
        request['term_config'] = {'world': {'cost': {'position_bound': {'enabled': False}}}}
        request['cost_config'] = {'roles': {}, 'world': {}}
    elif model == 'free_space_position_efim_2d':
        request['term_config'] = {'roles': {role: {'observation': {'measurement': {'enabled': False}}}
                                          for role in composition['agent_roles']}}
    return catalog, request


def sensing_plan(root, *, steps=200, **kwargs):
    from muse_server.resolver import resolve_experiment
    catalog, request = sensing_catalog(root, **kwargs)
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment)
    return catalog, catalog.create_run_plan(identity, mode='EVALUATE', device='cuda:0', max_decision_steps=steps)
