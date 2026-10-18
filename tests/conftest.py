import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml
from flight_helpers import EXAMPLE, archive


@pytest.fixture(scope='module')
def pipeline(tmp_path_factory):
    from muse_server.catalog import Catalog
    root = tmp_path_factory.mktemp('flight')
    metadata = root / 'builtin.json'
    subprocess.run([sys.executable, '-m', 'muse_builtins.metadata', '--output', str(metadata),
                    '--controller', str(EXAMPLE / 'controller.json')], check=True)
    catalog = Catalog(root / 'catalog', metadata)
    environment = catalog.register_archive('environment', archive(EXAMPLE / 'environment', root / 'env.zip'))
    package = catalog.register_archive('training_package', archive(EXAMPLE / 'training_package', root / 'pkg.zip'))
    request = json.loads((EXAMPLE / 'request.json').read_text())
    request.update(environment_id=environment.resource_id, environment_revision=environment.revision,
                   training_package_id=package.resource_id, training_package_revision=package.revision)
    return catalog, request


@pytest.fixture
def checkpoint_plan(pipeline, tmp_path):
    from muse_sdk.experiment import RunPlan
    from muse_server.resolver import resolve_experiment
    catalog, request = pipeline
    exp = resolve_experiment(request, catalog.snapshot()).resolved_experiment
    return RunPlan(run_id='source', experiment_id='exp', mode='TRAIN', device='cpu',
                   platform_revision=exp.platform_revision, artifact_directory=str(tmp_path),
                   resource_paths={}, experiment=exp, max_decision_steps=8)


@pytest.fixture(scope='session', autouse=True)
def project_source_environment():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('PYTHONPATH', str(Path(__file__).resolve().parents[1] / 'src'))
        patch.setenv('PYTHONDONTWRITEBYTECODE', '1')
        yield


@pytest.fixture
def fixture_root():
    return Path(__file__).parent / 'fixtures' / 'cpu_package'


@pytest.fixture
def definition(fixture_root):
    name = '_muse_test_definition'
    spec = importlib.util.spec_from_file_location(name, fixture_root / '__init__.py', submodule_search_locations=[str(fixture_root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module.build_training_package()
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + '.'):
                del sys.modules[key]


@pytest.fixture
def composition(fixture_root):
    return yaml.safe_load((fixture_root / 'training_package.yaml').read_text())['compositions']['default']


@pytest.fixture
def descriptor(definition, composition):
    from muse_sdk.descriptor import build_descriptor
    return build_descriptor(definition, {'default': composition}, 'a' * 64, default_composition_id='default')
