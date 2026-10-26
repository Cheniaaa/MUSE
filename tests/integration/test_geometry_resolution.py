"""真实资源摘要进入resolver、Catalog和RunPlan，而非旁路构造布局。"""
from copy import deepcopy

import pytest
import yaml
from muse_sdk.run_protocol import RunRequest
from flight_helpers import archive, add_fixed_anchors
from muse_server.resolver import resolve_experiment


def geometry_request(pipeline, tmp_path, *, rf=True, blockers=()):
    catalog, source = pipeline
    root = tmp_path / 'environment'
    root.mkdir()
    bundle = dict(scene='scene.xml')
    if rf:
        bundle['rf_geometry'] = {'rf_blocking': list(blockers)}
    (root / 'environment.yaml').write_text(yaml.safe_dump(bundle))
    (root / 'scene.xml').write_text('''<mujoco><asset><hfield name="height" nrow="2" ncol="2" size="1 1 1 .1"/></asset><worldbody>
      <geom name="wall" type="box" pos="5 0 0" size=".5 1 1"/>
      <geom name="sphere" size="1"/><geom name="floor" type="plane" size="0 0 .1"/>
      <body name="moving"><joint/><geom name="dynamic" size=".1" type="box"/></body>
      <body name="tracked" mocap="true"><geom name="mocap" size=".1 .1 .1" type="box"/></body>
      <geom name="terrain" type="hfield" hfield="height"/>
      <flexcomp name="cloth" type="grid" count="2 2 1" spacing=".2 .2 .2" dim="2"/>
    </worldbody></mujoco>'''.replace('size=".1" type="box"', 'size=".1 .1 .1" type="box"'))
    record = catalog.register_archive('environment', archive(root, tmp_path / 'environment.zip'))
    request = deepcopy(source)
    request.update(environment_id=record.resource_id, environment_revision=record.revision)
    add_fixed_anchors(request)
    request['scenario']['radio_geometry'] = dict(endpoints={
        'tx': dict(group_id='scout_group', slots=[0], device_id='ideal_point_radio'),
        'rx': dict(group_id='anchor_group', slots=[0, 1], device_id='ideal_point_radio')},
        links=[dict(link_id='uplink', tx='tx', rx='rx')])
    return catalog, request


def test_geometry_is_frozen_through_experiment_and_run_plan(pipeline, tmp_path):
    catalog, request = geometry_request(pipeline, tmp_path, blockers=['wall'])
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    exp = report.resolved_experiment
    assert len(exp.geometry.layout.desired) == 2
    assert exp.geometry.blockers == ('wall',)
    assert exp.geometry.endpoint_epsilon_m > 0
    identity = catalog.save_experiment(exp, source=request)
    assert catalog.create_run_plan(RunRequest(experiment_id=identity, mode='EVALUATE', max_decision_steps=8), device='cpu').experiment.geometry == exp.geometry
    assert exp.model_bindings == {} and 'radio' not in exp.activation
    request['model_config'] = {'test': {}}
    assert resolve_experiment(request, catalog.snapshot()).errors[0]['field'] == 'model_config'


@pytest.mark.parametrize('rf,blockers', [(False, []), (True, ['missing']), (True, ['sphere']),
    (True, ['floor']), (True, ['dynamic']), (True, ['mocap']), (True, ['terrain']), (True, ['cloth'])])
def test_missing_semantics_or_unsupported_rf_selection_is_rejected(pipeline, tmp_path, rf, blockers):
    catalog, request = geometry_request(pipeline, tmp_path, rf=rf, blockers=blockers)
    report = resolve_experiment(request, catalog.snapshot())
    assert not report.valid
    assert any('rf_geometry' in e['field'] for e in report.errors), report.errors


def test_explicit_empty_rf_is_valid_but_missing_summary_is_not(pipeline, tmp_path):
    catalog, request = geometry_request(pipeline, tmp_path)
    snapshot = catalog.snapshot()
    report = resolve_experiment(request, snapshot)
    assert report.valid, report.errors
    assert report.resolved_experiment.geometry.blockers == ()
    record = next(r for r in snapshot.resources if r.resource_id == request['environment_id'])
    del record.metadata['geometry_summary']
    report = resolve_experiment(request, snapshot)
    assert not report.valid and 'geometry_summary' in report.errors[0]['field']


def test_ordinary_flight_does_not_require_rf_semantics(pipeline):
    catalog, request = pipeline
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid and report.resolved_experiment.geometry is None
