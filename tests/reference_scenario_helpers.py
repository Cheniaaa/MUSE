"""真实参考包归档与Catalog；CPU测试和主线程验收共用入口。"""
import json
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys

from flight_helpers import EXAMPLE, archive


REFERENCE_EXAMPLES = EXAMPLE.parent
ISAC = REFERENCE_EXAMPLES / 'isac_cost'
HISTORY = REFERENCE_EXAMPLES / 'hybrid_history_markov'


def reference_catalog(root: Path, *, scenario: str, algorithm: str, costs: bool,
                      history_length: int, worlds: int):
    from muse_server.catalog import Catalog

    if scenario not in ('isac_cost', 'hybrid_history_markov'):
        raise ValueError('Unknown reference scenario')
    if algorithm not in ('mappo', 'ippo'):
        raise ValueError('Unknown reference algorithm')
    if costs and (scenario != 'isac_cost' or algorithm != 'mappo'):
        raise ValueError('Only ISAC MAPPO supports the cost-on reference')
    if scenario == 'hybrid_history_markov' and algorithm != 'ippo':
        raise ValueError('The history reference uses IPPO')
    source = REFERENCE_EXAMPLES / scenario
    root.mkdir(parents=True, exist_ok=True)
    metadata = root / 'builtin.json'
    subprocess.run([sys.executable, '-m', 'muse_builtins.metadata', '--output', str(metadata),
                    '--controller', str(EXAMPLE / 'controller.json')], check=True)
    catalog = Catalog(root / 'catalog', metadata)
    environment = catalog.register_archive('environment', archive(source / 'environment', root / 'environment.zip'))
    record = catalog.register_archive('training_package', archive(source / 'training_package', root / 'package.zip'))
    request = json.loads((source / 'request_train.json').read_text())
    request.update(environment_id=environment.resource_id, environment_revision=environment.revision,
                   training_package_id=record.resource_id, training_package_revision=record.revision,
                   composition_id=('mappo_cost_on' if costs else 'mappo_cost_off')
                   if algorithm == 'mappo' else 'ippo')
    request['topology']['world_count'] = worlds
    if scenario == 'hybrid_history_markov':
        request['term_config'] = {
            'roles': {'pilot': {'observation': {'feedback': {'history_length': history_length}}}}}
    if not costs:
        request['cost_config'] = {'roles': {}, 'world': {}}
    return catalog, request


@contextmanager
def reference_env(catalog, request):
    """使用真实冻结RunPlan和已登记包；物理只由调用方串行推进。"""
    from muse_builtins.models import registrations, sensing_registrations
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.package_loader import load_package
    from muse_runtime.rng import WorldRng
    from muse_server.resolver import resolve_experiment
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    experiment = report.resolved_experiment
    identity = catalog.save_experiment(experiment, source=request)
    plan = catalog.create_run_plan(identity, mode='TRAIN', device='cuda:0',
                                   max_decision_steps=experiment.training_schedule.collection_steps)
    with load_package(Path(plan.resource_paths['training_package'])) as package:
        models = {'builtin:' + name: cls for name, cls in {**registrations(), **sensing_registrations()}.items()}
        context = RuntimeContext('cuda:0', plan.resource_paths, package.definition.terms,
            task_factory=package.definition.tasks[experiment.task_id] if experiment.task_id is not None else None,
            model_types={kind: models[binding.reference] for kind, binding in experiment.model_bindings.items()})
        env = MuseEnv(experiment, context, WorldRng(experiment.root_seed, experiment.topology.world_count, 'cuda:0'))
        try:
            yield env
        finally:
            env.close()


class ReferenceRule:
    """测试侧悬停/固定运动与RF对照；复用真实Env/Task/driver。"""
    def __init__(self, env, *, power=.00025, same_channel=False, velocity_y=0.):
        self.env, self.power, self.same_channel, self.velocity_y = env, power, same_channel, velocity_y
        self.last_transition = None

    def set_mode(self, mode):
        pass

    def act(self, observations, masks, rng, deterministic):
        import torch
        from muse_sdk.runtime_types import ActResult, MotionRadioAction, RadioAction
        actions = {}
        for index, (role, spec) in enumerate(self.env.experiment.action_specs.items()):
            motion = torch.zeros(spec.shape, device=self.env.device)
            motion[..., 1] = self.velocity_y
            channel = spec.radio.channel_ids[0 if self.same_channel else index % len(spec.radio.channel_ids)]
            actions[role] = MotionRadioAction(motion, RadioAction(
                torch.full(spec.shape[:2], self.power, device=self.env.device),
                torch.full(spec.shape[:2], channel, dtype=torch.int32, device=self.env.device)))
        return ActResult(actions, None)

    def process_env_step(self, transition, act_aux):
        self.last_transition = transition


def rule_evaluation(env, destination, *, decisions=500, **rule_options):
    import time
    from muse_runtime.driver import run
    from examples.reference_benchmark import first_episode_metrics
    rule = ReferenceRule(env, **rule_options)
    events = []
    def receive(kind, payload):
        if kind == 'step':
            task = {key: value.detach().cpu().tolist() for key, value in payload['diagnostics']['task'].items()}
            events.append(dict(kind='metric', payload=dict(source='step', steps=payload['steps'],
                updates=payload['updates'], diagnostics={'task': task})))
    started = time.monotonic()
    result = run(env, rule, lambda: None, receive, mode='EVALUATE', collection_steps=None,
                 max_decision_steps=decisions)
    summary = first_episode_metrics(events, worlds=env.num_envs,
        discount=env.experiment.algorithm_config['gamma'], sensing='sensing' in env.experiment.model_bindings)
    destination.mkdir(parents=True)
    (destination / 'metrics.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in events))
    (destination / 'result.json').write_text(json.dumps(dict(result=result, episodes=summary,
        root_seed=env.experiment.root_seed, worlds=env.num_envs, rule_options=rule_options,
        wall_seconds=time.monotonic()-started), indent=2))
    return summary, rule.last_transition
