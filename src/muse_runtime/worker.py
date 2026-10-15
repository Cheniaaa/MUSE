"""单Run入口；在任何用户import之前隔离protocol与native日志。"""
import argparse
import os
import socket
from pathlib import Path
import threading
import traceback

from muse_sdk.experiment import RunPlan

from .protocol import WorkerProtocol


def execute(plan, protocol):
    from .package_loader import hash_package, load_package
    resources = {'environment': plan.experiment.environment,
                 **{'entity_model:' + name: ref for name, ref in plan.experiment.entity_models.items()}}
    for name, reference in resources.items():
        if hash_package(Path(plan.resource_paths[name])) != reference.sha256:
            raise ValueError(f'Frozen resource hash mismatch: {name}')
    with load_package(Path(plan.resource_paths['training_package'])) as package:
        if package.package_hash != plan.experiment.training_package.sha256:
            raise ValueError('Frozen training package hash mismatch')
        import torch
        import warp as wp
        from muse_builtins.algorithms import registrations as algorithm_registrations
        from muse_builtins.models import registrations as radio_registrations, sensing_registrations
        from .checkpoint import load_checkpoint, publish_checkpoint
        from .driver import run
        from .env import MuseEnv, RuntimeContext
        from .rng import WorldRng

        if plan.device != 'cuda:0':
            raise ValueError('M1 Worker requires the single CUDA device cuda:0; no CPU fallback')
        wp.init()
        stream = torch.cuda.Stream(device=plan.device)
        with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
            experiment = plan.experiment
            rng = WorldRng(experiment.root_seed, experiment.topology.world_count, plan.device)
            task_factory = package.definition.tasks[experiment.task_id] if experiment.task_id is not None else None
            radio_models = {f'builtin:{name}': implementation for name, implementation in radio_registrations().items()}
            radio_models.update(package.definition.radio_models)
            sensing_models = {f'builtin:{name}': implementation for name, implementation in sensing_registrations().items()}
            sensing_models.update(package.definition.sensing_models)
            models = {'radio': radio_models, 'sensing': sensing_models}
            model_types = {kind: models[kind][binding.reference]
                           for kind, binding in experiment.model_bindings.items()}
            context = RuntimeContext(plan.device, plan.resource_paths, package.definition.terms, task_factory, model_types)
            protocol.emit('compiling')
            env = MuseEnv(experiment, context, rng)
            preview_socket = None
            try:
                from .preview import WorkerPreview
                preview_socket = socket.socket(fileno=int(os.environ['MUSE_PREVIEW_FD']))
                preview_socket.setblocking(False)
                preview = WorkerPreview(plan.run_id, env, plan.artifact_directory, preview_socket, protocol.emit)
                algorithms = {f'builtin:{name}': implementation for name, implementation in algorithm_registrations().items()}
                algorithms.update(package.definition.algorithms)
                algorithm = algorithms[experiment.algorithm_id](experiment.algorithm_config)
                algorithm.initialize(experiment, context, rng)
                if plan.checkpoint is not None:
                    load_checkpoint(plan, algorithm)
                protocol.emit('running')

                def json_value(value):
                    if isinstance(value, torch.Tensor):
                        return value.detach().cpu().tolist()
                    if isinstance(value, dict):
                        return {key: json_value(item) for key, item in value.items()}
                    if isinstance(value, (tuple, list)):
                        return [json_value(item) for item in value]
                    return value

                def event_sink(kind, payload):
                    if kind == 'reset_world':
                        preview.after_reset(payload['world_id'])
                        preview.sample()
                        return
                    if kind == 'command_result':
                        protocol.emit(kind, payload)
                        return
                    if kind == 'step':
                        preview.after_step()
                    preview.sample()
                    sim_time = payload['steps'] * experiment.scenario.scene.physics_dt * experiment.scenario.scene.control_substeps
                    protocol.emit('metric', dict(source=kind, **json_value(payload)), sim_time=sim_time)

                def read_control():
                    if protocol.read_control() == 'cancel':
                        return 'cancel'
                    for control in protocol.take_preview_controls():
                        preview.apply(control)
                    preview.sample()
                    return protocol.read_control()

                def checkpoint_sink(*, steps, updates):
                    artifact = publish_checkpoint(plan, algorithm, steps=steps, updates=updates)
                    sim_time = steps * experiment.scenario.scene.physics_dt * experiment.scenario.scene.control_substeps
                    protocol.emit('checkpoint', artifact.model_dump(mode='json'), sim_time=sim_time)

                result = run(env, algorithm, read_control, event_sink, mode=plan.mode,
                             collection_steps=experiment.training_schedule.collection_steps
                             if experiment.training_schedule else None, max_decision_steps=plan.max_decision_steps,
                             checkpoint_sink=checkpoint_sink,
                             checkpoint_interval_updates=plan.checkpoint_interval_updates,
                             command_reader=protocol.take_command)
                stream.synchronize()
                return result
            finally:
                if preview_socket is not None:
                    preview_socket.close()
                env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = RunPlan.model_validate_json(args.plan.read_text())
    protocol = WorkerProtocol(plan.run_id)
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(plan.timeouts.heartbeat_interval):
            try:
                protocol.emit('heartbeat')
            except OSError:
                return

    thread = threading.Thread(target=heartbeat, name='muse-heartbeat', daemon=True)
    thread.start()
    code = 0
    try:
        protocol.emit('preparing')
        result = execute(plan, protocol)
    except BaseException as exc:
        traceback.print_exc()
        code = 1
        protocol.emit('failure', {'message': str(exc), 'type': type(exc).__name__})
        result = {'state': 'FAILED'}
    finally:
        stop.set()
        thread.join()
    protocol.emit('terminal', result)
    protocol.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
