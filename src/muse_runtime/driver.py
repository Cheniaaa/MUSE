"""唯一决策/学习调度；不拥有rollout或第二份环境状态。"""
import time

from muse_sdk.run_protocol import CommandResult
from muse_sdk.runtime_types import PolicyInput


def run(env, algorithm, control_reader, event_sink, *, mode, collection_steps, max_decision_steps,
        checkpoint_sink=None, checkpoint_interval_updates=None, command_reader=None):
    if mode not in ('TRAIN', 'EVALUATE', 'INTERACTIVE'):
        raise ValueError('Unsupported run mode')
    if mode == 'TRAIN' and (type(collection_steps) is not int or collection_steps <= 0):
        raise ValueError('TRAIN requires positive integer collection_steps')
    if type(max_decision_steps) is not int or max_decision_steps <= 0:
        raise ValueError('max_decision_steps must be a positive integer')
    algorithm.set_mode('EVALUATE' if mode == 'INTERACTIVE' else mode)
    policy = env.policy_input
    steps = updates = 0
    state = 'COMPLETED'
    paused = False
    last_command = last_result = None

    def acknowledge(command, error=None, *, remember=True):
        nonlocal last_command, last_result
        result = CommandResult(**command.model_dump(), status='rejected' if error else 'applied',
                               paused=paused, steps=steps, updates=updates, error=error)
        if remember:
            last_command, last_result = command, result
        event_sink('command_result', result.model_dump())

    while steps < max_decision_steps:
        if control_reader() == 'cancel':
            state = 'CANCELLED'
            break
        step_command = None
        command = command_reader() if command_reader is not None else None
        if command is not None:
            expected = 1 if last_command is None else last_command.command_seq + 1
            if last_command is not None and command == last_command:
                event_sink('command_result', last_result.model_dump())
            elif command.command_seq != expected:
                acknowledge(command, f'Expected command_seq {expected}', remember=False)
            elif mode != 'INTERACTIVE':
                acknowledge(command, 'Commands require INTERACTIVE')
            elif command.command in ('step', 'reset_world') and not paused:
                acknowledge(command, 'Command requires paused Run')
            elif command.command == 'reset_world':
                if command.world_id >= env.num_envs:
                    acknowledge(command, 'Invalid world_id')
                else:
                    import torch
                    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                    mask[command.world_id] = True
                    policy = env.reset(mask)
                    algorithm.reset_worlds(mask)
                    event_sink('reset_world', {'world_id': command.world_id})
                    acknowledge(command)
            elif command.command == 'step':
                step_command = command
            else:
                paused = command.command == 'pause'
                acknowledge(command)
        if paused and step_command is None:
            time.sleep(.01)
            continue
        result = algorithm.act(policy.observations, policy.role_masks, env.rng,
                               deterministic=mode != 'TRAIN')
        try:
            output = env.step(result.role_actions)
        except BaseException:
            discard = getattr(algorithm, 'discard_pending', None)
            if discard is not None:
                discard()
            raise
        algorithm.process_env_step(output.transition, result.act_aux)
        steps += 1
        # 保留原返回引用；环境负责借用buffer生命周期，算法负责storage复制。
        policy = PolicyInput(output.policy_observations, output.policy_role_masks)
        event_sink('step', {'steps': steps, 'updates': updates, 'diagnostics': output.diagnostics})
        if step_command is not None:
            acknowledge(step_command)
        if mode == 'TRAIN' and steps % collection_steps == 0:
            if control_reader() == 'cancel':
                state = 'CANCELLED'
                break
            metrics = algorithm.update()
            updates += 1
            event_sink('update', {'steps': steps, 'updates': updates, 'metrics': metrics})
            if (checkpoint_sink is not None and checkpoint_interval_updates is not None
                    and updates % checkpoint_interval_updates == 0):
                checkpoint_sink(steps=steps, updates=updates)
    if state != 'CANCELLED' and control_reader() == 'cancel':
        state = 'CANCELLED'
    if mode == 'TRAIN' and checkpoint_sink is not None:
        checkpoint_sink(steps=steps, updates=updates)
    return {'state': state, 'steps': steps, 'updates': updates}
