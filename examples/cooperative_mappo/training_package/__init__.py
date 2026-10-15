"""合作跟踪任务；注册路径不导入 torch 或物理运行时。"""

from pydantic import Field

from muse_sdk.config import DataModel
from muse_sdk.registration import TermRegistration, TrainingPackageDefinition
from muse_sdk.runtime_types import TaskUpdateResult


class EmptyConfig(DataModel):
    pass


class TeamProgressConfig(DataModel):
    learning_horizon: int = Field(default=3, gt=0)


def tensor(shape, *, dtype="float32", units="1", frame="none"):
    return dict(shape=shape, dtype=dtype, units=units, frame=frame, rate="decision")


class TeamProgress:
    """学习边界演示，不改变物理存活、可控性或 world episode。"""

    @classmethod
    def describe(cls):
        return dict(
            task_id="team_progress",
            config_schema=TeamProgressConfig,
            input_fields=[
                dict(
                    path=["physics_outputs", "episode_decision"],
                    phase="post_physics",
                    tensor=tensor(["B"], dtype="int64"),
                )
            ],
            published_fields={"phase": tensor(["B", 1])},
            role_status_bindings={},
            learning_done_roles=["scout"],
        )

    def __init__(self, config, env):
        import torch

        self.config = TeamProgressConfig.model_validate(config)
        self.phase = torch.zeros(
            (env.num_envs, 1), dtype=torch.float32, device=env.device
        )

    def reset(self, env_ids):
        self.phase[env_ids] = 0

    def update(self, env):
        self.phase.copy_(env.episode_length_buf[:, None])
        done = (env.episode_length_buf == self.config.learning_horizon)[:, None]
        return TaskUpdateResult(
            role_status={},
            learning_done={
                "scout": done.expand_as(env.role_masks["scout"].present).clone()
            },
            diagnostics={},
        )


def team_distance(env):
    """每个 slot 权重相同，使用动作时目标，不使用下一步命令。"""
    import torch

    distances = [
        torch.linalg.vector_norm(
            env.role_positions_world[role] - env.act_commands.target_world[role], dim=-1
        )
        for role in ("scout", "rover")
    ]
    return -torch.cat(distances, dim=1).mean(dim=1)


def team_phase(env):
    return env.task.phase


def scout_squared_error(env, *, role_id):
    """每个 slot 的三维平方距离，单位 m^2；使用动作时目标。"""
    return (
        (env.role_positions_world[role_id] - env.act_commands.target_world[role_id])
        .square()
        .sum(dim=-1)
    )


def team_squared_error(env):
    """全部 slot 等权平均的平方距离，不是能耗或无线性能指标。"""
    import torch

    return torch.cat(
        [scout_squared_error(env, role_id=role) for role in ("scout", "rover")], dim=1
    ).mean(dim=1)


def build_training_package():
    inputs = tuple(
        dict(
            path=path,
            phase=phase,
            tensor=tensor(["B", count, 3], units="m", frame="world_z_up"),
        )
        for role, count in [("scout", 1), ("rover", 2)]
        for path, phase in [
            (["role_positions_world", role], "post_physics"),
            (["act_commands", "target_world", role], "action"),
        ]
    )
    return TrainingPackageDefinition(
        terms={
            "team_distance": TermRegistration(
                team_distance,
                "reward",
                "world",
                EmptyConfig,
                inputs,
                tensor(["B"], units="m"),
            ),
            "team_phase": TermRegistration(
                team_phase,
                "observation",
                "world",
                EmptyConfig,
                (
                    dict(
                        path=["task", "phase"],
                        phase="post_task",
                        tensor=tensor(["B", 1]),
                    ),
                ),
                tensor(["B", 1]),
            ),
            "team_squared_error": TermRegistration(
                team_squared_error,
                "cost",
                "world",
                EmptyConfig,
                inputs,
                tensor(["B"], units="m^2"),
                cost_id="team_squared_error",
            ),
            "scout_squared_error": TermRegistration(
                scout_squared_error,
                "cost",
                "role",
                EmptyConfig,
                tuple(
                    dict(
                        path=path,
                        phase=phase,
                        tensor=tensor(
                            ["B", "N_role", 3], units="m", frame="world_z_up"
                        ),
                    )
                    for path, phase in [
                        (
                            ["role_positions_world", {"binding": "role_id"}],
                            "post_physics",
                        ),
                        (
                            ["act_commands", "target_world", {"binding": "role_id"}],
                            "action",
                        ),
                    ]
                ),
                tensor(["B", "N_role"], units="m^2"),
                cost_id="scout_squared_error",
            ),
        },
        tasks={"team_progress": TeamProgress},
        algorithms={},
        radio_models={},
        sensing_models={},
    )
