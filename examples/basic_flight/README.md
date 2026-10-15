# Basic Flight: 规则飞行与 IPPO

此目录提供普通飞行规则 `EVALUATE` 和真实 IPPO `TRAIN` 的环境包、训练包和请求，不是独立运行器。
M1已贯通CPU资源注册、真实Warp/IPPO、正式API生命周期及浏览器Viser；完整证据见`docs/validation/m1-browser.md`。
算法接入与复杂度收口结果见`docs/validation/m1-complexity-cleanup.md`；平台更新后使用新的data-dir注册本版资源。
本例同时作为M4.3普通飞行参考，扩展后的motion-only真实学习回归已通过；三场景工程交付与研究效果状态见[交付记录](../../docs/validation/m4-reference-scenarios.md)。
`controller.json` 是T2已验证的唯一配置来源。学习策略 EVALUATE / INTERACTIVE 使用已登记 checkpoint；规则策略无需 checkpoint。INTERACTIVE 已支持冻结策略的暂停、继续、单步和指定 world reset，操作方式见下文。

## 运行入口

从`/project/mjc_ws`激活环境后启动Server；端口被占用时选择另一空闲端口：

```bash
source /project/mjc_ws/.myenv/bin/activate
python -m muse_server.app --host 127.0.0.1 --port 8001 --viewer-port 8081 --data-dir /tmp/muse-demo
```

另一个终端提交示例，客户端只调用正式HTTP API：

```bash
source /project/mjc_ws/.myenv/bin/activate
python examples/basic_flight/submit.py --server http://127.0.0.1:8001 --mode TRAIN --max-decision-steps 2400
```

输出Run ID与实际Viser URL。API文档为`http://127.0.0.1:8001/docs`；取消用`POST /runs/{run_id}/cancel`，显示开关用`PATCH /runs/{run_id}/display`。Run结束自动关闭显示；重新提交生成新会话。规则飞行用`--mode EVALUATE`；`--no-display`为headless。TRAIN总步数接受正整数，每完整H步才更新，默认H=24；episode长度500与Run总预算独立。

```bash
uv pip install --require-hashes -r requirements.lock --python .myenv/bin/python
MUSE_TEST_GPU=1 MUSE_VERIFY_LOCK=1 python -m pytest -q --junitxml=docs/validation/m1-tests.xml
```

锁从项目根安装；复制到其他目录时增加`--torch-backend cu128`。HTTP测试使用`httpx2==2.13.0`，与Starlette实际TestClient一致。

## IPPO 训练

规则示例仍使用 `training_package/` 与 `request.json`；训练选择独立的 `training_package_ippo/` 与 `request_train.json`。
两者共用 `environment/` 和唯一 controller，不替换已有规则fixtures。注册训练包后，用返回的资源ID/revision替换训练请求中的占位值。
按下文相同流程导出metadata、注册、resolve和保存，然后调用 `create_run_plan(..., mode='TRAIN', device='cuda:0')`。

训练示例的角色为scout1/guard2，B=2；每role独立actor/local critic，同role的B*N样本共享参数。
输入为本role的目标误差、world速度、wxyz姿态、body角速度和上一步motion；不读取其他role或centralized critic。
progress按selection独立持有历史，另有低权重distance奖励约束停留；并不声明已训练收敛。
四轴Beta先在[0,1]采样，再按resolved ActionSpec映射到物理motion；原始动作/logprob存入唯一RSL storage，没有二次clip。

算法 `describe()` 的 `training_schedule={'collection_steps': 'collection_steps'}` 指向 config 的顶层整数参数名；resolver 完成默认值/覆盖合并后，冻结为实际正整数 H。参数名由声明决定，不限于 `collection_steps`。规则算法声明 `training_schedule=None`。
`actor_observation='all_local'` 表示选择每个 role 所有已启用的本地观测，并要求各 role 非空；也可使用现有 role → InputField 列表显式选择。`algorithm_inputs` 分别保存 `observations/actors/critics` 的绑定，IPPO网络只消费其 `actors`，不会从 selections 或 critic 声明扩大 policy 输入。
学习算法必须通过 `reward_inputs` 声明实际消费的 `role`/`world` 奖励范围；当前 IPPO 为 `['role']`，启用 world reward 会在 resolver 报错。规则 EVALUATE 无学习目标，仍可计算奖励作为诊断。包内注册与内置注册遵循同一解析规则，不根据算法名切换行为。

| 预算 | 正式示例 | 短验收 |
|---|---|---|
| `algorithm_config.collection_steps` = resolved `training_schedule.collection_steps` | H=24 | H=4 |
| `world.termination.episode_limit.config.episode_max_steps` | 每episode 500 decision | 每episode 4 decision |
| Run/driver `max_decision_steps` | 建议24,000 decision，即1,000次update | 8 decision，即2次update |

`max_decision_steps` 是一次 Run 的决策步数上限：每次同时推进全部 world，不乘 world 数，也不计内部物理子步。`episode_max_steps` 是每个 world 的回合最大决策步数，碰撞等条件可以提前结束回合。`collection_steps` 是每次算法更新前的采样步数 H。
总Run预算不放algorithm config，也不能用episode上限代替。正式API接受正整数TRAIN预算；正常结束或取消时未满H的数据不更新，也不进入checkpoint，恢复从新episode开始。

创建 Run 的请求示例（`experiment_id` 使用已保存实验的实际ID）：

```json
{"experiment_id": "REGISTERED_EXPERIMENT_ID", "mode": "TRAIN", "max_decision_steps": 2400}
```

在 Experiment 请求中通过 `term_config.world.termination.episode_limit.config.episode_max_steps` 覆盖回合上限。两种字段及CLI均使用新名称，不接受旧 `max_steps` / `--max-steps` 别名。
Worker初始化顺序及单一执行入口如下；`control_reader`/`event_sink`由Worker注入，不另写循环：

```python
from types import SimpleNamespace
from muse_builtins.algorithms.ippo import IPPO
from muse_runtime.driver import run
from muse_runtime.env import MuseEnv, RuntimeContext
from muse_runtime.rng import WorldRng

experiment = plan.experiment
rng = WorldRng(experiment.root_seed, experiment.topology.world_count, plan.device)
env = MuseEnv(experiment, RuntimeContext(plan.device, plan.resource_paths, {}), rng)
try:
    algorithm = IPPO(experiment.algorithm_config)
    algorithm.initialize(experiment, SimpleNamespace(device=plan.device), rng)
    result = run(env, algorithm, control_reader, event_sink, mode='TRAIN',
                 collection_steps=experiment.training_schedule.collection_steps, max_decision_steps=24000)
finally:
    env.close()
```

`initialize`不要求context携带观测，网络shape取冻结algorithm_inputs；env构造已reset，driver直接用`env.policy_input`。
`IPPO.update()`无参数：每步已在storage保存pre-reset final value，H末continue同样使用真实next value。
`terminated | learning_done`禁止bootstrap；truncated只bootstrap final observation并断递推；关闭RSL time_out补偿。
actor loss有效性为present/alive/controllable，local critic为present，失活本身不等于learning_done。
RSL MLP/Beta/RolloutStorage/PPO loss与Adam原样复用；storage仅增加边界/masks和有效minibatch投影。
每个driver update内分别冻结另一个网络调用原生PPO，actor和critic各按配置执行epochs，只有一个optimizer owner。
CPU/CUDA learner RNG在算法调用范围保存/恢复，不消费WorldRng。失败step丢pending；cancel半block不update。
学习state hooks返回tensor/primitive字典，含modules、带参数名校验的lazy Adam、update count和learner RNG；Runtime发布边界负责独立稳定snapshot。
加载不恢复环境/episode/半block，拒绝CPU与CUDA learner RNG类型互换；正式checkpoint发布、恢复TRAIN和冻结EVALUATE已支持。

```bash
PYTHONPATH=src python -m pytest tests/unit/test_returns.py tests/gpu/test_flight_learning.py -q
```

## INTERACTIVE 基础调试

通过正式 API 创建 Run。下例的 Experiment 和 checkpoint ID 使用已登记的实际值；纯规则 Experiment 省略 `checkpoint_artifact_id`。现有 `submit.py` CLI 仍提供 TRAIN / EVALUATE，交互创建使用 API。

```json
{"experiment_id": "REGISTERED_EXPERIMENT_ID", "mode": "INTERACTIVE", "max_decision_steps": 24000, "checkpoint_artifact_id": "REGISTERED_CHECKPOINT_ID"}
```

提交到 `POST /runs` 后，使用 `PATCH /runs/{run_id}/display` 的 `{"enabled": true}` 打开 Viser；从 `GET /runs/{run_id}/display` 读取 `viewer_url`。INTERACTIVE 会显示 `Pause`、`Resume`、`Step`、`Reset world` 四个按钮，reset 使用 `World` 下拉框选中的 world。

Run 默认持续运行至 decision 预算耗尽；`Pause` 暂停整个 Run。暂停时 `Step` 同时推进全部 world 一次完整 decision（含其物理子步），执行后仍暂停；`Reset world` 只重置指定 world 的 episode 与对应策略 hidden，不增加 step、不生成学习 transition。心跳、显示切换和取消在暂停时继续工作；这些控件不会让冻结策略学习。

也可向 `POST /runs/{run_id}/commands` 顺序发送：

```json
{"command_seq": 1, "command": "pause"}
{"command_seq": 2, "command": "step"}
{"command_seq": 3, "command": "reset_world", "world_id": 0}
{"command_seq": 4, "command": "resume"}
```

每行是独立请求，须等上一条确认后再发下一条。返回的 `interaction` 以及 `GET /runs/{run_id}` 的同名字段包含 `command` 和 `result`；`result=null` 表示待执行，实际执行后报告 `status`、`paused`、`steps`、`updates` 和 `error`。暂停时 Run 生命周期仍为 RUNNING；是否暂停以结果中的 `paused` 为准。

`command_seq` 每Run从1连续递增，HTTP和Viser共享序列。重试最后一条完全相同的请求返回已有结果，不再次step/reset；旧序号、同序号不同内容、跳号或上一条仍pending时的新请求明确拒绝。多客户端操作时先读取当前序号。step/reset需要已暂停；不满足时返回rejected结果并消耗该序号。TRAIN/EVALUATE拒绝环境命令，取消仍使用既有 `/cancel`。

自定义算法须提供 `reset_worlds(mask)`：mask为bool `[B]`，仅清对应world的episode/recurrent状态。有状态算法自行实现；无状态算法可继承 `muse_sdk.interfaces.Algorithm` 默认实现。包验证会拒绝缺失hook的算法。

本轮未实现外部 `set_action`、指令失联行为或 Task inactive fallback；它们仍在M2.5剩余范围中。完整证据见[基础交互验收](../../docs/validation/m2-interactive.md)。

以下保留T3资源/DTO交接细节。

## 使用顺序

从 `/project/mjc_ws` 激活 `.myenv`；项目未安装时使用 `PYTHONPATH=src`，与M0 CLI一致。

```bash
source /project/mjc_ws/.myenv/bin/activate
PYTHONPATH=src python -m muse_builtins.metadata \
  --controller examples/basic_flight/controller.json --output /tmp/muse-builtin-metadata.json
python -m pytest tests/unit tests/integration/test_experiment_pipeline.py -q
```

metadata输出路径必须不存在。导出器只生成CPU数据，不import Torch/mjlab/MuJoCo/Warp，不构造sim。
`Catalog(directory, builtin_metadata)`固定该目录的platform metadata；修改平台后使用新的目录，不覆盖既有Experiment。
将 `environment/` 和 `training_package/` 的内容分别打包为ZIP，根文件不可套额外目录；其他归档格式明确拒绝。
调用 `register_archive` 获得Server生成的ID/revision，替换 `request.json` 中两个明确占位ID；不要把占位ID当READY资源。

```python
catalog = Catalog(data_directory, metadata_path)
environment = catalog.register_archive('environment', environment_archive)
package = catalog.register_archive('training_package', package_archive)
request.update(environment_id=environment.resource_id, environment_revision=environment.revision,
               training_package_id=package.resource_id, training_package_revision=package.revision)
report = resolve_experiment(request, catalog.snapshot())
assert report.valid, report.errors
experiment_id = catalog.save_experiment(report.resolved_experiment)
plan = catalog.create_run_plan(experiment_id, mode='EVALUATE', device='cuda:0')
```

CPU测试使用 `device='cpu'` 仅验证RunPlan线格式，不执行CPU物理。真正飞行仍只有Warp路径。
`Catalog`没有启动Run；执行/调度属于T4/T6。资源、SQLite记录、schema和RunPlan都位于传入的data directory。

## 公共函数

| 模块 | 签名/行为 |
|---|---|
| `muse_server.resolver` | `resolve_experiment(request: dict, catalog_snapshot: CatalogSnapshot) -> ValidationReport` |
| 同上 | `resolve_package_selections(descriptor, term_config, builtin_catalog=None) -> ResolvedPackageSelection`；保留M0诊断用途 |
| `muse_sdk.descriptor` | `build_descriptor(definition, composition, package_hash, builtin_catalog=None) -> PackageDescriptor` |
| 同上 | `parse_descriptor_json(content, builtin_catalog=None) -> PackageDescriptor`；`validate_composition(descriptor, builtin_catalog=None) -> None` |
| `muse_builtins.metadata` | `export_metadata(controller_path: Path) -> BuiltinMetadata`；CLI参数见上文 |
| `muse_server.catalog` | `Catalog(directory: Path, builtin_metadata: Path)` |
| Catalog | `register_archive(kind: str, archive: Path, *, resource_id: str | None = None) -> ResourceRecord`；无ID创建新资源，已有ID创建新revision |
| Catalog | `snapshot() -> CatalogSnapshot`；`save_experiment(experiment: ExperimentDefinition) -> str`；`get_experiment(identity: str) -> ExperimentDefinition` |
| Catalog | `create_run_plan(experiment_id: str, *, mode: str, device: str, checkpoint_artifact_id: str | None = None, max_decision_steps: int = 24000, timeouts=None) -> RunPlan` |
| `muse_server.storage` | `extract_archive(archive: Path, destination: Path) -> None`；`read_document(path: Path)`；`validate_bundled_assets(root: Path) -> None`；`write_json_new(path: Path, content: str) -> None` |

`save_experiment`消费resolver的已解析对象，API在T6必须先resolve，不能把任意用户JSON直接交给它。
注册包调用原one-shot `muse_runtime.validation --package ... --output-dir ... --builtin-metadata ...`。
Server不import Runtime/builtin/用户包；环境hash通过独立CPU子进程复用M0 `hash_package`。

## 全部共享DTO字段

除运行tensor容器外，DTO位于 `muse_sdk.experiment`，继承严格CPU `DataModel`，未知字段拒绝。
`ExperimentRequest.model_parameters` 只是避开Pydantic保留属性的Python内部名，JSON字段始终为 `model_config`。
数据的不可变性由独立解析副本、不可覆盖的资源revision/SQLite Experiment快照保证，不声称Python容器深度只读。

| DTO | 字段 |
|---|---|
| `ResourceRef` | `resource_id, revision, sha256` |
| `ResourceRecord(ResourceRef)` | `kind, path, metadata`；kind为environment/training_package/entity_model |
| `EntityModelRef(ResourceRef)` | `physics_asset`；冻结实际装配的模型内XML路径 |
| `EnvironmentBundle` | `scene, metadata`；archive内environment.yaml的严格结构 |
| `Pose` | `position[3], orientation_wxyz[4]` |
| `EntityGroup` | `group_id, model_id, model_revision, count, placement_id, controller_id, sensor_profile_id` |
| `SceneConfig` | `physics_dt, control_substeps, integrator, density, viscosity, precision` |
| `ScenarioDefinition` | `entity_groups, placements, scene, command_targets_world` |
| `RoleConfig` | `group_id, slots` |
| `Topology` | `world_count` |
| `Preview` | `world_ids`（当期一个world）, `rate_hz`（1..30） |
| `ExperimentRequest` | `environment_id, environment_revision, training_package_id, training_package_revision, scenario, task_id, task_config, algorithm_id, role_config, term_config, model_config, algorithm_config, topology, root_seed, preview` |
| `ActionSpec` | `interface, dtype, shape, fields, units, frames, minimum, maximum` |
| `EntityBindingPlan` | `role_id, group_id, model_id, entity_kind, role_slot_count, role_slot_to_group_slot, action_interfaces, controller_id, sensor_profile_id, network_profile_id=None` |
| `BoundSelection(ResolvedSelection)` | M0字段 `owner_path, term_kind, selection_id, term, enabled, config, weight, binding_role_id`；新增 `input_fields, output_spec, time_out, cost_id` |
| `ActionSlice` | `role_id, field='motion', start, stop, shape`；start/stop为每world packed轴，不乘B |
| `AlgorithmInputs` | `observations, actors, critics`；保持通用输入、role actor和critic group的声明归属 |
| `ExperimentDefinition` | `environment, environment_scene, environment_metadata, training_package, entity_models, platform_revision, scenario, bindings, controllers, sensors, task_id, task_config, algorithm_id, algorithm_config, training_schedule, algorithm_inputs, role_inputs, selections, termination_reasons, action_specs, action_layout, activation, model_bindings, costs, topology, root_seed, preview` |
| `ValidationReport` | `valid, resolved_experiment, errors, warnings`；CPU成功仍明确提示GPU未验证 |
| `RunPlan` | `run_id, experiment_id, mode, device, platform_revision, artifact_directory, resource_paths, experiment, max_decision_steps, timeouts, checkpoint_artifact_id=None` |
| `BuiltinMetadata` | `platform_revision, descriptor, entity_models, controllers, sensors` |
| `CatalogSnapshot` | `resources, builtins` |

既有 `InputField(path, phase, tensor)`、`TensorSpec(shape, dtype, units, frame, rate)` 原样复用。
role占位使用 `{'binding': 'role_id'}`；enabled Term的输入输出shape替换为具体B/N。
disabled selection保留参数和reason编号，`input_fields=[]`、`output_spec=None`，不要求运行依赖。
`controllers` 和 `sensors` 按group索引；前者保留原 `id/config/motion`，后者当期仅显式 `none`。
空能力固定 `model_bindings={}`、`costs={'roles': {}, 'world': {}}`，不创建radio/sensing/queue状态或dummy action。
`termination_reasons`由全部候选声明顺序编号，0表示未结束，停用不重编号。
Catalog落盘 `experiment.schema.json`、`run-plan.schema.json`；Run目录保存 `resolved_run_plan.json`。

## T4必须提供的缓存

所有缓存均为env公开字段，热路径无需Server参与；N来自该role的静态slot映射。

| 实际字段路径 | shape/dtype | 单位/frame | 声明阶段/消费者 |
|---|---|---|---|
| `env.role_positions_world[role]` | `[B,N,3]` float32 | m/world_z_up | post_physics；relative_target、distance |
| `env.commands.target_world[role]` | `[B,N,3]` float32 | m/world_z_up | next_command；relative_target |
| `env.act_commands.target_world[role]` | `[B,N,3]` float32 | m/world_z_up | action；distance |
| `env.physics_outputs.crashed[role]` | `[B,N]` bool | 1/none | post_physics；collision |
| `env.physics_outputs.episode_decision` | `[B]` int64 | 1/none | post_physics；timeout |

所有上述字段rate为 `decision`。`physics_outputs.episode_decision`是env唯一decision计数的只读视图，不能另存计时器。
`commands.target_world`由CommandManager拥有：reset按 `scenario.command_targets_world` 建立目标，episode内不重采样。
decision开始冻结 `act_commands`；末端观测读取next目标，奖励按原action-time目标评分。reset不求值奖励/termination。
`crashed`由真实contacts及翻覆判据生成，再按role映射投影；纯函数collision执行 `any`，输出world `[B]`。
engine nonfinite/overflow不是crashed，T4必须在reset前FAILED。flip具体检测仍由T4落实，T3不计算或伪造接触结果。
ObservationManager构造shape probe前建立位置与Command目标缓存；probe不得采样/推进history。
后续reward可读 `env.observations.actors[role][selection_id]` 和critic group字段，阶段为post_observation。

## Term与规则动作源装配

`muse_builtins.terms.registrations()`返回四个真实TermRegistration；Worker既可按包注册引用实现，也可显式使用builtin注册。

```text
relative_target(env, *, role_id) -> next_target - position            [B,N,3]
distance_reward(env, *, role_id, scale) -> -scale * norm(position-act_target) [B,N]
collision(env, *, controlled_role, reduction) -> crashed.any(-1) [B] bool
timeout(env, *, episode_max_steps) -> episode_decision >= episode_max_steps [B] bool
```

collision的role占位由冻结 `config.controlled_role` 绑定，world输出不注入额外role_id。
role-scoped callable的 `role_id` 从 `binding_role_id` 注入；业务config不包含它，不能再次合并默认值。
reward weight仅在manager聚合应用一次；distance无状态，不是ProgressReward隔离证据。
`Episode_Reward/*` 统一报告本次 reset 的 episode 累计奖励均值，不除以按 timeout 名称推测的时长。
T4需要另行实现/验证stateful ProgressReward，以及Command/Task-before-Term、subset reset和terminal快照。

`muse_sdk.runtime_types` 当前只定义当期消费者需要的：

```text
MotionAction(NamedTuple): motion: Any
RoleActions = dict[str, MotionAction]
ActResult(NamedTuple): role_actions: RoleActions, act_aux: Any
```

Worker用 `.motion` 取float32 `[B,N,4]`；SDK不import Torch。T4可在同文件按规范追加Observations/RoleMasks/TransitionView等。

```text
ConstantVelocity(config: dict)
describe() -> AlgorithmSpec字典（training_schedule=None）
initialize(experiment, context, rng) -> None；context.device决定tensor device
set_mode(mode) -> None；仅EVALUATE
act(observations, role_masks, rng, deterministic) -> ActResult
process_env_step(transition, act_aux) -> None
```

规则源无optimizer、storage、update或学习state hooks；driver的无learner分支不可调用update。
动作配置来自resolved `algorithm_config.motion`，没有执行前clip。环境验证每个typed leaf的dtype/shape/bounds。
原T2 motion bounds/units/frames逐字段冻结到ActionSpec，body速度每substep按当前姿态转换，yaw_rate为world水平heading变化率。

## 范围与限制

- T3是具体CF2、一个controller、显式无sensor、固定Command目标的普通飞行，不是通用profile engine。
- 请求额外Task、radio/sensing/cost返回明确字段错误；T5已开放有真实training_schedule的学习Algorithm。
- Scene配置显式保存 `dt=.002, substeps=20, RK4, density=1.225, viscosity=1.8e-5`；T4必须应用，不能丢给引擎默认。
- 示例scout1/guard2三个起点相距至少1m；代码和测试覆盖B1/B2，实体数量来自bindings，不写死三机。
- resource paths是Server分配的绝对目录；环境scene及MJCF资产引用留在bundle内，不依赖用户原资产路径。
- CPU只验证引用/参数/字段阶段；模型feature、contacts容量、物理/数值结果和真实rollout仍归T4/T6。
