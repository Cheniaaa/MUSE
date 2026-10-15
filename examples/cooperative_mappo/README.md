# 合作 UAV + UGV MAPPO 示例

本例复用平台 `bitcraze_crazyflie_2` / `ugv_diff_drive` 模型及对应 controller，不复制物理资产。目录沿用 heterogeneous：`environment/` 是平地场景，`training_package/` 是可上传包，`request_train.json` 是实验请求。内置 metadata 使用现有 `examples/basic_flight/controller.json`，无需额外 controller 文件。

固定角色布局为 scout×1、rover×2，B=2、collection_steps=4、world timeout=8。scout 局部输入17维、motion4维；rover局部输入9维、motion2维。`critic_inputs.team` 显式声明两角色目标、速度以及 Task phase，展开共15维；集中字段不进入 actor。

`team_distance` 是唯一学习 reward：所有三个 slot 到 `act_commands.target_world` 的三维欧氏距离求算术均值并取负，权重1；没有 role reward。目标是持续跟踪，达到目标不终止。碰撞和 timeout 仍由独立 world Term 决定。

## 可选命名 cost

训练后可用本 Experiment 和其 checkpoint 创建 `mode=INTERACTIVE` 的冻结策略 Run，通过 API/Viser 暂停、单步和指定 world reset。GRU/LSTM 的 subset hidden reset、cost 学习状态冻结及 HTTP 恢复已验证；操作与命令序号说明见[基础交互用法](../basic_flight/README.md#interactive-基础调试)。

`request_train.json` 保持原有无 cost 行为。包中两个 cost 候选项默认关闭；上传同一个 `training_package/` 后，改用完整请求 `request_cost.json` 可同时开启以下指标，无需复制包或资产：

| cost_id | scope | 每步数值 | 单位 | 累计边界 |
|---|---|---|---|---|
| `team_squared_error` | world | scout×1、rover×2 的三维平方目标距离均值，每个 slot 等权 | `m^2` | world episode |
| `scout_squared_error` | role（scout） | 每个 scout slot 的三维平方目标距离 | `m^2` | world episode 或 scout learning_done |

两项均使用本步 `act_commands.target_world` 与 post_physics 位置，不使用下一步命令。它们是真实目标误差，不能解释为能耗、无线性能或安全概率。world 项是平方距离的均值，不是平均距离的平方。

每项独立设置 `discount=0.9`、`budget=0.01`、`lambda_initial=0.1`、`lambda_learning_rate=0.05`。预算约束从 reset/学习段起点开始的期望折扣累计 `sum(gamma^t * cost_t)`，单位仍为 `m^2`，不乘控制周期，也不是每步平均阈值。此处小预算用于短程验收观察 lambda 更新，不表示已验证可行或收敛；研究运行应按自己的误差容忍度和 episode 长度设置预算。

TRAIN 为每项保留独立 cost critic、Adam、lambda。每个 collection block 对新完成的有效 episode 估计取均值并更新一次 lambda；无新完成 episode 时保持 lambda。terminal 保留末步 cost 且不 bootstrap，timeout 使用 reset 前下一观测的 cost value。checkpoint 保留学习状态；恢复 Run 从新 episode 开始，清空未完成累计及 recurrent 采样状态。EVALUATE 冻结 actor/critic、optimizer、lambda、学习 RNG 和 cost 统计/hidden。

cost 验收使用非共享 actor + GRU，真实 GPU TRAIN 12步（3更新）、恢复 TRAIN 8步（累计5更新）、EVALUATE 9步（0更新，跨 timeout）。HTTP 使用相同步数的三个串行 Run，并下载校验两份 checkpoint；EVALUATE 不发布学习 checkpoint，逐 tensor 冻结由直接 GPU 验收检查。此验收证明链路与状态语义，不证明任务收敛。

```bash
source /project/mjc_ws/.myenv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 MUSE_VERIFY_LOCK=0
MUSE_TEST_GPU=0 python -m pytest -q -p no:cacheprovider tests/integration/test_cost_api.py
# 以下两条 GPU 命令依次执行
MUSE_TEST_GPU=1 python -m pytest -q -p no:cacheprovider tests/gpu/test_cost_training.py
MUSE_TEST_GPU=1 python -m pytest -q -s -p no:cacheprovider tests/integration/test_cost_api.py -k http
```

`share_actor_parameters=true` 时同 role 共用参数，异构 role 不共享；设为 false 时 scout/slot/0、rover/slot/0、rover/slot/1 分别拥有 actor、critic 和 optimizer。请求 `algorithm_config` 可覆盖该值。

## Task 接口

包入口 `build_training_package()` 注册：

- `tasks['team_progress'] = TeamProgress`，工厂调用为 `TeamProgress(config, env)`。
- `terms['team_distance']` 为 world reward，`terms['team_phase']` 为 world observation。
- `describe()` 的 input 是 `physics_outputs.episode_decision`，`post_physics`、int64[B]；published field 是 `phase`，float32[B,1]，units=1/frame=none/rate=decision。
- 构造时建立独立 phase tensor，使 observation shape probe 可读取；`update(env)` 直接复制 `env.episode_length_buf`，没有第二个 timer。
- `reset(env_ids)` 只清指定 world 的 phase，不写环境 clock。
- `learning_done_roles=['scout']`；`update` 返回轻量 `TaskUpdateResult(role_status={}, learning_done={'scout': bool[B,1]}, diagnostics={})`，不修改 alive/controllable。

`task_config.learning_horizon` 是严格正整数，默认3。在每个 world episode 的计数恰好等于该值时发出单步 learning done；随后恢复 false。它仅用于演示可配置学习边界，不代表到达目标、物理终止或失活。修改该字段只改变学习边界，world timeout 仍单独配置。若物理 episode 在 horizon 前结束，就不会出现该脉冲。

Worker 从已登记包获取 Task 工厂，按冻结的配置构造。Task 输出在提交前检查声明的角色、shape、dtype 与 device；共享字段在构造、更新及 reset 后检查。Task 的物理失活接管尚未开放，非空 `role_status_bindings` 在解析时拒绝。

## 注册与运行

可在提交Experiment前向`scenario.entity_groups`添加固定基站标记：

```json
{"group_id":"anchor_group","model_id":"radio_base_station","model_revision":1,"count":1,"placement_id":"anchor_start","controller_id":"none","sensor_profile_id":"none"}
```

同时在`scenario.placements`增加`"anchor_start": [{"position": [2.0, 0.0, 1.5], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]}]`。无需添加role、动作或command target；原UAV/UGV仍按原布局学习。

基站是固定端点的蓝色示意球，没有碰撞体，显示半径不代表设备实际尺寸。真实塔杆、机柜、建筑的碰撞几何应由场景资产显式描述；RF遮挡另按无线模型定义。当前加入此实体不会激活radio/sensing/queue，也不产生发射，不能将标记作为已实现的无线链路或近距离基站避障模型。

分别将 `environment/` 和 `training_package/` 的内容打包为 zip（压缩根必须直接含各自 yaml），POST `/catalog/environments` 与 `/catalog/training-packages`。把返回的 resource_id/revision 填入请求，再 POST `/experiments`。随后 POST `/runs`：

```json
{"experiment_id":"<experiment_id>","mode":"TRAIN","max_decision_steps":8}
```

首 Run 完成后从 `/runs/{run_id}/artifacts` 取 checkpoint 的 `id`，作为下一次 Run 的 `checkpoint_artifact_id`；恢复 TRAIN 8，再使用第二个 checkpoint 执行 EVALUATE 3。三个 Run 串行执行。

## 验证状态（2026-09-28）

本例已通过正式 HTTP TRAIN→恢复TRAIN→EVALUATE；共享参数＋GRU、独立参数＋LSTM各完成三个真实GPU Run。CPU契约/算法、真实物理/Task/recurrent与冻结评估证据见 [M2.2验收报告](../../docs/validation/m2-mappo.md)。本阶段证明训练链路和参数更新，不证明任务收敛。

`algorithm_config.recurrent` 为 `null` 时使用feedforward；启用时使用对象，例如 `{"type":"gru","hidden_size":64}` 或 `{"type":"lstm","hidden_size":16}`，对象内省略 `hidden_size` 时默认为64。feedforward没有无效的RNN参数；旧字符串形式及顶层 `recurrent_hidden_size` 明确拒绝。recurrent状态按学习/episode边界清理，checkpoint只恢复学习参数、optimizer、RNG和进度；新Run从新episode开始。不同world数量兼容已由CPU学习恢复测试验证。配置更新后重新登记Experiment并使用新的data-dir，不迁移旧checkpoint。

```bash
source /project/mjc_ws/.myenv/bin/activate
# 从 /project/mjc_ws 执行；CPU
MUSE_TEST_GPU=0 python -m pytest tests/integration/test_mappo_api.py -q
# GPU验收串行执行
MUSE_TEST_GPU=1 python -m pytest tests/integration/test_mappo_api.py -k http -q
```
