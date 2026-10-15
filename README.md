# Muse

通过 Server 的正式 HTTP API 注册 Experiment、提交 Run 和读取 artifacts。示例脚本复用生产 Worker 与唯一 driver。

UAV＋UGV同world、不同动作/观测维度的使用方式见 [异构IPPO示例](examples/heterogeneous/README.md)，物理、学习和恢复证据见 [M2.1验收报告](docs/validation/m2-heterogeneous.md)。

局部actor＋集中critic的合作训练见 [MAPPO示例](examples/cooperative_mappo/README.md)：支持同role参数共享开关、GRU/LSTM、显式world team reward和Task学习边界。[M2.2验收报告](docs/validation/m2-mappo.md)记录真实GPU与HTTP恢复证据。

M4.3三参考场景已完成工程交付：[普通飞行](examples/basic_flight/README.md)、[ISAC＋cost](examples/isac_cost/README.md)、[history＋Markov](examples/hybrid_history_markov/README.md)，沿同一HTTP/Worker/checkpoint路径。混合动作和历史机制、真实训练／恢复／冻结评价已验证；ISAC策略仍未达到无飞行失败要求，完整学习对照矩阵未完成。2026-10-16经用户确认，学习优化与剩余效果验证转入后续研究／M6，状态见[交付记录](docs/validation/m4-reference-scenarios.md)。`examples/reference_benchmark.py`保留为可选研究矩阵入口。

M2跨阶段审查的四项问题已完成有限修复，当前配置及学习state快照契约见 [收口验收报告](docs/validation/m2-complexity-cleanup.md)。

## 启动与训练

在 `/project/mjc_ws` 执行，所有 Python 命令先激活项目环境。使用新的 data-dir 保存当前资源 revision：

```bash
source /project/mjc_ws/.myenv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m muse_server.app --port 8001 --data-dir /tmp/muse-m2-checkpoint
```

另一个终端激活同一环境后提交训练：

```bash
PYTHONDONTWRITEBYTECODE=1 python examples/basic_flight/submit.py --mode TRAIN --max-decision-steps 2400 --checkpoint-interval-updates 25 --no-display
```

输出包含 `experiment_id`、`run_id` 与 `run_url`。`--no-display` 提交后立即返回；默认尝试打开该 Run 的显示服务。`max-decision-steps` 是本次 Run 的正整数决策步数上限，每步同时推进所有 world。TRAIN 每收满 Experiment 中的 `collection_steps=H`（示例默认 24）才更新一次。预算不足 H 时正常结束但更新次数为 0；非整块尾段不更新，也不保存进 checkpoint，恢复时从新 episode 开始。

## 查询 checkpoint 并恢复

将下面占位符替换为实际 ID。先查询 Run 状态及 artifacts：

```bash
curl http://127.0.0.1:8001/runs/RUN_ID
curl http://127.0.0.1:8001/runs/RUN_ID/artifacts
```

从列表选择 `kind` 为 `checkpoint` 的条目，使用其完整 `id`（形如 `RUN_ID:checkpoint:UUID`），而非 kind 或本地文件路径。条目包含 `size` 和 `sha256`；下载使用 `GET /runs/RUN_ID/artifacts/ARTIFACT_ID`。只有 Server 已登记的文件可恢复。

```bash
PYTHONDONTWRITEBYTECODE=1 python examples/basic_flight/submit.py --experiment-id EXP_ID --mode TRAIN --checkpoint-artifact-id ARTIFACT_ID --max-decision-steps 2400 --no-display
PYTHONDONTWRITEBYTECODE=1 python examples/basic_flight/submit.py --experiment-id EXP_ID --mode EVALUATE --checkpoint-artifact-id ARTIFACT_ID --max-decision-steps 500 --no-display
```

三个 Run 可复用同一个 `EXP_ID`。`--experiment-id` 跳过上传与解析，保留原资源 revision；`--checkpoint-artifact-id` 必须与它同时提供。恢复保留 actor、critic、optimizer、learner RNG 与累计学习进度；新 Run 的 steps/updates 从 0 开始，环境从新 episode 开始，不恢复物理状态或半段 rollout。

`--checkpoint-interval-updates N` 仅用于 TRAIN，每 N 次成功 update 发布独立 checkpoint。省略时仍在正常结束或安全取消时保存；终点可与最后一次周期保存具有相同学习进度。失败或强制退出仅保留此前已发布并登记的文件。

学习 EVALUATE 必须提供 checkpoint，执行确定性动作并冻结学习状态，不产生新 checkpoint，预算无需按 collection_steps 整除。不给 `--experiment-id` 的 `--mode EVALUATE` 仍上传原规则算法示例，无需 checkpoint。

取消在decision/update安全边界优先停止；即使刚收满H步，只要尚未开始update，也会停止并保存此前已完成的学习状态。尚未更新的rollout不进入checkpoint。已经开始的update在返回后处理取消。

算法扩展的 `learning_state_dict()` 可返回借用参数/Adam tensor的普通state dict。Runtime在发布时生成一次独立CPU快照；直接调用该hook并需要跨后续更新保留基准时，应显式调用 `muse_runtime.checkpoint.snapshot_learning_state(...)`。

## Checkpoint 验证

CPU 边界测试位于 `tests/unit/test_checkpoint_contract.py`、`tests/unit/test_checkpoint_io.py` 与 `tests/integration/test_checkpoint.py`。真实 GPU 验收应串行执行：

```bash
MUSE_TEST_GPU=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/gpu/test_checkpoint_learning.py tests/gpu/test_flight_learning.py tests/integration/test_checkpoint.py::test_real_gpu_checkpoint_roundtrip
```

GPU 测试覆盖 B2→B1 的学习状态恢复、新 episode、跨 timeout 的冻结评估，以及内置 IPPO/包内 CustomIPPO 的 HTTP TRAIN→恢复 TRAIN→EVALUATE。集成 GPU 用例默认跳过，需显式设置 `MUSE_TEST_GPU=1`。当前验收结果见 [M2.4 验收报告](docs/validation/m2-checkpoint.md)。
