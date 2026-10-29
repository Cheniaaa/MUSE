# Muse

**面向多无人机与异构机器人协同研究的 GPU 仿真与强化学习平台。**

Muse 将实验配置、并行仿真、策略学习和结果复盘串联在同一条运行链路中。研究者通过 Python 研究包定义任务与算法，通过 HTTP API 或桌面工作台配置实验、提交运行、观察场景并分析结果。

本仓库提供 **SDK、Server、Runtime 和参考示例**；配套桌面客户端见 [MUSE-GUI](https://github.com/Cheniaaa/MUSE-GUI)。

## 核心能力

| 能力 | 说明 |
|---|---|
| 并行与异构仿真 | 基于 mjlab / MuJoCo Warp，在多个 world 中运行 UAV、UGV 等实体，支持不同角色的动作与观测维度 |
| 多智能体学习 | IPPO、MAPPO、参数共享、GRU / LSTM，以及命名 cost 与约束学习组件 |
| 可扩展研究任务 | 通过研究包声明 Task、Term、Algorithm 及 radio / sensing 模型，组合普通飞行、无线通信与感知场景 |
| 实验与运行管理 | 资源版本、冻结实验配置、单 GPU 运行队列、checkpoint 恢复、指标与日志查询 |
| 实时观察与交互 | 浏览器 Viser 和官方 MuJoCo Native 窗口；INTERACTIVE 模式支持暂停、继续、单步和指定 world reset |
| 轨迹与结果复盘 | 独立诊断轨迹采集、历史三维回放、同样本状态与动作诊断，以及基于已有轨迹的 MP4 导出 |

## 架构

```text
桌面 GUI / HTTP 客户端
          │ HTTP / WebSocket
     Muse Server ── 资源、实验、队列与持久结果
          │
     Run Worker ─── MuseEnv + driver + Algorithm
          ├─ GPU 并行仿真与学习
          ├─ 只读快照 → 实时 Viser / Native
          └─ 诊断轨迹 → 历史回放 / MP4
```

- **Experiment** 是解析后冻结的实验配置；**Run** 是一次有明确模式、预算和输出选择的执行。
- Worker 持有唯一仿真推进循环；显示、回放和视频进程不推进物理，也不重新执行策略。
- 研究者编写研究逻辑，平台负责配置解析、调度、reset、数据采集和产物管理。研究包会执行 Python 代码，请仅加载可信来源。

## 环境要求

| 项目 | 当前基线 |
|---|---|
| 后端平台 | Linux，当前目标环境为 Ubuntu 24.04 / WSL2；单主机、单 NVIDIA GPU |
| Python | 3.12，项目基线为 3.12.3 |
| 物理与学习 | mjlab 1.6.0、MuJoCo / MuJoCo Warp 3.11.0、PyTorch 2.11.0 + CUDA 12.8、RSL-RL 5.4.2 |
| 可视化 | Viser；Native 窗口与当前软件 MP4 导出路径需要可用的显示服务，如 WSLg |
| 依赖安装 | [uv](https://docs.astral.sh/uv/getting-started/installation/)；完整固定依赖见 [requirements.lock](requirements.lock) |

GPU 仿真需要可用的 NVIDIA 驱动和 CUDA 环境，不提供 CPU 物理回退。同一 Server 同时执行一个 GPU Run，其余请求排队。

## 快速开始

### 1. 安装后端

准备 Git、Python 3.12 和 uv，在 Linux / WSL 终端执行：

```bash
git clone https://github.com/Cheniaaa/MUSE.git
cd MUSE
python3.12 -m venv .myenv
source .myenv/bin/activate
uv pip install --python .myenv/bin/python --require-hashes --torch-backend cu128 -r requirements.lock
uv pip install --python .myenv/bin/python --no-deps -e .
```

已有环境可直接激活，无需重复安装。后续命令均在仓库根目录和该环境中执行。

### 2. 启动 Server

```bash
python -m muse_server.app \
  --host 127.0.0.1 \
  --port 8001 \
  --viewer-port 8081 \
  --data-dir "$HOME/.local/share/muse/catalog"
```

`--data-dir` 保存资源版本、实验、运行记录和产物；首次使用请指定新目录，后续启动使用同一目录。平台版本不兼容时使用独立目录，不覆盖或自动迁移旧研究数据。默认仅监听本机，请勿将研究服务直接暴露到公网。

Server 启动后，交互式 API 文档位于 **http://127.0.0.1:8001/docs**。

### 3. 提交基础飞行训练

另开终端，进入同一仓库并激活环境：

```bash
source .myenv/bin/activate
python examples/basic_flight/submit.py \
  --server http://127.0.0.1:8001 \
  --mode TRAIN \
  --max-decision-steps 2400 \
  --checkpoint-interval-updates 25
```

脚本通过正式 API 上传资源、保存 Experiment 并提交 Run，输出 `experiment_id`、`run_id`、`run_url` 和就绪后的 Viser 地址。追加 `--no-display` 可跳过显示请求，提交后立即返回；Run 在后台继续。

`max_decision_steps` 是整个 Run 的决策步数上限，一步同时推进全部 world，不乘 world 数，也不计内部物理子步。TRAIN 每收满 `collection_steps=H` 才更新一次；不足一个完整 block 的尾段不更新，也不进入 checkpoint。

如需可视化编辑、运行管理和历史回放，请安装 [桌面工作台](https://github.com/Cheniaaa/MUSE-GUI)。关闭 GUI 不会取消后台 Run。

## 参考示例

所有示例沿相同的 Server / Worker / checkpoint 链路运行，不另建训练循环。

| 示例 | 内容 |
|---|---|
| [Basic Flight](examples/basic_flight/README.md) | Crazyflie 规则飞行、IPPO 训练与 checkpoint 使用 |
| [Heterogeneous](examples/heterogeneous/README.md) | UAV + UGV 同 world，不同角色的动作与观测布局 |
| [Cooperative MAPPO](examples/cooperative_mappo/README.md) | 局部 actor、集中 critic、团队奖励与循环网络 |
| [ISAC + Cost](examples/isac_cost/README.md) | 运动、功率与信道混合动作，以及命名 cost |
| [History + Markov](examples/hybrid_history_markov/README.md) | 历史观测与混合动作研究配置 |
| [IMU](examples/imu_flight/README.md) / [Rangefinder](examples/rangefinder_flight/README.md) | 飞行传感器接入与观测 |
| [Indoor Radio](examples/indoor_radio_reference/README.md) / [Sensing](examples/sensing_reference/README.md) | 室内无线与感知参考场景 |

研究包在 `training_package.yaml` 中通过 `compositions` 声明任务、算法和模型组合；请求通过 `composition_id`、`model_choices` 及配置覆盖选择研究设置。选择模型不会自动启用未声明的功能。

## Checkpoint 与运行模式

| 模式 | 行为 |
|---|---|
| `TRAIN` | 采集 transition 并更新算法；正常结束或安全取消时保存学习 checkpoint，可配置周期保存 |
| `EVALUATE` | 学习策略从 checkpoint 执行确定性动作，不更新学习状态、不生成新 checkpoint；规则策略无需 checkpoint |
| `INTERACTIVE` | 冻结策略运行，并提供暂停、继续、全 world 单步和指定 world reset；学习策略需要 checkpoint |

查询状态与已登记产物，下面的 ID 均需替换为实际值：

```bash
curl http://127.0.0.1:8001/runs/RUN_ID
curl http://127.0.0.1:8001/runs/RUN_ID/artifacts
```

从列表选择 `kind=checkpoint` 条目的完整 `id`，复用原 Experiment 恢复训练或评价：

```bash
python examples/basic_flight/submit.py --experiment-id EXP_ID --mode TRAIN --checkpoint-artifact-id ARTIFACT_ID --max-decision-steps 2400 --no-display
python examples/basic_flight/submit.py --experiment-id EXP_ID --mode EVALUATE --checkpoint-artifact-id ARTIFACT_ID --max-decision-steps 500 --no-display
```

恢复保留网络、optimizer、learner RNG 和累计学习进度，但环境从新 episode 开始，不恢复历史物理状态或半段 rollout。只有 Server 已发布并登记的 checkpoint 可用于恢复；失败或强制退出不保证未发布数据。

## 诊断轨迹、历史回放与视频

录制是 Run 的可选输出，与实时显示独立。在 GUI 运行参数中配置，或向 `POST /runs` 提供 `recording`：

```json
{
  "experiment_id": "REGISTERED_EXPERIMENT_ID",
  "mode": "TRAIN",
  "max_decision_steps": 2400,
  "recording": {
    "world_ids": [0],
    "every_n_decisions": 3,
    "max_data_bytes": 16777216
  }
}
```

- **采集**：按所选 world 和 decision 间隔保存刚体位姿、速度、策略动作、reward、已启用 cost 及结束原因，并保留初态、终止和 reset 边界。`recording=null` 关闭录制。
- **存档**：轨迹使用 ZIP + JSON + NumPy NPY，包含冻结模型、配置和对象映射。容量预算针对未压缩动态数组，不是最终文件大小；耗尽后封存已有前缀，Run 继续。
- **复盘**：GUI 读取已发布轨迹，支持 world / episode 切换、前后定位和同样本诊断；Run 结束或 Server 重启后可重新打开，回放时 Server 必须在线。
- **导出**：从同一轨迹选择一个 world / episode，冻结相机、FPS 和分辨率后生成 MP4，无需重新运行仿真。视频帧率不会补出未采集的运动细节；稀疏 reward / cost 也不是区间累计值。

checkpoint 用于恢复学习，trajectory 用于保存运行历史，MP4 用于展示，三者不能互相替代。录制不包含完整 observation、原始传感器 / 无线矩阵或每个物理子步的数据。

## 项目结构

| 目录 | 职责 |
|---|---|
| [src/muse_sdk](src/muse_sdk) | 配置、研究声明、运行协议和轨迹格式 |
| [src/muse_runtime](src/muse_runtime) | 环境装配、物理与学习调度、checkpoint 和采集 |
| [src/muse_server](src/muse_server) | HTTP / WebSocket、资源、队列、显示与历史会话 |
| [src/muse_builtins](src/muse_builtins) | 内置控制器、算法、模型、计算项和资产 |
| [examples](examples) | 参考研究包与提交入口 |

## 当前边界

Muse 仍在研究开发中。已有能力包括训练、恢复、冻结评价、交互调试、实时显示、轨迹回放和视频导出，但这不等于全部场景的科学验证或策略收敛。

- 当前只支持单主机、单 NVIDIA GPU，不提供分布式训练或多 GPU 调度。
- ISAC 参考策略尚未达到无飞行失败目标，完整多 seed / 算法对照及 history 学习收益验证尚未完成。
- 交互接口当前不提供外部逐实体动作注入；messaging 及部分传感器扩展仍未完成。
- 当前 MP4 路径依赖可用显示服务，不承诺无显示服务的纯 headless 视频导出；既有小场景结果不代表通用吞吐或显存上限。
