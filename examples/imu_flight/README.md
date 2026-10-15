# IMU Flight：理想 IMU 与 IPPO

本示例沿用 `../basic_flight/training_package_ippo/` 的双 role IPPO 定义：scout 1 架、guard 2 架，B=2，`root_seed=7`。动作接口、奖励、终止条件、网络及优化配置保持不变，只替换/增加 IMU 观测。使用 builtin CF2 资产，不复制环境、控制器或创建新运行器。

## 观测与测量边界

两个 role 均将原 `angular_velocity_body` 真值观测替换为 `imu_gyro`，并启用其余三个 IMU selection。每个 slot 的 actor 输入为 24 维：

| selection_id | 维度 | 含义 |
|---|---:|---|
| `target` | 3 | `relative_target`，目标减 world 位置真值 |
| `velocity` | 3 | `velocity_world`，world 速度真值 |
| `orientation` | 4 | `orientation_wxyz`，姿态真值 |
| `imu_gyro` | 3 | `imu_flu` 局部轴角速度，rad/s |
| `imu_specific_force` | 3 | `imu_flu` 安装位置处比力，m/s² |
| `imu_valid` | 2 | 固定顺序 `[gyro, specific_force]`，有效性 0/1 |
| `imu_age` | 2 | 同序测量距采样时刻的 age，s |
| `previous_action` | 4 | 上一步 motion |

**这不是纯 IMU 导航**：策略仍可读取位置相关目标误差、速度和姿态真值；没有状态估计器。两个 role 各自使用本地 actor/critic，不引入集中式 critic。

- **理想测量**：复用 CF2 原生 `body_gyro` / `body_linacc` 与 `imu` site，当前 site 与 body 重合且同向。没有设备噪声、偏置、延迟、jitter 或 dropout；理想不代表没有仿真数值误差。
- **采样率**：两个 group 显式选择 `cf2_imu`、`update_rate_hz=100`。`physics_dt=0.002` 对应 500 Hz 物理推进，每 5 physics ticks 同步采样 gyro / specific_force；`control_substeps=20` 对应 25 Hz 决策。缓存按 group 采样，role 只读投影，读取不触发新采样。100 Hz 是本实验配置，不是实机硬件频率声明。
- **初始读数**：reset 在不推进物理的 tick 0 捕获真实初值，`valid=[1,1]`、`age=[0,0]`。本示例在空中零推力起步，初始 gyro 和比力接近 0，是自由落体读数，不应人为填入 +g。平动时比力为 `Rᵀ(a_world-g_world)`；支撑静止或施加悬停推力时才对应约 `Rᵀ(-g_world)`。
- **保持与 age**：未到采样 tick 时保留上一测量，age 继续按物理时间增长；合法但陈旧的值不会自动 invalid。本配置决策末端恰逢采样，因此正常发布时 age 为 0，不意味着读取时才测量。
- **后续误差边界**：未来 noise/bias 属于到期采样边界，latency/dropout 属于测量发布边界，不属于 Term 或 policy observation corruption。当前配置不能设置这些尚未实现的字段；不能据此证明实机鲁棒性，也不保证加入误差后旧策略无需重新评估或训练。

除 sensor profile/config 外，`request_train.json` 沿用普通飞行请求的所有物理参数、起点和目标：RK4、density=1.225、viscosity=0.000018、float32；没有为了采样修改动力学。

## 上传与训练

所有命令从 `/project/mjc_ws` 执行，先激活环境：

```bash
source /project/mjc_ws/.myenv/bin/activate
python -m muse_server.app --host 127.0.0.1 --port 8001 --viewer-port 8081 \
  --controller examples/basic_flight/controller.json --data-dir /tmp/muse-imu-demo
```

使用新的 data-dir 导出当前平台 metadata；控制器唯一来源仍为 `examples/basic_flight/controller.json`。资源注册沿用 [Basic Flight](../basic_flight/README.md) 的正式 API 流程：

1. **环境从 `examples/basic_flight/environment/` 上传**，不是当前目录。将其内容打包为 ZIP，根层须直接包含 `environment.yaml` 及其资产，不套外层目录。
2. 将 `examples/imu_flight/training_package/` 的内容打包为另一 ZIP，根层直接包含 `__init__.py` 和 `training_package.yaml`。
3. 分别以 `Content-Type: application/zip` 向 `POST /catalog/environments` 与 `POST /catalog/training-packages` 上传。将响应中的实际 `resource_id` / `revision` 写入本示例 `request_train.json` 的对应占位字段，再提交到 `POST /experiments`。不要上传 `basic_flight/training_package_ippo/`：它没有本示例的 IMU selections。
4. 用返回的 `experiment_id` 创建训练 Run：

   ```json
   {"experiment_id": "REGISTERED_EXPERIMENT_ID", "mode": "TRAIN", "max_decision_steps": 2400}
   ```

   默认 H=`collection_steps=24`，TRAIN 预算必须为 H 的整数倍；episode 上限为 500 decision，独立于 Run 预算。短验收可在保存实验前将 `algorithm_config` 覆盖为 `{"collection_steps":4,"num_learning_epochs":1,"num_mini_batches":1}`，然后运行 8 decision（两次更新）。短训练只证明链路，不证明策略收敛。
5. 从 `GET /runs/{run_id}` 读取状态、指标与已登记 checkpoint。恢复训练或冻结评估时，向 `POST /runs` 同时传入原实验 ID 和 `checkpoint_artifact_id`；恢复开启新 episode，并重新捕获 tick 0 测量，不恢复中途物理状态。

本目录没有 `submit.py`。`basic_flight/submit.py` 默认上传它自身的普通飞行包，不能用于首次注册本 IMU 示例；实验登记后可复用其 `--experiment-id` / `--checkpoint-artifact-id` 路径。API 字段和资源上传入口也可通过 Server 的 `/docs` 查看。

## CPU 契约检查

```bash
source /project/mjc_ws/.myenv/bin/activate
python -m pytest -q tests/integration/test_imu_training.py::test_imu_example_resolves_real_actor_inputs --tb=short
```

该用例通过真实 Catalog 注册、解析示例，检查每 slot 24 维 actor 输入、四个 IMU selections、5-tick 采样周期及 sensor activation；不执行物理、GPU 或 HTTP 训练。实际传感器物理值及短训练需单独串行验收。
