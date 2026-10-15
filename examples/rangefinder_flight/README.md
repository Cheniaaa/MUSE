# Rangefinder Flight：理想下视测距、IMU 与 IPPO

沿用 [IMU Flight](../imu_flight/README.md) 的双 role IPPO：scout 1 架、guard 2 架，B=2、`root_seed=7`。动作、奖励、终止条件、网络和优化配置不变，在每 slot 的24维观测上增加三个测距 selection，得到 **27维 actor 输入**。环境和控制器直接复用 `examples/basic_flight/`，不复制脚本、环境或运行框架。

## 观测与设备工作点

两个 role 都使用以下观测：

| selection_id | 维度 | 含义 |
|---|---:|---|
| `target` | 3 | 目标减 world 位置真值 |
| `velocity` | 3 | world 速度真值 |
| `orientation` | 4 | wxyz 姿态真值 |
| `imu_gyro` | 3 | `imu_flu` 局部角速度，rad/s |
| `imu_specific_force` | 3 | IMU 安装位置处比力，m/s² |
| `imu_valid` | 2 | `[gyro, specific_force]` 有效性0/1 |
| `imu_age` | 2 | 同序测量 age，s |
| `rangefinder_distance` | 1 | 从名义测距原点沿下视中心射线到表面的距离，m |
| `rangefinder_valid` | 1 | 当前测距是否可用，0/1 |
| `rangefinder_age` | 1 | 距上次测距采样的 age，s |
| `previous_action` | 4 | 上一步 motion |

**不是纯传感器导航，也不是完整 Flow deck 仿真**：仍保留位置相关目标误差、速度和姿态真值，没有状态估计器或光流。两个 role 使用各自本地 actor/critic。

- **平台定义**：两个 group 选择 `cf2_imu_rangefinder_down`。安装、量程、输出定义由平台冻结，用户只设置 `imu_update_rate_hz=100`、`rangefinder_update_rate_hz=5`，不能覆盖设备 mount 或量程。
- **名义安装，不是实物测量**：相对当前 CF2 root body，位置为 `[0,0,-.01]`m，wxyz为 `[0,1,0,0]`，使 site +Z 沿名义 body −Z。射线跟随机体完整姿态，测得沿束距离而非 world 垂直高度。原始资产未提供精确 deck 配准；该1cm下偏是获批平台建模选择，不是官方 CAD 值、单件标定或精度承诺。
- **有条件的硬件参考**：采用 VL53L1X **LONG、140ms** 测量预算，理想有效区间 **0.04..4m（含端点）**。4m来自暗环境、白色88%反射率目标覆盖完整 FoV 的典型条件，不保证强光、低反射率或任意表面下仍有4m。近盲区不意味着近处没有物体。
- **不是 Bitcraze 默认固件**：所核实官方固件使用 MEDIUM/25ms预算、25ms任务周期并等待 dataReady；本例的 LONG/140ms、5Hz不是该默认执行时序，也不把“4m最高距离”与“50Hz最高频率”拼成一个保证工作点。
- **独立采样**：`physics_dt=.002`、`control_substeps=20` 保持原物理基线；IMU每5ticks同步采样，range每100ticks（200ms/5Hz）发布。所选预算要求周期严格大于144ms；2ms网格最短合法周期146ms（73ticks）。40Hz的25ms在此网格非整数tick，会被拒绝，不舍入、不改dt。
- **缓存与初值**：reset不推进物理，在tick0捕获真实几何。水平初始body高度1m、名义测距原点下偏.01m，地面距离为.99m，valid=1、tick=0、age=0；IMU空中零推力自由落体初值接近0，不填入+g。前4次决策（tick20/40/60/80）range持有初值、age为.04/.08/.12/.16s；tick100更新并归零age。读取 Term 不触发查询，未到期保留测量；有效但陈旧不会自动变为invalid。
- **无可用距离**：miss、低于.04m或高于4m均发布distance=4m、valid=0，采样tick仍推进，age从这次采样计时，不是距最后命中的时间；4m本身可为有效端点，必须同时读valid。非finite读数是错误，不伪装成miss。

## 近似与适用边界

1. 到期读取**当前几何 snapshot**，省略实际140ms积分过程与交付延迟；预算用于约束参考工作点和允许发布率，不能把本例当作已模拟历史积分或硬件时序。
2. 单中心 ray 不是 VL53L1X 典型27°完整 FoV（对角视角）的光学回波模型。连续平坦表面可用；桌沿、细杆、多表面和有限视场遮挡可能给出不同实物读数。原生 alpha/几何命中规则不模拟材料反射、弱信号、噪声或 dropout。
3. 保留原 CF2 动力学，**未增加 Flow deck 的1.6g载荷及相应惯量**；不声称验证了实物装载 deck 后的飞行性能。
4. 5Hz控制 Muse 缓存发布，不节流底层原生测距计算；原生 rangefinder 随 forward/RK4持续计算，新增查询成本不能用慢发布率掩盖。

来源与研究记录：

- [Bitcraze Flow deck v2](https://www.bitcraze.io/products/flow-deck-v2/)：下方安装、VL53L1X、4m能力条件和整块deck质量。
- [ST DS12385 Rev8](https://www.st.com/resource/en/datasheet/vl53l1x.pdf) §3.5.2、§7.1/7.3/7.4：LONG/140ms、近距与目标/光照条件；[UM2356 Rev3](https://www.st.com/resource/en/user_manual/um2356-vl53l1x-api-user-manual-stmicroelectronics.pdf) §2.2/2.5.2：周期严格大于预算+4ms。
- [固定官方 firmware zranger2.c](https://github.com/bitcraze/crazyflie-firmware/blob/f6e0f0a3b526861328caf3296a7ecf00bb915d39/src/deck/drivers/src/zranger2.c) 与 [Flow v2 调用](https://github.com/bitcraze/crazyflie-firmware/blob/f6e0f0a3b526861328caf3296a7ecf00bb915d39/src/deck/drivers/src/flowdeck_v1v2.c)。
- [时序/量程研究报告](../../.planning/mjlab-spike/rangefinder_timing_report.md)含资料revision、hash和固定firmware commit；[安装研究报告](../../.planning/mjlab-spike/rangefinder_mount_report.md)记录缺失的精确机械坐标链。安装报告的历史阻塞已由[当前获批设计§2](../../docs/superpowers/specs/2026-10-12-muse-rangefinder-design.md)中的名义近似授权替代，不把它当成实物坐标证据。

## 上传、短训练与恢复

从 `/project/mjc_ws` 激活环境并启动 Server，使用新的 data-dir 导出当前 metadata；控制器仍只有 `basic_flight/controller.json`：

```bash
source /project/mjc_ws/.myenv/bin/activate
python -m muse_server.app --host 127.0.0.1 --port 8001 --viewer-port 8081 \
  --controller examples/basic_flight/controller.json --data-dir /tmp/muse-rangefinder-demo
```

沿用 [Basic Flight 正式 API 流程](../basic_flight/README.md)：

1. 将 `examples/basic_flight/environment/` 内容打包为ZIP，根层直接包含 `environment.yaml` 及资产，上传到 `POST /catalog/environments`。
2. 将本目录 `training_package/` 内容打包为另一ZIP，根层直接包含 `__init__.py`、`training_package.yaml`，上传到 `POST /catalog/training-packages`。两次上传均使用 `Content-Type: application/zip`，不套外层目录。
3. 把实际返回的 `resource_id` / `revision` 写入 `request_train.json` 对应占位字段，再 `POST /experiments`。不要上传普通飞行或 IMU 包替代本包，它们没有三个新增 selections。
4. 用返回的 experiment ID 创建 TRAIN Run；默认 `collection_steps=24`，预算须为24的整数倍，例如：

   ```json
   {"experiment_id":"REGISTERED_EXPERIMENT_ID","mode":"TRAIN","max_decision_steps":2400}
   ```

   短验收在保存实验前将 `algorithm_config` 改为 `{"collection_steps":4,"num_learning_epochs":1,"num_mini_batches":1}`，运行 **8 decisions / 2 updates**（160 physics ticks），覆盖tick100后的测距更新。episode上限500 decisions与Run预算独立。**短训练仅证明注册、测量、学习与checkpoint链路，不证明策略收敛或实机可用。**
5. 从 `GET /runs/{run_id}` 获取正式登记的 checkpoint artifact ID，冻结评估：

   ```json
   {"experiment_id":"REGISTERED_EXPERIMENT_ID","mode":"EVALUATE","checkpoint_artifact_id":"REGISTERED_CHECKPOINT_ARTIFACT_ID","max_decision_steps":2}
   ```

   预期2 decisions / 0 updates。恢复训练也使用原实验ID和checkpoint ID。恢复开启新episode，真实初始化测量为tick0/age0，不恢复中途物理状态或继承训练末端采样tick。

本目录没有新增 `submit.py`。`basic_flight/submit.py` 默认上传普通飞行包，不能用于首次注册本例；登记后可复用其 `--experiment-id` / `--checkpoint-artifact-id` 路径。Server `/docs` 提供API字段。

## CPU 契约检查

```bash
source /project/mjc_ws/.myenv/bin/activate
python -m pytest -q tests/integration/test_rangefinder_training.py::test_rangefinder_example_resolves_real_actor_inputs --tb=short
```

该用例真实注册环境/训练包并resolve，检查27维、平台固定设备定义、IMU5ticks/range100ticks及activation；不运行GPU、HTTP或物理。真实HTTP短训练和 `tests/gpu/test_rangefinder_checkpoint.py` 的Worker恢复验收须独占GPU串行执行，不能用CPU解析通过代替。
