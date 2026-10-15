# 路程测量 / 二维位置 EFIM 参考

这是 **EVALUATE-only** 科学接线例，不训练策略、不实现定位器。将 `environment/` 和
`training_package/` 分别通过现有 Catalog/API 注册，替换 `request_evaluate.json` 中四个资源身份字段，
用该请求 validate / 保存 Experiment，再创建 `mode=EVALUATE`、`max_decision_steps=200` 的 Run。
规则没有 checkpoint；沿唯一 Worker/driver 推进。

## 固定实验设定

- 两个 world；`alpha/beta` 各一架 Crazyflie，起点 `(0,0,1)/(8,0,1)` m。
- 静止点目标为 `anchor_group[2]` 的 root，`(4,4,0)` m；已知高度，只评价二维位置参数。
  固定实体仅提供理想散射中心，不给目标分配 RF endpoint 或发射。
- station `(4,8,1)` m；jammer `(4,-8,1)` m，在 band_a 固定发射 1 mW。
- 2.412/2.437 GHz，带宽各 5 MHz，channel ID `7/-3`；同一实际发射同时供上行通信和单站探测。
- 显式合成参数：290 K、NF 7 dB、RCS 1 m²、minimum distance 1 m。
- physics dt 2 ms、decision 20 ticks、epoch 上限 7 ticks：`0→7→14→20`。
  最新测量只对应末窗口 14→20（12 ms）；EFIM 累计完整 40 ms，**不再乘时长**。
- 四段各 50 decision：零功率、各自频道 0.1 mW、各自频道 1 mW、alpha 切到 -3 与 beta 同频。
  每段 motion `[0,.1,0,0]`，遵循现有 body_velocity 的 0.1 m/s 控制范围；timeout 500 decision。

`blocker` 静态 box 默认远离路径 `(20,20,.5)`，half-size `(.1,.05,.5)`。
测试可移动到 `(2,2,.5)` 阻断 alpha→target，或 `(2,-4,1)` 阻断 jammer→alpha；不会全场统一衰减。

## 合成 law 与可见性

自由空间各向同性点散射、已知关联/同步、平坦基带谱（β²=B²/12）、独立有效 Gaussian 噪声；
理想抑制自泄漏/探测 tx 的 direct 分量，其他同频发射用自身 direct 路径计白噪声，box 不透射。
这不是波形处理器或实测雷达精度模型，也没有宣称低 SNR/极短窗口的真实工作范围。
每次 measurement 是总路程 `h=d_tx_target+d_target_rx` 的带噪样本，**单站为两倍单程距离**，允许负 Gaussian 样本。
每观测噪声强度是自由未知 nuisance；EFIM 不利用已知位置相关 variance 的额外信息。

局部 Observation 只输出本 slot `[sample/100m, valid, window_ticks/20]`，不读 truth、variance 或 FIM。
world cost 逐 target 在 S/F 上求和，仅完整有效且满足 `λ_min>2 eps64 λ_max` 时计算
`sqrt(trace(J^-1))`；任一 target 不可逆/invalid/超过 0.25 m 则 cost=1，否则0。
不使用 pinv、jitter 或有限大数代替无穷界。完整采集记录的理论界不是仅看最新样本的 actor 可达误差。
约束是 `episode_discounted_sum`：discount=.99、budget=10、lambda_initial=0、lambda_learning_rate=.01；
规则仅记录 cost，不更新 lambda。radio rate world reward 是通信模型的真实消费者。

## 显式变体

在 YAML `model_bindings.sensing` 选择：

| builtin | 必须同步显式配置 |
|---|---|
| `free_space_path_length_efim_2d` | 默认，measurement + EFIM |
| `free_space_path_length_measurement` | 请求 `term_config.world.cost.position_bound.enabled=false`，清空 `cost_config.world` |
| `free_space_position_efim_2d` | 请求每个 role 的 `term_config.roles.<role>.observation.measurement.enabled=false` |

没有自动能力补齐。关闭两个消费者后 sensing binding 不激活/不实例化。
无 jammer 对照：将请求 `background_emitters[0].config.active` 改为 `[false]`，而不是删掉其唯一归属。
仅 sensing：删除 geometry.links 与 communication_allocations，关闭 world reward `rate`；radio 无消费者不执行。
每个 role 的 phase control 必须显式配置；role/slot、目标数、resource 和关系由场景决定。
局部 Term 的“一条单站关系/slot”选择规则不适用于多目标或异站时，应关闭该 Term 或另选消费者，科学 model 不固化首例拓扑。

## 原始收据

`RunPlan.artifact_directory/logs/worker.log` 中逐步输出 `SENSING_RESULT ` + JSON：
`policy_observations` 是 **action 前**实际收到的局部输入；`sensing` 是 **同一步结束、auto-reset 前**的 data/valid/start/end。
二者采集时刻不同，不直接做同一步相等断言。收据另含 `world_reward/world_cost/reset`、radio data/valid 与 sample tick/interval。
无 radio 时 radio 为空且其时间为 null；未启用 world reward 时 world_reward 为 null。
subset reset 不重置其他 world 的 phase。

名义静止量级（非运行验收）：无 jammer 最新12ms标准差约 .01131/.01142 m，40ms位置界 .004470 m；
仅 band_a jammer 时 alpha 标准差约1.248 m、位置界 .3471 m。实际科学/Worker证据由 M4.1 验收报告提供。
