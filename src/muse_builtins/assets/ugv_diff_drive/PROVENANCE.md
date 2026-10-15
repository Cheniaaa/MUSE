# 差速车资产来源

- 来源：Google DeepMind MuJoCo 官方示例 `model/car/car.xml`。
- 固定版本：本机 `mujoco-3.12.0` 发行目录；原文件 `/home/chen/MuJoCo/mujoco-3.12.0/model/car/car.xml`。
- 上游：https://github.com/google-deepmind/mujoco/blob/3.12.0/model/car/car.xml
- 许可：Apache-2.0，`LICENSE` 原样复制自该发行目录根部的 `LICENSE`，不使用相邻 Menagerie 的许可。
- 项目编译/运行版本仍为锁定的 MuJoCo 3.11.0；来源版本不改变依赖。

原文件 SHA-256：`car.xml` 为 `fc506521837557f6c6c65c624294241315ac59a1a0679ebfaa0253755a3db13f`，发行目录 `LICENSE` 为 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`。

拆分只移除环境 plane 及其 grid texture/material，并为左右轮 geom 增加 `left_wheel`、`right_wheel` 名称。车体 mesh、light/site/sensor、关节、fixed tendon、motor 与全部物理参数原样保留。

根 body 为 `car`，初始 root 高度 .03 m。支撑 geom 为 `front wheel`、`left_wheel`、`right_wheel`。轮半径 .03 m，轮距 .12 m，joint damping .03；joint actuator force 范围 ±.5，motor control 范围 ±1。控制 metadata 的速度范围是工程验收范围，不是物理限幅修改或实车极限。
