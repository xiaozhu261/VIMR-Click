# VIMR-Click 全部实验整理与分析

生成日期：2026-09-09

## 1. 结论摘要

1. Fusion-only 已确认训练适配是总提升的主要来源：mIoU@5=0.7853，相对其运行内 Frozen baseline 提升 +0.1061。+Viewport 相对 Fusion-only 的净增益只有 +0.0144，且一个 seed 为负。
2. 当前可归因的模块链条是：Viewport 提供小幅、不完全一致的增益；History 提供持续增益；Residual 进一步提高精度；Safe Update 主要降低更新破坏与严重回退。
3. Safe Update 权重不存在单调最优：S1 的 mIoU@5 最高，S0 的普通回退频率最低，S3 的 NoC@85、Success@85 与严重 late-stage regression 更好。S3 是偏向交互效率与安全性的多目标选择。
4. Router 在严格 matched 对照上把 mIoU@5 从 0.7971 提高到 0.8173（+0.0202），同时把 seed 标准差从 0.0424 降到 0.0032；但普通和 late 回退率略有上升。
5. Detail Expert 没有在 Router 上继续增益：mIoU@5 下降 0.0019，早期 IoU 与 late-stage stability 也更差；其路由 argmax 在三个 seed 上全部塌缩到 Detail。
6. Viewport 的独立语义证据没有成立：Synthetic−Default 为 +0.0329，但 Synthetic−Shuffled 只有 +0.0012，且不是每个 seed 都为正。当前不能声称模型真正理解了用户 zoom/pan 意图。

## 2. 统一实验口径

- 数据：COCO 2017 train split 的固定 image-disjoint holdout；每图取最大合格非 crowd 实例，面积比例 0.01–0.60。
- 正式协议：500 个训练实例、200 个验证实例、3 epochs、最多 5 次点击、seeds 42/43/44、split seed 2026。
- 点击：首次点击 GT 距离变换最大点；后续点击预测误差最大点，正负标签取自 GT。
- 冻结：encoder、prompt encoder、mask decoder 均冻结，只训练 VIMR 新增模块。
- NoC 未达到阈值时记为 6；Success 表示 5 次内曾达到阈值；FinalSuccess 表示第 5 次后仍保持。
- n=3 只支持描述性结论；本报告不把 quick/smoke 结果用于排名或显著性声称。

## 3. Safe Update 权重扫描

模型 | λ_safe | λ_monotonic | gate init | mIoU@5 | NoC@85 | Regression | Late Regression | Late Severe>0.05
--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:
S0 (0.0/0.0) | 0 | 0 | 0.1 | 0.8201 ± 0.0134 | 3.0133 | 0.3562 | 0.4208 | 0.0133
S1 (0.2/0.1) | 0.2 | 0.1 | 0.1 | 0.8211 ± 0.0054 | 3.0200 | 0.3662 | 0.4375 | 0.0192
S2 (0.5/0.2) | 0.5 | 0.2 | 0.1 | 0.8174 ± 0.0023 | 2.9700 | 0.3917 | 0.4792 | 0.0175
S3 (1.0/0.5) | 1 | 0.5 | 0.1 | 0.8173 ± 0.0158 | 2.9183 | 0.3604 | 0.4358 | 0.0117
R0 旧 Gate≈0.5 | — | — | ≈0.5 | 0.8227 ± 0.0180 | 2.9617 | 0.3617 | 0.4225 | —

分析：S1 的 mIoU@5 最高（0.8211），S0 的 Regression 最低（0.3562），S3 的 NoC@85 最低（2.9183）且严重 late regression 为 0.0117。因此应把 S3 表述为安全/效率偏好的折中，而不是整体最优。R0 采用旧 gate 初始化，不能与 S0–S3 做严格因果归因。

## 4. 累计组件消融

模型 | 训练参数 | mIoU@5 | Regression | Late Regression
--- | ---: | ---: | ---: | ---:
Frozen baseline | 0 | 0.6772 ± 0.0000 | 0.4300 | 0.4500
Fusion-only | 686,723 | 0.7853 ± 0.0164 | 0.4650 | 0.5208
+Viewport | 686,723 | 0.7997 ± 0.0103 | 0.4538 | 0.5117
+History | 3,932,099 | 0.8121 ± 0.0085 | 0.4129 | 0.4717
+Residual | 4,099,491 | 0.8180 ± 0.0050 | 0.4471 | 0.5175
Full VIMR (S3 reference) | 4,159,588 | 0.8173 ± 0.0158 | 0.3604 | 0.4358

分析：Fusion-only 已达到 0.7853，相对其运行内 Frozen baseline 提升 +0.1061。+Viewport 相对 Fusion-only 只增加 0.0144，逐 seed 差值为 +0.0326/-0.0058/+0.0164；+History 再增加 0.0124；+Residual 再增加 0.0059，但回退变多；加入 Safe Update 后 mIoU@5 基本不变（-0.0007），Regression 从 0.4471 降至 0.3604。Fusion-only 是本次新运行，+Viewport 复用旧结果；两次 Frozen baseline 相差约 0.0021，说明净差值仍含少量 GPU 非确定性。

## 5. Viewport 信号消融

模式 | mIoU@5 | NoC@85 | Regression | Late Regression
--- | ---: | ---: | ---: | ---:
Default viewport | 0.7529 ± 0.0237 | 3.3150 | 0.4671 | 0.5225
Synthetic viewport | 0.7858 ± 0.0047 | 3.0417 | 0.5229 | 0.5867
Shuffled viewport | 0.7846 ± 0.0050 | 3.0900 | 0.5062 | 0.5617
Correct oracle | 0.7969 ± 0.0160 | 3.0117 | 0.4608 | 0.5150

预注册判据要求 Synthetic−Default 与 Synthetic−Shuffled 都至少 +0.01 且每个 seed 为正。实际第二项只有 +0.0012，判据失败。Synthetic 和 Shuffled 几乎持平，说明主要收益可能来自额外 viewport 条件/训练分布，而不是位置正确性。Correct oracle 比 Synthetic 高 0.0111，但 seed 一致性仍不足。

## 6. Router 与 Detail Expert

模型 | 训练参数 | mIoU@5 | NoC@85 | Regression | Late Regression
--- | ---: | ---: | ---: | ---: | ---:
Full matched | 4,159,588 | 0.7971 ± 0.0424 | 3.1850 | 0.3433 | 0.3875
+G/O/D Router | 4,750,948 | 0.8173 ± 0.0032 | 2.9817 | 0.3658 | 0.4242
+Router + Detail Expert | 4,754,229 | 0.8154 ± 0.0055 | 3.0250 | 0.3754 | 0.4625

分析：Router 的 matched 精度收益明确且方差更低，但稳定性指标略差。Detail Expert 相对 Router 没有收益，并且三个 seed 的 argmax 都选择 Detail Expert，属于路由塌缩，不能据此声称 global/object/detail 已形成可解释分工。

## 7. 论文可用结论与暂不可用结论

可用：

- Fusion-only 控制证明监督训练的融合适配器解释了总提升的大部分；模块贡献必须相对该控制组报告。
- History/Residual 在 Fusion-only 与 Viewport 之后仍提供递进增益。
- Safe Update 显著降低回退幅度、LostAfterSuccess 和严重 late-stage regression；其权重体现精度与安全性的权衡。
- Router 在 matched 设置下改善 mIoU@5 与跨 seed 稳定性。

暂不可用：

- 不能把 Frozen baseline 到 Full VIMR 的全部差值归因于 VIMR 创新模块。
- 不能说 synthetic viewport 提供了独立于点击位置的用户意图，因为 shuffled 没有明显下降。
- 不能说 Detail Expert 有效或路由具备清晰语义，因为性能未提升且路由塌缩。
- 不能把旧 S3 与新 Router 直接对比；新增模块改变了初始化随机数序列，必须使用 Full matched。

## 8. 下一步优先级

1. 先做 Router 的 route balancing/entropy 或弱监督，使三路均被使用，再重训 Detail Expert。
2. 为 viewport 构造与 click 去相关的真实或半真实 zoom/pan 轨迹，并加入 viewport-only、click-only、遮蔽/扰动对照。
3. 将正式验证扩大到至少 5 seeds 和更大实例集，报告 paired bootstrap CI；必要时再上 COCO val2017。
4. 主论文表建议同时报告 S3（安全性主模型）和 Router（精度主模型），不要挑单一指标决定唯一冠军。

## 9. 数据文件

- Excel 工作簿包含：实验索引、全部汇总指标、每 seed 指标、每 epoch 轨迹、四组消融、路由使用率和非正式检查。
- 所有来源路径均在工作簿中逐行保留。