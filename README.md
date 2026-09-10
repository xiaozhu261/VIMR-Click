# VIMR-Click

基于 [EfficientViT-SAM](https://github.com/mit-han-lab/efficientvit) 的 click-based interactive segmentation 科研原型。项目在冻结的 EfficientViT-SAM 主干上加入视口状态、交互历史、掩码残差和安全更新机制，用于研究多轮点击分割的精度、交互效率与更新稳定性。

> 当前状态：研究原型。仓库中的结论来自 COCO 2017 小规模固定子集，不代表完整 COCO benchmark 结果。

## 方法概览

原始 EfficientViT-SAM 的 Image Encoder、Prompt Encoder 和 Mask Decoder 在正式协议中保持冻结。标有 `NEW` 的部分是本项目新增模块：

```text
image
  -> [Frozen] EfficientViT Image Encoder
  -> image embedding [B,256,64,64] -----------------------------+
                                                                  |
clicks -> [Frozen] SAM Prompt Encoder -> sparse/dense prompts -----+
                                                                  |
viewport [B,7] -> [NEW] Viewport Intent Encoder ------------------+
                         -> [NEW, optional] G/O/D Router           |
click/view/mask history -> [NEW] Click History Memory ------------+-> [NEW] Interaction Fusion
previous masks -> [NEW] Mask Residual Analyzer -------------------+          |
                                                                              v
                                                               [Frozen] SAM Mask Decoder
                                                                              |
                                                                              v
                                                                        draft mask
                                                                              |
                                                                              v
                                                                 [NEW] Safe Update Gate
                                                                              |
                                                                              v
                                                               [NEW, optional] Detail Expert
                                                                              |
                                                                              v
                                                                         final mask
```

新增模块位于 [`efficientvit/models/vimr`](efficientvit/models/vimr)：

| 模块 | 作用 | 插入位置 |
| --- | --- | --- |
| Viewport Intent Encoder | 将 7 维 zoom/pan/action 状态编码为 viewport token 和 ROI map | Prompt 融合前 |
| Click History Memory | 用因果 Transformer 汇总点击、视口和历史 mask 统计 | Prompt 融合前 |
| Mask Residual Analyzer | 分析前两轮 mask 的不确定性与时间残差 | Fusion 和 Safe Update 输入端 |
| Interaction Fusion | 让交互 token 对缓存图像特征做 cross-attention | Prompt Encoder 与 Mask Decoder 之间 |
| Safe Update Gate | 以像素级 gate 保守融合上一轮 mask 与新 draft mask | Mask Decoder 后 |
| G/O/D Router | 融合 global、object、detail 三种图像上下文 | Viewport Encoder 后，可选 |
| Detail Expert | 在边界与不确定区域预测局部修正 | Safe Update 后，可选 |

Safe Update 使用：

```text
final = previous + gate * (draft - previous)
```

Gate 最后一层初始化为约 `0.1`，训练损失为 BCE、Dice、Safe loss 和 Monotonic loss 的组合。

## 当前实验结果

统一正式协议：COCO 2017 `train2017` 固定 image-disjoint 划分，500 个训练实例、200 个验证实例、3 epochs、最多 5 次点击、seeds 42/43/44、split seed 2026。每个 epoch 验证并按 `mIoU@5` 保存最佳 checkpoint。

### 累计组件消融

| 模型 | 训练参数 | mIoU@5 | Regression Rate | Late Regression |
| --- | ---: | ---: | ---: | ---: |
| Frozen baseline | 0 | 0.6772 | 0.4300 | 0.4500 |
| Fusion-only | 686,723 | 0.7853 ± 0.0164 | 0.4650 | 0.5208 |
| +Viewport | 686,723 | 0.7997 ± 0.0103 | 0.4538 | 0.5117 |
| +History | 3,932,099 | 0.8121 ± 0.0085 | 0.4129 | 0.4717 |
| +Residual | 4,099,491 | 0.8180 ± 0.0050 | 0.4471 | 0.5175 |
| Full VIMR (S3) | 4,159,588 | 0.8173 ± 0.0158 | 0.3604 | 0.4358 |

Fusion-only 已解释 Frozen baseline 到 `+Viewport` 的大部分提升，因此它是必要的可训练强基线，不能将全部增益归因于 viewport。History 和 Residual 提供递进的精度收益；Safe Update 的主要贡献是降低回退幅度和严重 late-stage regression，而不是进一步提高最终 mIoU。

### Router 与 Detail Expert

| 模型 | mIoU@5 | 说明 |
| --- | ---: | --- |
| Full VIMR matched rerun | 0.7971 ± 0.0424 | 严格匹配对照 |
| +G/O/D Router | 0.8173 ± 0.0032 | 精度和跨 seed 稳定性提高 |
| +Router + Detail Expert | 0.8154 ± 0.0055 | 未超过 Router-only |

Router 当前是有潜力的可选扩展；Detail Expert 尚未产生稳定增益，并观察到路由偏向 Detail 的塌缩现象。

### Viewport 信号风险

Synthetic viewport 相对 Default viewport 的 `mIoU@5` 提高约 `0.0329`，但相对 Shuffled viewport 仅提高约 `0.0012`，且跨 seed 不一致。因此目前不能声称模型已经学习到独立于点击位置的真实 zoom/pan 意图。

完整结果与限制见 [`outputs/vimr_all_experiments_summary/VIMR_all_experiments_report.md`](outputs/vimr_all_experiments_summary/VIMR_all_experiments_report.md)。

## 环境安装

建议使用 Python 3.10：

```bash
conda create -n efficientvit python=3.10
conda activate efficientvit
pip install -U -r requirements.txt
```

准备 EfficientViT-SAM-L0 checkpoint：

```text
assets/checkpoints/efficientvit_sam/efficientvit_sam_l0.pt
```

模型权重和 COCO 数据不随仓库发布。

## COCO 数据布局

默认脚本读取：

```text
dataset/coco/
├── annotations_trainval2017/
│   └── annotations/
│       └── instances_train2017.json
└── train2017/
    └── train2017/
        └── *.jpg
```

`dataset/` 已加入 `.gitignore`，不会被提交到 GitHub。

## 训练与评估

运行统一三种子协议：

```bash
python tools/run_vimr_coco_protocol.py
```

主要指标包括 `mIoU@1/2/3/5`、`NoC@85/90`、`Success@85/90`、Regression Rate，以及第 4/5 次点击后的 late-stage regression。

运行 Safe Update 权重扫描：

```bash
python tools/run_vimr_safe_update_sweep.py
```

运行累计组件消融，其中包含严格的 Fusion-only 对照：

```bash
python tools/run_vimr_component_ablation.py
```

运行 viewport 信号对照：

```bash
python tools/run_vimr_viewport_signal_ablation.py
```

运行 Router/Detail Expert matched 消融：

```bash
python tools/run_vimr_router_detail_ablation.py
```

## 可视化

完成 viewport 信号实验后运行：

```bash
python tools/visualize_vimr_viewport_ablation.py
```

该工具对比 Frozen baseline、Default、Synthetic、Shuffled 和 Correct-oracle viewport 的逐点击预测、IoU 与 viewport 框。

## 测试

```bash
python -m unittest discover -s tests -v
python tools/count_vimr_params.py --model efficientvit-sam-vimr-l0
```

## 代码入口

- 模型组装与前向流程：[`efficientvit/models/vimr/model.py`](efficientvit/models/vimr/model.py)
- 新增模块：[`efficientvit/models/vimr/modules.py`](efficientvit/models/vimr/modules.py)
- 训练损失：[`efficientvit/models/vimr/losses.py`](efficientvit/models/vimr/losses.py)
- 多轮状态预测器：[`efficientvit/models/vimr/predictor.py`](efficientvit/models/vimr/predictor.py)
- 正式 COCO 协议：[`tools/run_vimr_coco_protocol.py`](tools/run_vimr_coco_protocol.py)
- 详细设计说明：[`docs/vimr_click_mvp.md`](docs/vimr_click_mvp.md)

## 当前限制

- 正式结果仅基于小规模 COCO 子集和 3 个随机种子，主要用于方法筛选。
- 同一验证集同时用于 checkpoint 选择与最终报告，结果可能存在乐观偏差。
- Synthetic viewport 与点击位置高度相关，独立 viewport 语义尚未建立。
- Detail Expert 当前没有稳定收益，G/O/D Router 仍需 route balancing 或弱监督。
- 尚未在更大测试集、其他数据集和其他 SAM 骨干上验证泛化能力。

## 上游项目与许可证

本项目基于 MIT HAN Lab 的 [EfficientViT](https://github.com/mit-han-lab/efficientvit) 与 [EfficientViT-SAM](https://arxiv.org/abs/2402.05008)。原始实现保留在现有 `efficientvit` 与 `applications` 目录中。

仓库沿用上游项目的 [Apache License 2.0](LICENSE)。使用本项目时，请同时遵守 EfficientViT、Segment Anything、COCO 数据集和相关 checkpoint 的许可条款。

如果 EfficientViT-SAM 对研究有帮助，请引用：

```bibtex
@article{zhang2024efficientvit,
  title={EfficientViT-SAM: Accelerated Segment Anything Model Without Performance Loss},
  author={Zhang, Zhuoyang and Cai, Han and Han, Song},
  journal={arXiv preprint arXiv:2402.05008},
  year={2024}
}
```
