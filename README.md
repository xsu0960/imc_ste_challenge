<p align="right">
  <strong>简体中文</strong> |
  <a href="./README_EN.md">English</a>
</p>

# 2026 INNO 存算算法赛道五赛题二：Architecture-Aware IMC-STE

本项目是 [2026 INNO 存算算法竞赛「赛道五：存算算法」](https://modelscope.cn/events/189/%E8%B5%9B%E9%81%93%E4%BA%94%EF%BC%9A%E5%AD%98%E7%AE%97%E7%AE%97%E6%B3%95) 中 **赛题二「直通估计器在噪声感知训练中的应用」** 的开源实现。项目面向模拟存算一体（Analog In-Memory Computing, IMC）芯片的复杂非理想性，提供架构感知的直通估计器（Straight-Through Estimator, STE）训练框架。

**作者：Xin Su**

## 论文

> **Architecture-Aware Straight-Through Estimation for Robust Analog In-Memory Computing**
>
> **[阅读英文论文（PDF）](./paper/main.pdf)** |
> **[LaTeX 源文件](./paper/main.tex)** |
> **[参考文献](./paper/references.bib)**

论文系统介绍了噪声算子、架构感知 STE、激活预调节、层类型感知重复读策略、理论分析及完整实验结果。中文技术说明见 [正式技术报告](./docs/final_report.md)。

## 项目简介

Architecture-Aware IMC-STE 是一个基于 PyTorch 的可复现噪声感知训练框架。它将完整的存算芯片噪声前向过程与可控的代理梯度结合，并针对普通卷积、逐点卷积、深度卷积和线性层采用架构感知的稳定化策略。

噪声模型覆盖九类非理想性：

- 编程误差、漂移、保持损失和温度变化；
- 输入串扰；
- 非对称非线性饱和；
- ADC 量化；
- 独立及空间相关输出噪声；
- 电源波动。

框架支持 `Linear`、普通 `Conv2d`、分组卷积、深度卷积和逐点卷积。所有主结果均采用严格统一协议：每个被转换的卷积层和线性层都在 `noise_scale=1.0` 下承受完整噪声。重复读取平均的是多次独立、全强度的物理读出，不会降低单次读出的噪声水平。

## 核心方法

- 将网络中的 `Linear` 和 `Conv2d` 递归转换为 `NoisyLinear` 和 `NoisyConv2d`；
- 提供基础、饱和感知、自适应饱和感知、方差感知及组合式代理梯度；
- 使用干净统计量对饱和敏感层进行激活预调节；
- 支持精确重复读取及仅用于训练的矩匹配近似；
- 使用有界逐层激活缩放和输出噪声感知的读取次数分配；
- 在大图卷积的空间分块和 MAC 分块之间共享物理读状态；
- 支持离线及在线的逐层、逐通道噪声统计分析。

完整算法结构和训练流程见 [算法架构与流程图](./docs/algorithm_architecture_and_workflow.md)。

## 主要结果

下表所有结果均使用 `noise_scale=1.0` 的严格统一噪声设置：

| 数据集与模型 | Clean | Direct noisy | 最佳 STE | 恢复幅度 |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-10 + ResNet18 | 95.62% | 61.42% | 90.63% | +29.21 pp |
| CIFAR-10 + EfficientNet-B0 | 91.13% | 38.93% | 86.12% | +47.19 pp |
| CIFAR-100 + ResNet18 | 78.32% | 62.98% | 69.96% | +6.98 pp |
| CIFAR-100 + EfficientNet-B0 | 71.14% | 27.76% | 61.60% | +33.84 pp |
| TinyImageNet + ResNet18 | 71.60% | 53.83% | 58.07% | +4.24 pp |
| TinyImageNet + EfficientNet-B0 | 81.00% | 0.50% | 76.30% ± 0.25% | +75.80 pp |

TinyImageNet + EfficientNet-B0 使用干净统计量激活预调节，训练阶段对深度卷积和逐点卷积执行 4 次读取，评估阶段执行 8 次精确读取。进一步采用有界的 4--8 次自适应读取策略后，敏感层读取次数降低 25.5%，完整验证耗时降低 23.0%，准确率变化不具有统计显著性。

项目还提供 VOC2007 Faster R-CNN 和 VOC2012 DeepLabV3 的跨领域机制验证，用于检验内部噪声层和大图卷积共享读取机制；这些结果不作为 COCO 或 Cityscapes 基准结果报告。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `src/imc_ste/` | 噪声模型、噪声层、网络转换、STE 和噪声统计 |
| `scripts/` | 训练、评估、统计、绘图和发布验证脚本 |
| `configs/` | 可复现实验配置 |
| `tests/` | 不依赖真实数据集的单元测试 |
| `runs/` | 发布包保留的规范化结果表 |
| `docs/final_report.md` | 中文正式技术报告 |
| `docs/theory_and_complexity.md` | 收敛性与复杂度分析 |
| `paper/main.pdf` | 英文论文 |

## 快速开始

克隆仓库：

```bash
git clone https://github.com/xsu0960/imc_ste_challenge.git
cd imc_ste_challenge
```

创建项目环境：

```bash
conda env create -f environment.yml
conda activate imc-ste
```

也可以在已有的兼容 PyTorch 环境中安装项目及测试依赖：

```bash
python -m pip install -e '.[dev]'
```

## 项目验证

运行不依赖真实数据集的完整发布检查：

```bash
PYTHONPATH=src python scripts/verify_project.py
```

该命令将检查运行环境，执行 smoke test 和 48 项单元测试，重建规范化结果表，并验证报告图片和文件哈希。

运行最小合成数据训练：

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/cifar10_resnet18.yaml \
  --dataset fake \
  --mode ste \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 2
```

## 训练示例

在 CIFAR-10 + ResNet18 上运行饱和感知 STE：

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/cifar10_resnet18.yaml \
  --dataset cifar10 \
  --mode sat_aware_ste \
  --eval-mode noise \
  --grad-clip-norm 1.0 \
  --stop-on-nonfinite
```

运行多模式、多随机种子对照实验：

```bash
PYTHONPATH=src python scripts/run_matrix.py \
  --modes clean noise ste sat_aware_ste adaptive_sat_aware_ste \
  --seeds 1 2 3
```

使用重复噪声读取评估已有 checkpoint：

```bash
PYTHONPATH=src python scripts/evaluate_checkpoint.py \
  --dataset cifar10 \
  --checkpoints runs/<checkpoint>.pt \
  --noise-scales 1.0 \
  --noise-repeats 5
```

数据集和 checkpoint 不包含在发布包中。训练脚本可以下载受支持的公开分类数据集，其他数据路径请参考各脚本的命令行帮助。

## 重建结果与论文

重建规范化结果表及主要结果图：

```bash
PYTHONPATH=src python scripts/build_key_results.py
```

重建算法架构图和流程图：

```bash
make -C docs/diagrams
```

编译英文论文：

```bash
make -C paper
```

## 发布包

生成完整代码与文档发布包：

```bash
PYTHONPATH=src python scripts/build_submission.py
```

生成仅包含论文和正式文档的发布包：

```bash
PYTHONPATH=src python scripts/build_submission.py --documents-only
```

发布包仅包含公开文档、源代码、配置、测试、规范化结果表以及正式文档实际引用的图片，不包含数据集、checkpoint、JSONL 日志、缓存和内部研究记录。

## 文档

- [英文论文](./paper/main.pdf)
- [论文 LaTeX 源文件](./paper/main.tex)
- [中文正式技术报告](./docs/final_report.md)
- [算法架构与流程图](./docs/algorithm_architecture_and_workflow.md)
- [理论与复杂度分析](./docs/theory_and_complexity.md)
- [发布包说明](./docs/submission_readme.md)

## 许可与数据

公开数据集遵循各自的原始许可，本项目不重新分发数据集。竞赛题目及主办方提供的噪声模型遵循其原始使用条款。
