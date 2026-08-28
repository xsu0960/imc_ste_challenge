<p align="right">
  <a href="./README.md">简体中文</a> |
  <strong>English</strong>
</p>

# 2026 INNO Track 5, Problem 2: Architecture-Aware IMC-STE

This repository is the open-source implementation for **Problem 2, “Application of Straight-Through Estimators in Noise-Aware Training,”** in the [2026 INNO In-Memory Computing Algorithm Competition, Track 5: In-Memory Computing Algorithms](https://modelscope.cn/events/189/%E8%B5%9B%E9%81%93%E4%BA%94%EF%BC%9A%E5%AD%98%E7%AE%97%E7%AE%97%E6%B3%95). It provides an architecture-aware straight-through estimator (STE) framework for complex nonidealities in analog in-memory computing (IMC) hardware.

**Author:** Xin Su

## Paper

> **Architecture-Aware Straight-Through Estimation for Robust Analog In-Memory Computing**
>
> **[Read the paper (PDF)](./paper/main.pdf)** |
> **[LaTeX source](./paper/main.tex)** |
> **[References](./paper/references.bib)**

The paper presents the noisy operator, architecture-aware STE, activation preconditioning, layer-type-aware repeated reads, theoretical analysis, and complete experimental results. A formal Chinese technical report is available [here](./docs/final_report.md).

## Overview

Architecture-Aware IMC-STE is a reproducible PyTorch framework for training convolutional neural networks under analog IMC nonidealities. It combines a full noisy forward operator with controlled surrogate gradients and architecture-aware stabilization for dense, pointwise, depthwise, and linear layers.

The noise operator models nine classes of nonideality:

- programming error, drift, retention loss, and temperature variation;
- input crosstalk;
- asymmetric nonlinear saturation;
- ADC quantization;
- independent and spatially correlated output noise;
- supply variation.

The framework supports `Linear`, dense `Conv2d`, grouped convolution, depthwise convolution, and pointwise convolution. All primary results use a strict uniform protocol: every converted convolution and linear layer receives the full noise configuration at `noise_scale=1.0`. Repeated reads average independent full-strength physical reads; they do not reduce the noise assigned to an individual read.

## Method

- Recursive conversion from `Linear` and `Conv2d` to `NoisyLinear` and `NoisyConv2d`;
- identity, saturation-aware, adaptive saturation-aware, variance-aware, and combined surrogate gradients;
- clean-statistics activation preconditioning for saturation-sensitive layers;
- exact repeated reads and a training-only moment-matched approximation;
- bounded per-layer activation scales and output-noise-aware read allocation;
- shared physical read state across spatial chunks and MAC tiles for large-image convolution;
- offline and online per-layer and per-channel noise profiling.

See [Algorithm Architecture and Workflow](./docs/algorithm_architecture_and_workflow.md) for the complete design.

## Main Results

All values below use strict uniform noise at `noise_scale=1.0`.

| Dataset and model | Clean | Direct noisy | Best STE | Recovery |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-10 + ResNet18 | 95.62% | 61.42% | 90.63% | +29.21 pp |
| CIFAR-10 + EfficientNet-B0 | 91.13% | 38.93% | 86.12% | +47.19 pp |
| CIFAR-100 + ResNet18 | 78.32% | 62.98% | 69.96% | +6.98 pp |
| CIFAR-100 + EfficientNet-B0 | 71.14% | 27.76% | 61.60% | +33.84 pp |
| TinyImageNet + ResNet18 | 71.60% | 53.83% | 58.07% | +4.24 pp |
| TinyImageNet + EfficientNet-B0 | 81.00% | 0.50% | 76.30% +/- 0.25% | +75.80 pp |

The TinyImageNet EfficientNet-B0 result uses clean-statistics activation preconditioning, four depthwise and pointwise reads during training, and eight exact reads during evaluation. A bounded 4--8-read policy reduces sensitive-layer reads by 25.5% and measured full-validation time by 23.0% without a statistically significant accuracy change.

VOC2007 Faster R-CNN and VOC2012 DeepLabV3 are included as cross-domain mechanism tests for internal noisy layers and shared-read large-image convolution. They are not presented as COCO or Cityscapes benchmark results.

## Repository Layout

| Path | Contents |
| --- | --- |
| `src/imc_ste/` | noise model, noisy layers, model conversion, STE, and profiling |
| `scripts/` | training, evaluation, statistics, plotting, and release verification |
| `configs/` | reproducible experiment configurations |
| `tests/` | dataset-free unit tests |
| `runs/` | canonical result tables included in the release |
| `docs/final_report.md` | formal Chinese technical report |
| `docs/theory_and_complexity.md` | convergence and complexity analysis |
| `paper/main.pdf` | English paper |

## Quick Start

Clone the repository:

```bash
git clone https://github.com/xsu0960/imc_ste_challenge.git
cd imc_ste_challenge
```

Create the project environment:

```bash
conda env create -f environment.yml
conda activate imc-ste
```

Alternatively, install the package and test dependencies in an existing compatible PyTorch environment:

```bash
python -m pip install -e '.[dev]'
```

## Verification

Run the complete dataset-free release check:

```bash
PYTHONPATH=src python scripts/verify_project.py
```

The command checks runtime imports, executes the smoke test and 48 unit tests, rebuilds the canonical result table, checks report figures, and verifies artifact hashes.

Run a minimal synthetic training job:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/cifar10_resnet18.yaml \
  --dataset fake \
  --mode ste \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 2
```

## Training

Run CIFAR-10 ResNet18 with saturation-aware STE:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/cifar10_resnet18.yaml \
  --dataset cifar10 \
  --mode sat_aware_ste \
  --eval-mode noise \
  --grad-clip-norm 1.0 \
  --stop-on-nonfinite
```

Run a controlled multi-mode matrix:

```bash
PYTHONPATH=src python scripts/run_matrix.py \
  --modes clean noise ste sat_aware_ste adaptive_sat_aware_ste \
  --seeds 1 2 3
```

Evaluate a checkpoint with repeated noisy reads:

```bash
PYTHONPATH=src python scripts/evaluate_checkpoint.py \
  --dataset cifar10 \
  --checkpoints runs/<checkpoint>.pt \
  --noise-scales 1.0 \
  --noise-repeats 5
```

Datasets and checkpoints are intentionally excluded from the release archive. Training scripts download supported public classification datasets or accept paths documented by their command-line help.

## Rebuilding Results and Paper

Rebuild the canonical result table and main figure:

```bash
PYTHONPATH=src python scripts/build_key_results.py
```

Rebuild the algorithm diagrams:

```bash
make -C docs/diagrams
```

Build the English paper:

```bash
make -C paper
```

## Release Packages

Create the clean source and documentation archive:

```bash
PYTHONPATH=src python scripts/build_submission.py
```

Create a documentation-only archive:

```bash
PYTHONPATH=src python scripts/build_submission.py --documents-only
```

Release archives contain only public documentation, source code, configurations, tests, canonical result tables, and figures referenced by formal documents. They exclude datasets, checkpoints, JSONL logs, caches, and internal research records.

## Documentation

- [English paper](./paper/main.pdf)
- [Paper LaTeX source](./paper/main.tex)
- [Formal Chinese technical report](./docs/final_report.md)
- [Algorithm architecture and workflow](./docs/algorithm_architecture_and_workflow.md)
- [Theory and complexity analysis](./docs/theory_and_complexity.md)
- [Release package guide](./docs/submission_readme.md)

## License and Data

Public datasets retain their original licenses and are not redistributed. The challenge specification and organizer-provided noise model remain subject to their original terms.
