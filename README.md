# Architecture-Aware IMC-STE

**Author:** Xin Su

Architecture-Aware IMC-STE is a reproducible PyTorch framework for training
convolutional neural networks under analog in-memory computing (IMC)
nonidealities. The implementation combines a full noisy forward operator with
straight-through gradient estimators, architecture-aware activation
preconditioning, and layer-specific repeated-read policies.

## Scope

The noise operator models nine classes of nonideality:

- programming error, drift, retention loss, and temperature variation;
- input crosstalk;
- asymmetric nonlinear saturation;
- ADC quantization;
- independent and spatially correlated output noise;
- supply variation.

The framework supports `Linear`, dense `Conv2d`, grouped convolution,
depthwise convolution, and pointwise convolution. All mandatory results use a
strict uniform protocol: every converted convolution and linear layer receives
the full noise configuration at `noise_scale=1.0`. Repeated reads average
independent full-strength physical reads; they do not reduce the noise assigned
to an individual read.

## Method

The main implementation provides:

- recursive conversion from `Linear`/`Conv2d` to `NoisyLinear`/`NoisyConv2d`;
- identity, saturation-aware, adaptive saturation-aware, variance-aware, and
  combined surrogate-gradient estimators;
- clean-statistics activation preconditioning for saturation-sensitive layers;
- exact repeated reads and a training-only moment-matched approximation;
- bounded per-layer activation scales and output-noise-aware read allocation;
- shared physical read state across spatial chunks and MAC tiles for
  memory-bounded large-image convolution;
- offline and optional online per-layer/per-channel noise profiling.

The architecture and end-to-end workflow are documented in
[docs/algorithm_architecture_and_workflow.md](docs/algorithm_architecture_and_workflow.md).

## Main Results

All values below use strict uniform noise at scale 1.0.

| Dataset and model | Clean | Direct noisy | Best STE | Recovery |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-10 + ResNet18 | 95.62% | 61.42% | 90.63% | +29.21 pp |
| CIFAR-10 + EfficientNet-B0 | 91.13% | 38.93% | 86.12% | +47.19 pp |
| CIFAR-100 + ResNet18 | 78.32% | 62.98% | 69.96% | +6.98 pp |
| CIFAR-100 + EfficientNet-B0 | 71.14% | 27.76% | 61.60% | +33.84 pp |
| TinyImageNet + ResNet18 | 71.60% | 53.83% | 58.07% | +4.24 pp |
| TinyImageNet + EfficientNet-B0 | 81.00% | 0.50% | 76.30% +/- 0.25% | +75.80 pp |

The TinyImageNet EfficientNet-B0 result uses clean-statistics activation
preconditioning, four depthwise/pointwise reads during training, and eight
exact reads during evaluation. A bounded 4--8-read policy reduces sensitive
layer reads by 25.5% and measured full-validation time by 23.0% without a
statistically significant accuracy change.

VOC2007 Faster R-CNN and VOC2012 DeepLabV3 are included as cross-domain
mechanism tests. They validate internal noisy layers and shared-read
large-image convolution, but they are not presented as COCO or Cityscapes
benchmark results.

## Repository Layout

| Path | Contents |
| --- | --- |
| `src/imc_ste/` | noise model, noisy layers, model conversion, STE, and profiling |
| `scripts/` | training, evaluation, statistics, plotting, and verification tools |
| `configs/` | reproducible experiment configurations |
| `tests/` | dataset-free unit tests |
| `runs/` | compact result tables included in the release package |
| `docs/final_report.md` | formal Chinese technical report |
| `docs/theory_and_complexity.md` | convergence and complexity analysis |
| `paper/main.pdf` | English conference paper |

## Installation

Create the locked environment:

```bash
conda env create -f environment.yml
conda activate imc-ste
```

Alternatively, install the package and test dependencies in an existing
compatible PyTorch environment:

```bash
python -m pip install -e '.[dev]'
```

## Verification

Run the complete dataset-free release check:

```bash
PYTHONPATH=src python scripts/verify_project.py
```

The command checks runtime imports, executes the smoke test and 48 unit tests,
rebuilds the canonical result table, checks report figures, and verifies
artifact hashes.

To run a minimal synthetic training job:

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

Datasets and checkpoints are intentionally excluded from the release archive.
Training scripts download supported public classification datasets or accept
the paths documented by their command-line help.

## Rebuilding Results and Figures

Rebuild the canonical release table and main figure:

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

## Release Package

Create the clean source-and-documentation archive:

```bash
PYTHONPATH=src python scripts/build_submission.py
```

Create a separate documentation-only archive:

```bash
PYTHONPATH=src python scripts/build_submission.py --documents-only
```

The archive is written to `dist/imc_ste_challenge_submission.zip`. It contains
only release-facing documentation, source code, configurations, tests,
canonical result tables, and figures referenced by the formal documents. It
excludes datasets, checkpoints, JSONL logs, caches, drafts, research notes, and
intermediate experiment records.

The documentation-only archive is written to
`dist/imc_ste_documents_release.zip` and contains the release README, formal
reports, paper sources and PDF, referenced figures, diagram sources, and
canonical result tables.

After extraction, verify the archive with:

```bash
python scripts/verify_project.py --package-mode
```

## Documentation

- [Formal technical report](docs/final_report.md)
- [Algorithm architecture and workflow](docs/algorithm_architecture_and_workflow.md)
- [Theory and complexity analysis](docs/theory_and_complexity.md)
- [Release package guide](docs/submission_readme.md)
- [English paper](paper/main.pdf)
- [LaTeX source](paper/main.tex)
- [BibTeX references](paper/references.bib)

## License and Data

Public datasets retain their original licenses and are not redistributed. The
challenge specification and organizer-provided noise model remain subject to
their original terms.
