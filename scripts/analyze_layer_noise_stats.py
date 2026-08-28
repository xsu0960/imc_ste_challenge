#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imc_ste import NoiseConfig, NoisyConv2d, NoisyLinear, convert_model, set_compute_mode
from train import (
    DATASETS,
    build_loaders,
    build_model,
    infer_num_classes,
    limit_or_none,
    scale_noise_config,
    select_device,
)


@dataclass
class LayerStats:
    name: str
    kind: str
    module_type: str
    output_shape: str = ""
    clean_numel: int = 0
    noisy_numel: int = 0
    clean_sum_sq: float = 0.0
    clean_abs_sum: float = 0.0
    noisy_sum_sq: float = 0.0
    residual_sum_sq: float = 0.0
    stochastic_pair_sum_sq: float = 0.0
    stochastic_pair_numel: int = 0
    clean_abs_gt_5: int = 0
    clean_abs_gt_8: int = 0
    clean_abs_gt_10: int = 0
    clean_abs_p95_values: list[float] = field(default_factory=list)
    clean_abs_p99_values: list[float] = field(default_factory=list)
    residual_abs_p95_values: list[float] = field(default_factory=list)
    nonfinite_outputs: int = 0

    def add_clean(self, output: torch.Tensor) -> None:
        detached = output.detach()
        abs_output = detached.abs()
        self.output_shape = "x".join(str(item) for item in detached.shape)
        self.clean_numel += detached.numel()
        self.clean_sum_sq += float(detached.square().sum().cpu())
        self.clean_abs_sum += float(abs_output.sum().cpu())
        self.clean_abs_gt_5 += int((abs_output > 5).sum().cpu())
        self.clean_abs_gt_8 += int((abs_output > 8).sum().cpu())
        self.clean_abs_gt_10 += int((abs_output > 10).sum().cpu())
        flattened = abs_output.flatten()
        self.clean_abs_p95_values.append(float(torch.quantile(flattened, 0.95).cpu()))
        self.clean_abs_p99_values.append(float(torch.quantile(flattened, 0.99).cpu()))
        if not torch.isfinite(detached).all():
            self.nonfinite_outputs += 1

    def add_noisy(
        self,
        output: torch.Tensor,
        clean_output: torch.Tensor,
        previous_noisy_output: torch.Tensor | None = None,
    ) -> None:
        detached = output.detach()
        residual = detached - clean_output
        self.noisy_numel += detached.numel()
        self.noisy_sum_sq += float(detached.square().sum().cpu())
        self.residual_sum_sq += float(residual.square().sum().cpu())
        if previous_noisy_output is not None:
            pair_difference = detached - previous_noisy_output
            self.stochastic_pair_sum_sq += float(
                (0.5 * pair_difference.square()).sum().cpu()
            )
            self.stochastic_pair_numel += pair_difference.numel()
        flattened = residual.abs().flatten()
        self.residual_abs_p95_values.append(float(torch.quantile(flattened, 0.95).cpu()))
        if not torch.isfinite(detached).all():
            self.nonfinite_outputs += 1

    def row(self) -> dict[str, Any]:
        clean_rms = math.sqrt(self.clean_sum_sq / max(self.clean_numel, 1))
        noisy_rms = math.sqrt(self.noisy_sum_sq / max(self.noisy_numel, 1))
        residual_rms = math.sqrt(self.residual_sum_sq / max(self.noisy_numel, 1))
        stochastic_mse = self.stochastic_pair_sum_sq / max(
            self.stochastic_pair_numel, 1
        )
        total_mse = self.residual_sum_sq / max(self.noisy_numel, 1)
        bias_mse = max(total_mse - stochastic_mse, 0.0)
        stochastic_rms = math.sqrt(stochastic_mse)
        bias_rms = math.sqrt(bias_mse)
        return {
            "name": self.name,
            "kind": self.kind,
            "module_type": self.module_type,
            "output_shape": self.output_shape,
            "clean_rms": clean_rms,
            "noisy_rms": noisy_rms,
            "residual_rms": residual_rms,
            "residual_to_signal": residual_rms / max(clean_rms, 1e-12),
            "stochastic_rms": stochastic_rms,
            "stochastic_to_signal": stochastic_rms / max(clean_rms, 1e-12),
            "bias_rms": bias_rms,
            "bias_to_signal": bias_rms / max(clean_rms, 1e-12),
            "stochastic_fraction_of_mse": stochastic_mse / max(total_mse, 1e-24),
            "clean_abs_mean": self.clean_abs_sum / max(self.clean_numel, 1),
            "clean_abs_p95_mean": mean(self.clean_abs_p95_values)
            if self.clean_abs_p95_values
            else 0.0,
            "clean_abs_p99_mean": mean(self.clean_abs_p99_values)
            if self.clean_abs_p99_values
            else 0.0,
            "residual_abs_p95_mean": mean(self.residual_abs_p95_values)
            if self.residual_abs_p95_values
            else 0.0,
            "clean_abs_gt_5_frac": self.clean_abs_gt_5 / max(self.clean_numel, 1),
            "clean_abs_gt_8_frac": self.clean_abs_gt_8 / max(self.clean_numel, 1),
            "clean_abs_gt_10_frac": self.clean_abs_gt_10 / max(self.clean_numel, 1),
            "clean_numel": self.clean_numel,
            "noisy_numel": self.noisy_numel,
            "nonfinite_outputs": self.nonfinite_outputs,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect per-layer clean/noisy output statistics for a checkpoint."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--noise-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs/layer_noise_stats.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=PROJECT_ROOT / "docs/figures/layer_noise_stats.png",
    )
    return parser.parse_args()


def layer_kind(module: nn.Module) -> str:
    if isinstance(module, NoisyConv2d):
        kernel = (
            module.kernel_size
            if isinstance(module.kernel_size, tuple)
            else (module.kernel_size, module.kernel_size)
        )
        if module.is_depthwise:
            return "depthwise"
        if module.groups == 1 and kernel == (1, 1):
            return "pointwise"
        return "conv"
    if isinstance(module, NoisyLinear):
        return "linear"
    return "other"


def attach_hooks(
    model: nn.Module,
    *,
    stats: dict[str, LayerStats],
    clean_outputs: dict[str, torch.Tensor],
    previous_noisy_outputs: dict[str, torch.Tensor],
    mode: str,
) -> list[torch.utils.hooks.RemovableHandle]:
    handles = []

    def make_hook(name: str, module: nn.Module):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return
            if mode == "clean":
                clean_outputs[name] = output.detach()
                stats[name].add_clean(output)
            else:
                clean_output = clean_outputs.get(name)
                if clean_output is None:
                    raise RuntimeError(f"missing clean output for layer: {name}")
                previous_noisy_output = previous_noisy_outputs.get(name)
                stats[name].add_noisy(output, clean_output, previous_noisy_output)
                previous_noisy_outputs[name] = output.detach()

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (NoisyConv2d, NoisyLinear)):
            if name not in stats:
                stats[name] = LayerStats(
                    name=name,
                    kind=layer_kind(module),
                    module_type=module.__class__.__name__,
                )
            handles.append(module.register_forward_hook(make_hook(name, module)))
    return handles


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def color_for_kind(kind: str) -> str:
    return {
        "conv": "#4C78A8",
        "depthwise": "#F58518",
        "pointwise": "#54A24B",
        "linear": "#B279A2",
    }.get(kind, "#999999")


def plot_stats(rows: list[dict[str, Any]], figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    kinds = ["conv", "depthwise", "pointwise", "linear"]
    grouped = {
        kind: [
            float(row["residual_to_signal"])
            for row in rows
            if row["kind"] == kind and math.isfinite(float(row["residual_to_signal"]))
        ]
        for kind in kinds
    }

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2))
    ax = axes[0]
    width = 0.25
    positions = list(range(len(kinds)))
    for offset, (field, label, color) in enumerate(
        (
            ("residual_to_signal", "total", "#4C78A8"),
            ("stochastic_to_signal", "stochastic", "#F58518"),
            ("bias_to_signal", "systematic", "#54A24B"),
        )
    ):
        values = [
            mean([float(row[field]) for row in rows if row["kind"] == kind])
            if any(row["kind"] == kind for row in rows)
            else 0.0
            for kind in kinds
        ]
        ax.bar(
            [position + (offset - 1) * width for position in positions],
            values,
            width=width,
            label=label,
            color=color,
        )
    ax.set_xticks(positions, kinds)
    ax.set_title("Layer noise decomposition")
    ax.set_ylabel("RMS ratio to clean signal")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    top_rows = sorted(
        rows,
        key=lambda row: float(row["stochastic_to_signal"]),
        reverse=True,
    )[:20]
    labels = [row["name"].replace("features.", "f.") for row in top_rows][::-1]
    values = [float(row["stochastic_to_signal"]) for row in top_rows][::-1]
    colors = [color_for_kind(row["kind"]) for row in top_rows][::-1]
    ax.barh(labels, values, color=colors)
    ax.set_title("Top layers by stochastic noise")
    ax.set_xlabel("Stochastic-to-signal RMS ratio")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    top_bias_rows = sorted(
        rows,
        key=lambda row: float(row["bias_to_signal"]),
        reverse=True,
    )[:20]
    labels = [row["name"].replace("features.", "f.") for row in top_bias_rows][::-1]
    values = [float(row["bias_to_signal"]) for row in top_bias_rows][::-1]
    colors = [color_for_kind(row["kind"]) for row in top_bias_rows][::-1]
    ax.barh(labels, values, color=colors)
    ax.set_title("Top layers by systematic deviation")
    ax.set_xlabel("Bias-to-signal RMS ratio")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.noise_samples < 1:
        raise ValueError("--noise-samples must be positive")

    config = yaml.safe_load(args.config.read_text())
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)
    num_classes = infer_num_classes(args.dataset, config["model"]["num_classes"])
    noise_config = scale_noise_config(NoiseConfig(**config["noise"]), args.noise_scale)
    training_config = config["training"]

    _, eval_loader = build_loaders(
        args.dataset,
        args.batch_size,
        training_config.get("num_workers", 0),
        args.seed,
        num_classes,
        training_config.get("augmentation", "standard"),
        training_config.get("image_size"),
    )

    base_model = build_model(config["model"], num_classes)
    base_model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    clean_model = convert_model(base_model, noise_config, "clean").to(device)
    noisy_model = convert_model(base_model, noise_config, "noise").to(device)
    set_compute_mode(clean_model, "clean")
    set_compute_mode(noisy_model, "noise")
    clean_model.eval()
    noisy_model.eval()

    stats: dict[str, LayerStats] = {}
    clean_outputs: dict[str, torch.Tensor] = {}
    previous_noisy_outputs: dict[str, torch.Tensor] = {}
    clean_handles = attach_hooks(
        clean_model,
        stats=stats,
        clean_outputs=clean_outputs,
        previous_noisy_outputs=previous_noisy_outputs,
        mode="clean",
    )
    noisy_handles = attach_hooks(
        noisy_model,
        stats=stats,
        clean_outputs=clean_outputs,
        previous_noisy_outputs=previous_noisy_outputs,
        mode="noise",
    )

    max_batches = limit_or_none(args.max_batches)
    with torch.no_grad():
        for batch_index, (images, _labels) in enumerate(eval_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device)
            clean_outputs.clear()
            previous_noisy_outputs.clear()
            clean_model(images)
            for _ in range(args.noise_samples):
                noisy_model(images)

    for handle in clean_handles + noisy_handles:
        handle.remove()

    rows = [stats[name].row() for name in sorted(stats)]
    write_csv(args.output, rows)
    plot_stats(rows, args.figure)

    by_kind = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(float(row["residual_to_signal"]))
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "noise_scale": args.noise_scale,
        "batch_size": args.batch_size,
        "max_batches": max_batches,
        "noise_samples": args.noise_samples,
        "layers": len(rows),
        "residual_to_signal_mean_by_kind": {
            kind: mean(values) for kind, values in sorted(by_kind.items()) if values
        },
        "stochastic_to_signal_mean_by_kind": {
            kind: mean(
                float(row["stochastic_to_signal"])
                for row in rows
                if row["kind"] == kind
            )
            for kind in sorted(by_kind)
        },
        "bias_to_signal_mean_by_kind": {
            kind: mean(
                float(row["bias_to_signal"])
                for row in rows
                if row["kind"] == kind
            )
            for kind in sorted(by_kind)
        },
        "csv": str(args.output),
        "figure": str(args.figure),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
