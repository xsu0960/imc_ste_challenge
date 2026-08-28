#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imc_ste import (  # noqa: E402
    NoiseConfig,
    NoisyConv2d,
    NoisyLinear,
    scale_noise_config,
    set_compute_mode,
    set_conv_chunk_rows,
    set_conv_weight_noise_scope,
)
from train_optional_ste import (  # noqa: E402
    build_detection_model,
    build_loaders,
    build_segmentation_model,
    limit_or_none,
    select_device,
)


ROLE_COLORS = {
    "backbone": "#4C78A8",
    "fpn": "#59A14F",
    "rpn": "#F28E2B",
    "roi": "#E15759",
    "classifier": "#B279A2",
    "aux_classifier": "#76B7B2",
    "other": "#9D9D9D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect role-aware clean/noisy layer statistics for optional tasks."
    )
    parser.add_argument("--task", choices=["detection", "segmentation"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--noise-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--conv-chunk-rows", type=int, default=8)
    parser.add_argument(
        "--conv-weight-noise-scope",
        choices=["read", "chunk"],
        default="read",
    )
    parser.add_argument("--detection-min-size", type=int, default=320)
    parser.add_argument("--detection-max-size", type=int, default=512)
    parser.add_argument("--trainable-backbone-layers", type=int, default=3)
    parser.add_argument("--segmentation-image-size", type=int, default=256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--figure", type=Path)
    return parser.parse_args()


def layer_kind(module: nn.Module) -> str:
    if isinstance(module, NoisyConv2d):
        kernel = module.kernel_size
        kernel = kernel if isinstance(kernel, tuple) else (kernel, kernel)
        if module.is_depthwise:
            return "depthwise"
        if module.groups == 1 and kernel == (1, 1):
            return "pointwise"
        return "conv"
    if isinstance(module, NoisyLinear):
        return "linear"
    return "other"


def layer_role(task: str, name: str) -> str:
    if task == "detection":
        if name.startswith("backbone.fpn."):
            return "fpn"
        if name.startswith("backbone.body."):
            return "backbone"
        if name.startswith("rpn."):
            return "rpn"
        if name.startswith("roi_heads."):
            return "roi"
        return "other"
    if name.startswith("aux_classifier."):
        return "aux_classifier"
    if name.startswith("classifier."):
        return "classifier"
    if name.startswith("backbone."):
        return "backbone"
    return "other"


def sampled_quantile(value: torch.Tensor, quantile: float) -> float:
    flattened = value.detach().abs().flatten()
    max_values = 131_072
    if flattened.numel() > max_values:
        stride = math.ceil(flattened.numel() / max_values)
        flattened = flattened[::stride]
    flattened = flattened[torch.isfinite(flattened)]
    if flattened.numel() == 0:
        return float("nan")
    return float(torch.quantile(flattened, quantile).cpu())


@dataclass
class OptionalLayerStats:
    name: str
    role: str
    kind: str
    module_type: str
    output_shape: str = ""
    clean_calls: int = 0
    noisy_calls: int = 0
    matched_calls: int = 0
    shape_mismatch_calls: int = 0
    clean_numel: int = 0
    noisy_numel: int = 0
    matched_numel: int = 0
    stochastic_pair_numel: int = 0
    clean_sum_sq: float = 0.0
    clean_abs_sum: float = 0.0
    noisy_sum_sq: float = 0.0
    residual_sum_sq: float = 0.0
    stochastic_pair_sum_sq: float = 0.0
    clean_abs_p95_values: list[float] = field(default_factory=list)
    clean_abs_p99_values: list[float] = field(default_factory=list)
    residual_abs_p95_values: list[float] = field(default_factory=list)
    nonfinite_outputs: int = 0

    def add_clean(self, output: torch.Tensor) -> None:
        detached = output.detach()
        self.output_shape = "x".join(str(item) for item in detached.shape)
        self.clean_calls += 1
        self.clean_numel += detached.numel()
        self.clean_sum_sq += float(detached.square().sum().cpu())
        self.clean_abs_sum += float(detached.abs().sum().cpu())
        self.clean_abs_p95_values.append(sampled_quantile(detached, 0.95))
        self.clean_abs_p99_values.append(sampled_quantile(detached, 0.99))
        if not bool(torch.isfinite(detached).all()):
            self.nonfinite_outputs += 1

    def add_noisy(
        self,
        output: torch.Tensor,
        clean_output: torch.Tensor | None,
        previous_noisy_output: torch.Tensor | None,
    ) -> None:
        detached = output.detach()
        self.noisy_calls += 1
        self.noisy_numel += detached.numel()
        self.noisy_sum_sq += float(detached.square().sum().cpu())
        if not bool(torch.isfinite(detached).all()):
            self.nonfinite_outputs += 1
        if clean_output is None or detached.shape != clean_output.shape:
            self.shape_mismatch_calls += 1
            return

        residual = detached - clean_output
        self.matched_calls += 1
        self.matched_numel += residual.numel()
        self.residual_sum_sq += float(residual.square().sum().cpu())
        self.residual_abs_p95_values.append(sampled_quantile(residual, 0.95))
        if previous_noisy_output is not None and detached.shape == previous_noisy_output.shape:
            pair_difference = detached - previous_noisy_output
            self.stochastic_pair_sum_sq += float(
                (0.5 * pair_difference.square()).sum().cpu()
            )
            self.stochastic_pair_numel += pair_difference.numel()

    def row(self) -> dict[str, Any]:
        clean_rms = math.sqrt(self.clean_sum_sq / max(self.clean_numel, 1))
        noisy_rms = math.sqrt(self.noisy_sum_sq / max(self.noisy_numel, 1))
        total_mse = self.residual_sum_sq / max(self.matched_numel, 1)
        stochastic_mse = self.stochastic_pair_sum_sq / max(
            self.stochastic_pair_numel, 1
        )
        residual_rms = math.sqrt(total_mse)
        stochastic_rms = math.sqrt(stochastic_mse)
        bias_rms = math.sqrt(max(total_mse - stochastic_mse, 0.0))
        finite_p95 = [value for value in self.clean_abs_p95_values if math.isfinite(value)]
        finite_p99 = [value for value in self.clean_abs_p99_values if math.isfinite(value)]
        finite_residual_p95 = [
            value for value in self.residual_abs_p95_values if math.isfinite(value)
        ]
        return {
            "name": self.name,
            "role": self.role,
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
            "clean_abs_p95_mean": mean(finite_p95) if finite_p95 else float("nan"),
            "clean_abs_p99_mean": mean(finite_p99) if finite_p99 else float("nan"),
            "residual_abs_p95_mean": mean(finite_residual_p95)
            if finite_residual_p95
            else float("nan"),
            "clean_calls": self.clean_calls,
            "noisy_calls": self.noisy_calls,
            "matched_calls": self.matched_calls,
            "shape_mismatch_calls": self.shape_mismatch_calls,
            "clean_numel": self.clean_numel,
            "noisy_numel": self.noisy_numel,
            "nonfinite_outputs": self.nonfinite_outputs,
        }


class LayerCollector:
    def __init__(self, model: nn.Module, task: str):
        self.phase = "clean"
        self.call_indices: dict[str, int] = defaultdict(int)
        self.clean_outputs: dict[str, list[torch.Tensor]] = {}
        self.previous_noisy_outputs: dict[str, list[torch.Tensor | None]] = {}
        self.stats: dict[str, OptionalLayerStats] = {}
        self.handles = []
        for name, module in model.named_modules():
            if not isinstance(module, (NoisyConv2d, NoisyLinear)):
                continue
            self.stats[name] = OptionalLayerStats(
                name=name,
                role=layer_role(task, name),
                kind=layer_kind(module),
                module_type=module.__class__.__name__,
            )
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def begin_clean(self) -> None:
        self.phase = "clean"
        self.call_indices.clear()
        self.clean_outputs.clear()
        self.previous_noisy_outputs.clear()

    def begin_noisy(self) -> None:
        self.phase = "noise"
        self.call_indices.clear()

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            call_index = self.call_indices[name]
            self.call_indices[name] += 1
            if self.phase == "clean":
                outputs = self.clean_outputs.setdefault(name, [])
                outputs.append(output.detach())
                self.stats[name].add_clean(output)
                return

            clean_outputs = self.clean_outputs.get(name, [])
            clean_output = (
                clean_outputs[call_index] if call_index < len(clean_outputs) else None
            )
            previous_outputs = self.previous_noisy_outputs.setdefault(name, [])
            while len(previous_outputs) <= call_index:
                previous_outputs.append(None)
            previous_output = previous_outputs[call_index]
            self.stats[name].add_noisy(output, clean_output, previous_output)
            previous_outputs[call_index] = output.detach()

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return mean(values) if values else float("nan")


def plot_stats(rows: list[dict[str, Any]], figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    roles = [role for role in ROLE_COLORS if any(row["role"] == role for row in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.8))

    ax = axes[0]
    width = 0.25
    positions = list(range(len(roles)))
    for offset, (field, label, color) in enumerate(
        (
            ("residual_to_signal", "total", "#4C78A8"),
            ("stochastic_to_signal", "stochastic", "#F28E2B"),
            ("bias_to_signal", "systematic", "#59A14F"),
        )
    ):
        values = [
            finite_mean([row for row in rows if row["role"] == role], field)
            for role in roles
        ]
        ax.bar(
            [position + (offset - 1) * width for position in positions],
            values,
            width=width,
            label=label,
            color=color,
        )
    ax.set_xticks(positions, roles, rotation=20, ha="right")
    ax.set_title("Noise decomposition by subsystem")
    ax.set_ylabel("RMS ratio to clean signal")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    top_rows = sorted(
        rows, key=lambda row: float(row["residual_to_signal"]), reverse=True
    )[:18]
    labels = [row["name"] for row in top_rows][::-1]
    values = [float(row["residual_to_signal"]) for row in top_rows][::-1]
    colors = [ROLE_COLORS.get(row["role"], "#9D9D9D") for row in top_rows][::-1]
    ax.barh(labels, values, color=colors)
    ax.set_title("Most noise-sensitive layers")
    ax.set_xlabel("Residual-to-signal RMS ratio")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[2]
    top_range_rows = sorted(
        rows, key=lambda row: float(row["clean_abs_p99_mean"]), reverse=True
    )[:18]
    labels = [row["name"] for row in top_range_rows][::-1]
    values = [float(row["clean_abs_p99_mean"]) for row in top_range_rows][::-1]
    colors = [
        ROLE_COLORS.get(row["role"], "#9D9D9D") for row in top_range_rows
    ][::-1]
    ax.barh(labels, values, color=colors)
    ax.axvline(4.0, color="#E15759", linestyle="--", linewidth=1.4, label="target 4")
    ax.set_title("Largest clean MAC output ranges")
    ax.set_xlabel("Mean absolute P99")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.25)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.noise_samples < 2:
        raise ValueError("--noise-samples must be at least 2 for variance decomposition")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)
    args.mode = "clean"
    noise_config = scale_noise_config(NoiseConfig(), args.noise_scale)
    if args.task == "detection":
        model = build_detection_model(args, noise_config)
    else:
        model = build_segmentation_model(args, noise_config)
    set_conv_chunk_rows(model, args.conv_chunk_rows)
    set_conv_weight_noise_scope(model, args.conv_weight_noise_scope)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()
    _train_loader, eval_loader = build_loaders(args)

    output_path = args.output or (
        PROJECT_ROOT / "runs" / f"optional_{args.task}_layer_noise_stats.csv"
    )
    summary_path = args.summary_output or output_path.with_name(
        f"{output_path.stem}_summary.json"
    )
    figure_path = args.figure or (
        PROJECT_ROOT
        / "docs"
        / "figures"
        / f"optional_{args.task}_layer_noise_stats.png"
    )

    collector = LayerCollector(model, args.task)
    max_batches = limit_or_none(args.max_batches)
    started_at = time.perf_counter()
    processed_batches = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(eval_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            if args.task == "detection":
                images = [image.to(device) for image in batch[0]]
            else:
                images = batch[0].to(device)
            collector.begin_clean()
            set_compute_mode(model, "clean")
            model(images)
            for _ in range(args.noise_samples):
                collector.begin_noisy()
                set_compute_mode(model, "noise")
                model(images)
            processed_batches += 1
    elapsed_seconds = time.perf_counter() - started_at
    collector.close()

    rows = [collector.stats[name].row() for name in sorted(collector.stats)]
    write_csv(output_path, rows)
    plot_stats(rows, figure_path)

    rows_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_role[row["role"]].append(row)
    summary = {
        "task": args.task,
        "checkpoint": str(args.checkpoint),
        "noise_scale": args.noise_scale,
        "conv_weight_noise_scope": args.conv_weight_noise_scope,
        "processed_batches": processed_batches,
        "noise_samples": args.noise_samples,
        "elapsed_seconds": elapsed_seconds,
        "layers": len(rows),
        "layers_by_role": {
            role: len(role_rows) for role, role_rows in sorted(rows_by_role.items())
        },
        "residual_to_signal_mean_by_role": {
            role: finite_mean(role_rows, "residual_to_signal")
            for role, role_rows in sorted(rows_by_role.items())
        },
        "stochastic_to_signal_mean_by_role": {
            role: finite_mean(role_rows, "stochastic_to_signal")
            for role, role_rows in sorted(rows_by_role.items())
        },
        "bias_to_signal_mean_by_role": {
            role: finite_mean(role_rows, "bias_to_signal")
            for role, role_rows in sorted(rows_by_role.items())
        },
        "shape_mismatch_calls": sum(int(row["shape_mismatch_calls"]) for row in rows),
        "nonfinite_outputs": sum(int(row["nonfinite_outputs"]) for row in rows),
        "csv": str(output_path),
        "figure": str(figure_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
