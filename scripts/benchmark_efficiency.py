#!/usr/bin/env python3
"""Measure training cost for clean, plain STE, and online-profile STE."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from imc_ste import (  # noqa: E402
    NoiseConfig,
    NoisyConv2d,
    NoisyLinear,
    OnlineGradientProfile,
    convert_model,
    set_compute_mode,
)

BENCHMARK_MODES = ("clean", "ste", "online_profile")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--modes", nargs="+", choices=BENCHMARK_MODES, default=list(BENCHMARK_MODES))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "efficiency_benchmark.csv")
    parser.add_argument(
        "--figure",
        type=Path,
        default=PROJECT_ROOT / "docs" / "figures" / "efficiency_benchmark.png",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "efficiency_benchmark_metadata.json",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.image_size < 1 or args.num_classes < 2:
        parser.error("batch size and image size must be positive; num classes must be at least 2")
    if args.warmup_steps < 0 or args.steps < 1:
        parser.error("warmup steps must be non-negative and measured steps must be positive")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cifar_resnet18(num_classes: int) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[max(0, index)]


def make_model(
    mode: str,
    base_state: dict[str, torch.Tensor],
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    model = build_cifar_resnet18(num_classes)
    model.load_state_dict(base_state)
    if mode != "clean":
        compute_mode = "ste" if mode == "ste" else "variance_aware_ste"
        model = convert_model(model, NoiseConfig(), compute_mode=compute_mode)
    return model.to(device).train()


def run_step(
    *,
    mode: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    online_profile: OnlineGradientProfile | None,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    if online_profile is not None:
        online_profile.begin_clean()
        set_compute_mode(model, "clean")
        with torch.no_grad():
            model(inputs)
        online_profile.begin_noisy()
        set_compute_mode(model, "variance_aware_ste")

    logits = model(inputs)
    if online_profile is not None:
        online_profile.finalize_batch()
    loss = criterion(logits, targets)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss in benchmark mode {mode}")
    loss.backward()
    optimizer.step()
    if online_profile is not None:
        online_profile.disable()
    return float(loss.detach())


def benchmark_mode(
    *,
    mode: str,
    args: argparse.Namespace,
    base_state: dict[str, torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, float | int | str], dict[str, float | int]]:
    seed_everything(args.seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = make_model(mode, base_state, args.num_classes, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    online_profile = None
    if mode == "online_profile":
        online_profile = OnlineGradientProfile(
            model,
            ("",),
            ema_decay=0.95,
            variance_strength=0.5,
            bias_strength=0.25,
            scale_floor=0.5,
            warmup_updates=0,
        )

    for _ in range(args.warmup_steps):
        run_step(
            mode=mode,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            inputs=inputs,
            targets=targets,
            online_profile=online_profile,
        )
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
    else:
        baseline_allocated = baseline_reserved = 0

    durations_ms: list[float] = []
    losses: list[float] = []
    for _ in range(args.steps):
        synchronize(device)
        start = time.perf_counter()
        losses.append(
            run_step(
                mode=mode,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                inputs=inputs,
                targets=targets,
                online_profile=online_profile,
            )
        )
        synchronize(device)
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(durations_ms)
    std_ms = statistics.stdev(durations_ms) if len(durations_ms) > 1 else 0.0
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    else:
        peak_allocated = peak_reserved = 0

    noisy_layers = sum(isinstance(module, (NoisyConv2d, NoisyLinear)) for module in model.modules())
    row: dict[str, float | int | str] = {
        "mode": mode,
        "device": str(device),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps,
        "mean_step_ms": mean_ms,
        "std_step_ms": std_ms,
        "median_step_ms": statistics.median(durations_ms),
        "p95_step_ms": percentile(durations_ms, 0.95),
        "throughput_images_s": args.batch_size * 1000.0 / mean_ms,
        "baseline_allocated_mib": baseline_allocated / 2**20,
        "peak_allocated_mib": peak_allocated / 2**20,
        "incremental_peak_allocated_mib": max(0, peak_allocated - baseline_allocated) / 2**20,
        "baseline_reserved_mib": baseline_reserved / 2**20,
        "peak_reserved_mib": peak_reserved / 2**20,
        "mean_loss": statistics.mean(losses),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "noisy_layers": noisy_layers,
    }
    profile_summary = online_profile.summary() if online_profile is not None else {}
    if online_profile is not None:
        online_profile.close()
    del optimizer, criterion, model, online_profile
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, profile_summary


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["mode"]) for row in rows]
    colors = ["#4C78A8", "#F58518", "#54A24B"][: len(rows)]
    metrics = (
        ("mean_step_ms", "Mean step time (ms)", False),
        ("throughput_images_s", "Throughput (images/s)", False),
        ("incremental_peak_allocated_mib", "Incremental peak memory (MiB)", False),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, (field, title, _) in zip(axes, metrics):
        values = [float(row[field]) for row in rows]
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=18)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle("ResNet18 training efficiency on synthetic CIFAR input")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def environment_metadata(device: torch.device) -> dict[str, str | bool]:
    metadata: dict[str, str | bool] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if device.type == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name(device)
        metadata["cuda_runtime"] = str(torch.version.cuda)
    return metadata


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    seed_everything(args.seed)
    base_model = build_cifar_resnet18(args.num_classes)
    base_state = {name: value.detach().cpu().clone() for name, value in base_model.state_dict().items()}
    del base_model
    inputs = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    targets = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

    rows: list[dict[str, float | int | str]] = []
    profiles: dict[str, dict[str, float | int]] = {}
    for mode in args.modes:
        row, profile = benchmark_mode(
            mode=mode,
            args=args,
            base_state=base_state,
            inputs=inputs,
            targets=targets,
            device=device,
        )
        rows.append(row)
        profiles[mode] = profile
        print(json.dumps(row, ensure_ascii=True))

    clean_time = next((float(row["mean_step_ms"]) for row in rows if row["mode"] == "clean"), None)
    for row in rows:
        row["time_vs_clean"] = float(row["mean_step_ms"]) / clean_time if clean_time else float("nan")

    write_csv(args.output, rows)
    plot_results(args.figure, rows)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "benchmark": "single-process synthetic-data training-step microbenchmark",
        "model": "torchvision ResNet18 with CIFAR stem",
        "noise": asdict(NoiseConfig()),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": environment_metadata(device),
        "online_profile": profiles.get("online_profile", {}),
        "interpretation": (
            "Step-level dispersion is not an independent-run confidence interval. "
            "Online profile includes its paired clean forward and profile reductions."
        ),
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"csv: {args.output}")
    print(f"figure: {args.figure}")
    print(f"metadata: {args.metadata_output}")


if __name__ == "__main__":
    main()
