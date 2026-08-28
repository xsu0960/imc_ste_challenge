#!/usr/bin/env python3
import argparse
import csv
import math
import random
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imc_ste import (  # noqa: E402
    COMPUTE_MODES,
    NoiseConfig,
    apply_layerwise_noise_scales,
    convert_model,
    scale_noise_config,
    set_compute_mode,
)
from train import (  # noqa: E402
    DATASETS,
    build_loaders,
    build_model,
    infer_num_classes,
    limit_or_none,
    run_epoch,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep per-layer noise scales for one trained checkpoint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/cifar10_efficientnet_b0_formal.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, default="cifar10")
    parser.add_argument("--eval-mode", choices=COMPUTE_MODES, default="dw_clean_noise")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--image-size",
        type=int,
        help="Input image size. Defaults to training.image_size, or dataset native size.",
    )
    parser.add_argument("--depthwise-noise-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--pointwise-noise-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--linear-noise-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs/layerwise_noise_sweep.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "runs/layerwise_noise_sweep_summary.csv",
    )
    return parser.parse_args()


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    device = select_device(args.device)
    training_config = config["training"]
    batch_size = args.batch_size or training_config["batch_size"]
    max_eval_batches = limit_or_none(args.max_eval_batches)
    num_classes = infer_num_classes(args.dataset, config["model"]["num_classes"])

    _, eval_loader = build_loaders(
        args.dataset,
        batch_size,
        training_config.get("num_workers", 0),
        args.seed,
        num_classes,
        training_config.get("augmentation", "standard"),
        args.image_size or training_config.get("image_size"),
    )
    criterion = nn.CrossEntropyLoss()
    base_noise_config = scale_noise_config(
        NoiseConfig(**config["noise"]), args.noise_scale
    )

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    scale_grid = list(
        product(
            args.depthwise_noise_scales,
            args.pointwise_noise_scales,
            args.linear_noise_scales,
        )
    )

    for combo_index, (depthwise_scale, pointwise_scale, linear_scale) in enumerate(
        scale_grid
    ):
        model = build_model(config["model"], num_classes)
        model = convert_model(model, base_noise_config, args.eval_mode).to(device)
        layer_counts = apply_layerwise_noise_scales(
            model,
            base_noise_config,
            depthwise_noise_scale=depthwise_scale,
            pointwise_noise_scale=pointwise_scale,
            linear_noise_scale=linear_scale,
        )
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        set_compute_mode(model, args.eval_mode)

        values: list[float] = []
        for repeat in range(args.repeats):
            repeat_seed = args.seed + combo_index * 1000 + repeat
            random.seed(repeat_seed)
            torch.manual_seed(repeat_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(repeat_seed)
            metrics = run_epoch(
                model,
                eval_loader,
                criterion,
                device,
                max_batches=max_eval_batches,
                stop_on_nonfinite=True,
            )
            accuracy = float(metrics["accuracy"])
            values.append(accuracy)
            row = {
                "checkpoint": str(args.checkpoint),
                "dataset": args.dataset,
                "model_name": config["model"].get("name", ""),
                "eval_mode": args.eval_mode,
                "noise_scale": args.noise_scale,
                "depthwise_noise_scale": depthwise_scale,
                "pointwise_noise_scale": pointwise_scale,
                "linear_noise_scale": linear_scale,
                "repeat": repeat + 1,
                "seed": repeat_seed,
                "loss": metrics.get("loss"),
                "accuracy": accuracy,
                "examples": metrics.get("examples"),
                "nonfinite": metrics.get("nonfinite"),
                "max_eval_batches": max_eval_batches,
                "layerwise_noise_counts": layer_counts,
                "eval_noise": asdict(base_noise_config),
            }
            raw_rows.append(row)
            print(row, flush=True)

        summary_rows.append(
            {
                "checkpoint": str(args.checkpoint),
                "dataset": args.dataset,
                "model_name": config["model"].get("name", ""),
                "eval_mode": args.eval_mode,
                "noise_scale": args.noise_scale,
                "depthwise_noise_scale": depthwise_scale,
                "pointwise_noise_scale": pointwise_scale,
                "linear_noise_scale": linear_scale,
                "runs": len(values),
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "ci95": ci95(values),
                "min": min(values),
                "max": max(values),
                "layerwise_noise_counts": layer_counts,
            }
        )

    write_csv(args.output, raw_rows)
    write_csv(args.summary_output, summary_rows)
    print(f"layerwise_sweep_csv: {args.output}")
    print(f"layerwise_sweep_summary_csv: {args.summary_output}")


if __name__ == "__main__":
    main()
