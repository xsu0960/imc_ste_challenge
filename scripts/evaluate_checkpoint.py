#!/usr/bin/env python3
import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
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

from imc_ste import (
    COMPUTE_MODES,
    NoiseConfig,
    activation_scale_summary,
    apply_activation_range_scaling,
    apply_layerwise_mapping_gains,
    apply_layerwise_mac_tile_sizes,
    apply_layerwise_noise_scales,
    apply_layerwise_read_repeats,
    apply_output_noise_read_compensation,
    convert_model,
    enable_learnable_activation_scales,
    fold_batchnorms,
    set_compute_mode,
)
from train import (
    DATASETS,
    build_loaders,
    build_model,
    infer_num_classes,
    limit_or_none,
    scale_noise_config,
    select_device,
    run_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained checkpoints under one or more noise scales."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/cifar10_resnet18.yaml",
    )
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset", choices=DATASETS, default="cifar10")
    parser.add_argument("--eval-mode", choices=COMPUTE_MODES, default="noise")
    parser.add_argument("--noise-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument(
        "--image-size",
        type=int,
        help="Input image size. Defaults to training.image_size, or dataset native size.",
    )
    parser.add_argument(
        "--depthwise-noise-scale",
        type=float,
        help="Additional multiplier for depthwise convolution noise.",
    )
    parser.add_argument(
        "--pointwise-noise-scale",
        type=float,
        help="Additional multiplier for 1x1 pointwise convolution noise.",
    )
    parser.add_argument(
        "--linear-noise-scale",
        type=float,
        help="Additional multiplier for linear layer noise.",
    )
    parser.add_argument(
        "--mapping-gain",
        type=float,
        default=1.0,
        help="Global hardware mapping gain for noisy layers.",
    )
    parser.add_argument(
        "--depthwise-mapping-gain",
        type=float,
        help="Hardware mapping gain for depthwise convolutions.",
    )
    parser.add_argument(
        "--pointwise-mapping-gain",
        type=float,
        help="Hardware mapping gain for 1x1 pointwise convolutions.",
    )
    parser.add_argument(
        "--linear-mapping-gain",
        type=float,
        help="Hardware mapping gain for linear layers.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--fold-bn",
        action="store_true",
        help="Fold Conv2d+BatchNorm2d pairs before converting the model to noisy layers.",
    )
    parser.add_argument("--max-eval-batches", type=int, default=40)
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=1,
        help="Average logits over K independent noisy forward passes per input.",
    )
    parser.add_argument(
        "--layer-read-repeats",
        type=int,
        default=1,
        help="Average K independent noisy MAC reads inside every noisy layer.",
    )
    parser.add_argument("--depthwise-read-repeats", type=int)
    parser.add_argument("--pointwise-read-repeats", type=int)
    parser.add_argument("--linear-read-repeats", type=int)
    parser.add_argument("--activation-stat-csv", type=Path)
    parser.add_argument("--activation-target", type=float, default=4.0)
    parser.add_argument("--depthwise-activation-target", type=float)
    parser.add_argument("--pointwise-activation-target", type=float)
    parser.add_argument("--linear-activation-target", type=float)
    parser.add_argument("--activation-scale-floor", type=float, default=0.1)
    parser.add_argument(
        "--activation-stat-kinds",
        nargs="+",
        choices=["conv", "depthwise", "pointwise", "linear"],
        default=["depthwise", "pointwise"],
    )
    parser.add_argument(
        "--learnable-activation-scales",
        action="store_true",
        help="Enable bounded scales; learned checkpoints are detected automatically.",
    )
    parser.add_argument(
        "--learnable-activation-scale-max", type=float, default=1.0
    )
    parser.add_argument(
        "--output-noise-read-compensation", action="store_true"
    )
    parser.add_argument("--output-noise-read-base", type=int, default=1)
    parser.add_argument("--output-noise-read-max", type=int, default=8)
    parser.add_argument("--output-noise-read-exponent", type=float, default=2.0)
    parser.add_argument(
        "--output-noise-read-kinds",
        nargs="+",
        choices=["conv", "depthwise", "pointwise", "linear"],
        default=["depthwise", "pointwise"],
    )
    parser.add_argument("--mac-tile-size", type=int)
    parser.add_argument("--depthwise-mac-tile-size", type=int)
    parser.add_argument("--pointwise-mac-tile-size", type=int)
    parser.add_argument("--linear-mac-tile-size", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs/checkpoint_noise_sweep.csv",
    )
    return parser.parse_args()


def read_run_metadata(checkpoint: Path) -> dict[str, Any]:
    stem = checkpoint.stem.removesuffix("_best")
    metrics_path = checkpoint.with_name(f"{stem}.jsonl")
    if not metrics_path.exists():
        return {}
    metadata: dict[str, Any] = {}
    epoch_count = 0
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "metadata" in record:
            metadata = record["metadata"]
        elif "epoch" in record:
            epoch_count += 1
    if epoch_count and not metadata.get("epochs"):
        metadata["epochs"] = epoch_count
    return metadata


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_epoch_mc(
    model,
    loader,
    criterion,
    device,
    *,
    max_batches=None,
    mc_samples: int = 1,
) -> dict[str, Any]:
    if mc_samples < 1:
        raise ValueError("--mc-samples must be positive")
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    nonfinite_batches = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            logits_accum = None
            for _ in range(mc_samples):
                logits = model(images)
                if not torch.isfinite(logits).all():
                    nonfinite_batches += 1
                    break
                logits_accum = logits if logits_accum is None else logits_accum + logits
            if logits_accum is None:
                continue
            averaged_logits = logits_accum / mc_samples
            loss = criterion(averaged_logits, labels)
            if not torch.isfinite(loss):
                nonfinite_batches += 1
                continue

            total_loss += float(loss.detach().cpu()) * labels.numel()
            total_correct += (averaged_logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.numel()

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "examples": total_examples,
        "nonfinite": nonfinite_batches > 0,
        "nonfinite_batches": nonfinite_batches,
        "mc_samples": mc_samples,
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = args.seed if args.seed is not None else config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = select_device(args.device)
    training_config = config["training"]
    batch_size = args.batch_size or training_config["batch_size"]
    max_eval_batches = limit_or_none(args.max_eval_batches)
    num_classes = infer_num_classes(args.dataset, config["model"]["num_classes"])
    _, eval_loader = build_loaders(
        args.dataset,
        batch_size,
        training_config.get("num_workers", 0),
        seed,
        num_classes,
        training_config.get("augmentation", "standard"),
        args.image_size or training_config.get("image_size"),
    )
    criterion = nn.CrossEntropyLoss()

    rows = []
    for checkpoint in args.checkpoints:
        run_metadata = read_run_metadata(checkpoint)
        model_metadata = run_metadata.get("model", {})
        for noise_scale in args.noise_scales:
            noise_config = scale_noise_config(
                NoiseConfig(**config["noise"]), noise_scale
            )
            checkpoint_state = torch.load(checkpoint, map_location=device)
            learned_scale_state = {
                key: value
                for key, value in checkpoint_state.items()
                if key.endswith("activation_scale_logit")
            }
            base_checkpoint_state = {
                key: value
                for key, value in checkpoint_state.items()
                if not key.endswith("activation_scale_logit")
            }
            model = build_model(config["model"], num_classes)
            model.load_state_dict(base_checkpoint_state)
            folded_batchnorms = fold_batchnorms(model) if args.fold_bn else 0
            model = convert_model(model, noise_config, args.eval_mode).to(device)
            layerwise_noise_counts = apply_layerwise_noise_scales(
                model,
                noise_config,
                depthwise_noise_scale=args.depthwise_noise_scale,
                pointwise_noise_scale=args.pointwise_noise_scale,
                linear_noise_scale=args.linear_noise_scale,
            )
            layerwise_mapping_gain_counts = apply_layerwise_mapping_gains(
                model,
                mapping_gain=args.mapping_gain,
                depthwise_mapping_gain=args.depthwise_mapping_gain,
                pointwise_mapping_gain=args.pointwise_mapping_gain,
                linear_mapping_gain=args.linear_mapping_gain,
            )
            layerwise_read_repeat_counts = apply_layerwise_read_repeats(
                model,
                read_repeats=args.layer_read_repeats,
                depthwise_read_repeats=args.depthwise_read_repeats,
                pointwise_read_repeats=args.pointwise_read_repeats,
                linear_read_repeats=args.linear_read_repeats,
            )
            if args.activation_stat_csv is not None:
                activation_stat_counts = apply_activation_range_scaling(
                    model,
                    args.activation_stat_csv,
                    target_abs=args.activation_target,
                    scale_floor=args.activation_scale_floor,
                    kinds=tuple(args.activation_stat_kinds),
                    depthwise_target_abs=args.depthwise_activation_target,
                    pointwise_target_abs=args.pointwise_activation_target,
                    linear_target_abs=args.linear_activation_target,
                )
            else:
                activation_stat_counts = {
                    "conv": 0,
                    "depthwise": 0,
                    "pointwise": 0,
                    "linear": 0,
                    "scaled": 0,
                    "floor_limited": 0,
                    "missing": 0,
                }
            learnable_scales_enabled = bool(
                args.learnable_activation_scales or learned_scale_state
            )
            if learnable_scales_enabled:
                if args.activation_stat_csv is None:
                    raise ValueError(
                        "learnable activation scales require --activation-stat-csv"
                    )
                learnable_activation_counts = enable_learnable_activation_scales(
                    model,
                    scale_min=args.activation_scale_floor,
                    scale_max=args.learnable_activation_scale_max,
                    kinds=tuple(args.activation_stat_kinds),
                )
                if learned_scale_state:
                    incompatible = model.load_state_dict(
                        learned_scale_state, strict=False
                    )
                    if incompatible.unexpected_keys:
                        raise RuntimeError(
                            "learned activation-scale checkpoint mismatch: "
                            f"{incompatible.unexpected_keys}"
                        )
            else:
                learnable_activation_counts = {
                    "conv": 0,
                    "depthwise": 0,
                    "pointwise": 0,
                    "linear": 0,
                    "enabled": 0,
                }
            if args.output_noise_read_compensation:
                output_noise_read_counts = apply_output_noise_read_compensation(
                    model,
                    base_repeats=args.output_noise_read_base,
                    max_repeats=args.output_noise_read_max,
                    exponent=args.output_noise_read_exponent,
                    kinds=tuple(args.output_noise_read_kinds),
                )
            else:
                output_noise_read_counts = {
                    "conv": 0,
                    "depthwise": 0,
                    "pointwise": 0,
                    "linear": 0,
                    "layers": 0,
                    "read_histogram": {},
                    "mean_read_repeats": 0.0,
                    "total_read_repeats": 0,
                }
            layerwise_mac_tile_counts = apply_layerwise_mac_tile_sizes(
                model,
                mac_tile_size=args.mac_tile_size,
                depthwise_mac_tile_size=args.depthwise_mac_tile_size,
                pointwise_mac_tile_size=args.pointwise_mac_tile_size,
                linear_mac_tile_size=args.linear_mac_tile_size,
            )
            set_compute_mode(model, args.eval_mode)
            evaluation_started_at = time.perf_counter()
            if args.mc_samples == 1:
                metrics = run_epoch(
                    model,
                    eval_loader,
                    criterion,
                    device,
                    max_batches=max_eval_batches,
                    stop_on_nonfinite=True,
                )
                metrics["mc_samples"] = 1
            else:
                metrics = run_epoch_mc(
                    model,
                    eval_loader,
                    criterion,
                    device,
                    max_batches=max_eval_batches,
                    mc_samples=args.mc_samples,
                )
            evaluation_seconds = time.perf_counter() - evaluation_started_at
            row = {
                "checkpoint": str(checkpoint),
                "dataset": args.dataset,
                "model_name": model_metadata.get(
                    "name", config["model"].get("name", "")
                ),
                "checkpoint_train_mode": run_metadata.get("train_mode", ""),
                "checkpoint_seed": run_metadata.get("seed", ""),
                "checkpoint_epochs": run_metadata.get("epochs", ""),
                "checkpoint_learning_rate": run_metadata.get("learning_rate", ""),
                "checkpoint_max_train_batches": run_metadata.get(
                    "max_train_batches", ""
                ),
                "eval_seed": seed,
                "eval_mode": args.eval_mode,
                "noise_scale": noise_scale,
                "depthwise_noise_scale": args.depthwise_noise_scale,
                "pointwise_noise_scale": args.pointwise_noise_scale,
                "linear_noise_scale": args.linear_noise_scale,
                "mapping_gain": args.mapping_gain,
                "depthwise_mapping_gain": args.depthwise_mapping_gain,
                "pointwise_mapping_gain": args.pointwise_mapping_gain,
                "linear_mapping_gain": args.linear_mapping_gain,
                "fold_bn": args.fold_bn,
                "folded_batchnorms": folded_batchnorms,
                "layerwise_noise_counts": layerwise_noise_counts,
                "layerwise_mapping_gain_counts": layerwise_mapping_gain_counts,
                "layer_read_repeats": args.layer_read_repeats,
                "depthwise_read_repeats": args.depthwise_read_repeats,
                "pointwise_read_repeats": args.pointwise_read_repeats,
                "linear_read_repeats": args.linear_read_repeats,
                "layerwise_read_repeat_counts": layerwise_read_repeat_counts,
                "activation_stat_csv": str(args.activation_stat_csv)
                if args.activation_stat_csv
                else "",
                "activation_target": args.activation_target,
                "depthwise_activation_target": args.depthwise_activation_target,
                "pointwise_activation_target": args.pointwise_activation_target,
                "linear_activation_target": args.linear_activation_target,
                "activation_scale_floor": args.activation_scale_floor,
                "activation_stat_kinds": args.activation_stat_kinds,
                "activation_stat_counts": activation_stat_counts,
                "learnable_activation_scales": learnable_scales_enabled,
                "learnable_activation_scale_max": args.learnable_activation_scale_max,
                "learnable_activation_counts": learnable_activation_counts,
                "activation_scale_summary": activation_scale_summary(model),
                "output_noise_read_compensation": args.output_noise_read_compensation,
                "output_noise_read_base": args.output_noise_read_base,
                "output_noise_read_max": args.output_noise_read_max,
                "output_noise_read_exponent": args.output_noise_read_exponent,
                "output_noise_read_kinds": args.output_noise_read_kinds,
                "output_noise_read_counts": output_noise_read_counts,
                "mac_tile_size": args.mac_tile_size,
                "depthwise_mac_tile_size": args.depthwise_mac_tile_size,
                "pointwise_mac_tile_size": args.pointwise_mac_tile_size,
                "linear_mac_tile_size": args.linear_mac_tile_size,
                "layerwise_mac_tile_counts": layerwise_mac_tile_counts,
                "loss": metrics.get("loss"),
                "accuracy": metrics.get("accuracy"),
                "examples": metrics.get("examples"),
                "nonfinite": metrics.get("nonfinite"),
                "mc_samples": metrics.get("mc_samples"),
                "max_eval_batches": max_eval_batches,
                "evaluation_seconds": evaluation_seconds,
                "eval_noise": asdict(noise_config),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

    write_csv(args.output, rows)
    print(f"checkpoint_eval_csv: {args.output}")


if __name__ == "__main__":
    main()
