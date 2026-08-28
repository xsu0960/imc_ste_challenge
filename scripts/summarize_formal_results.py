#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"

TASKS = {
    "cifar10_resnet18": ("cifar10", "resnet18", "CIFAR-10 + ResNet18"),
    "cifar10_efficientnet_b0": (
        "cifar10",
        "efficientnet_b0",
        "CIFAR-10 + EfficientNet-B0",
    ),
    "cifar100_resnet18": ("cifar100", "resnet18", "CIFAR-100 + ResNet18"),
    "cifar100_efficientnet_b0": (
        "cifar100",
        "efficientnet_b0",
        "CIFAR-100 + EfficientNet-B0",
    ),
    "tinyimagenet_resnet18": (
        "tinyimagenet",
        "resnet18",
        "TinyImageNet + ResNet18",
    ),
    "tinyimagenet_resnet18_imagenet224": (
        "tinyimagenet",
        "resnet18",
        "TinyImageNet + ResNet18 ImageNet-224",
    ),
    "tinyimagenet_efficientnet_b0": (
        "tinyimagenet",
        "efficientnet_b0",
        "TinyImageNet + EfficientNet-B0",
    ),
    "tinyimagenet_efficientnet_b0_imagenet224": (
        "tinyimagenet",
        "efficientnet_b0",
        "TinyImageNet + EfficientNet-B0 ImageNet-224",
    ),
}

MODES = [
    "clean_checkpoint_noise_eval",
    "noise",
    "ste",
    "sat_aware_ste",
    "adaptive_sat_aware_ste",
]

CSV_CANDIDATES = {
    "clean_checkpoint_noise_eval": [
        "{task}_clean120_noise_eval_repeats.csv",
        "{task}_clean120_noise_eval.csv",
        "{task}_clean40_noise_eval_repeats.csv",
        "{task}_clean15_noise_eval_repeats.csv",
        "{task}_clean10_noise_eval_repeats.csv",
    ],
    "noise": [
        "{task}_noise_lr0p001_twostage_noise_eval_repeats.csv",
        "{task}_noise_twostage_noise_eval_repeats.csv",
    ],
    "ste": [
        "{task}_ste_uniform_lr0p0005_10ep300b_twostage_noise_eval_repeats.csv",
        "{task}_ste_lr0p001_twostage_noise_eval_repeats.csv",
        "{task}_ste_twostage_noise_eval_repeats.csv",
    ],
    "sat_aware_ste": [
        "{task}_sat_aware_ste_lr0p001_twostage_noise_eval_repeats.csv",
        "{task}_sat_aware_ste_twostage_noise_eval_repeats.csv",
        "{task}_twostage_sat_noise_eval_repeats.csv",
    ],
    "adaptive_sat_aware_ste": [
        "{task}_adaptive_sat_aware_ste_lr0p001_twostage_noise_eval_repeats.csv",
        "{task}_adaptive_sat_aware_ste_twostage_noise_eval_repeats.csv",
        "{task}_adaptive_sat_aware_ste_earlybest_noise_eval_repeats.csv",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the first-batch formal noisy evaluation CSVs."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory containing run jsonl files and repeat-eval CSVs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUNS_DIR / "formal_five_mode_summary.csv",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {}
    epochs: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "metadata" in record:
            metadata = record["metadata"]
        elif "epoch" in record:
            epochs.append(record)
    return metadata, epochs


def eval_accuracy(record: dict[str, Any]) -> float | None:
    try:
        value = record["eval"]["accuracy"]
    except KeyError:
        return None
    if isinstance(value, (float, int)) and math.isfinite(value):
        return float(value)
    return None


def best_training_eval(
    runs_dir: Path, dataset: str, model_name: str, mode: str
) -> tuple[float | None, str]:
    best_value: float | None = None
    best_path = ""
    for path in runs_dir.glob("*.jsonl"):
        metadata, epochs = read_jsonl(path)
        model = metadata.get("model", {})
        if metadata.get("dataset") != dataset:
            continue
        if model.get("name") != model_name:
            continue
        if metadata.get("train_mode") != mode:
            continue
        values = [value for epoch in epochs if (value := eval_accuracy(epoch)) is not None]
        if not values:
            continue
        candidate = max(values)
        if best_value is None or candidate > best_value:
            best_value = candidate
            best_path = str(path)
    return best_value, best_path


def clean_eval_accuracy(runs_dir: Path, dataset: str, model_name: str) -> tuple[float | None, str]:
    return best_training_eval(runs_dir, dataset, model_name, "clean")


def find_csv(runs_dir: Path, task: str, mode: str) -> Path | None:
    for template in CSV_CANDIDATES[mode]:
        path = runs_dir / template.format(task=task)
        if path.exists():
            return path
    return None


def read_accuracies(path: Path) -> list[float]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        values = []
        for row in reader:
            try:
                noise_scale = float(row.get("noise_scale", 1.0))
                accuracy = float(row["accuracy"])
            except (TypeError, ValueError, KeyError):
                continue
            if noise_scale == 1.0 and math.isfinite(accuracy):
                values.append(accuracy)
        return values


def summarize_values(values: list[float]) -> dict[str, float | int | str]:
    if not values:
        return {
            "runs": 0,
            "mean": "",
            "std": "",
            "ci95": "",
            "min": "",
            "max": "",
        }
    if len(values) == 1:
        deviation = 0.0
        ci95 = 0.0
    else:
        deviation = stdev(values)
        ci95 = 1.96 * deviation / math.sqrt(len(values))
    return {
        "runs": len(values),
        "mean": mean(values),
        "std": deviation,
        "ci95": ci95,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for task, (dataset, model_name, display_name) in TASKS.items():
        clean_accuracy, clean_source = clean_eval_accuracy(args.runs_dir, dataset, model_name)
        for mode in MODES:
            csv_path = find_csv(args.runs_dir, task, mode)
            values = read_accuracies(csv_path) if csv_path else []
            summary = summarize_values(values)
            best_eval, best_eval_source = (
                (clean_accuracy, clean_source)
                if mode == "clean_checkpoint_noise_eval"
                else best_training_eval(args.runs_dir, dataset, model_name, mode)
            )
            mean_value = summary["mean"]
            reached_90 = isinstance(mean_value, float) and mean_value >= 0.9
            rows.append(
                {
                    "task": task,
                    "display_name": display_name,
                    "dataset": dataset,
                    "model_name": model_name,
                    "mode": mode,
                    "clean_eval_accuracy": clean_accuracy if clean_accuracy is not None else "",
                    "best_training_eval_accuracy": best_eval if best_eval is not None else "",
                    **summary,
                    "reached_90_percent": "yes" if reached_90 else "no",
                    "source_csv": str(csv_path) if csv_path else "",
                    "best_training_source": best_eval_source,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"formal_summary_csv: {args.output}")


if __name__ == "__main__":
    main()
