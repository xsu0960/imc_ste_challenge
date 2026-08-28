#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/summary.csv")
    parser.add_argument("--metric", default="eval.accuracy")
    parser.add_argument(
        "--group-by",
        default=(
            "dataset,model_name,train_mode,eval_mode,epochs,"
            "learning_rate,lr_scheduler,label_smoothing,augmentation,"
            "grad_clip_norm,noise_scale,max_train_batches,max_eval_batches"
        ),
        help="Comma-separated metadata fields used for aggregate rows.",
    )
    return parser.parse_args()


def get_nested(value: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def numeric_metric(value: Any) -> float:
    if value is None:
        return -math.inf
    try:
        return float(value)
    except (TypeError, ValueError):
        return -math.inf


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


def flatten_run(path: Path, metric: str) -> Optional[dict[str, Any]]:
    metadata, epochs = read_jsonl(path)
    if not epochs:
        return None

    model = metadata.get("model", {})
    final_epoch = epochs[-1]
    best_epoch = max(epochs, key=lambda item: numeric_metric(get_nested(item, metric)))
    return {
        "path": str(path),
        "dataset": metadata.get("dataset", ""),
        "model_name": model.get("name", ""),
        "seed": metadata.get("seed", ""),
        "train_mode": metadata.get("train_mode", ""),
        "eval_mode": metadata.get("eval_mode", ""),
        "epochs": len(epochs),
        "learning_rate": metadata.get("learning_rate", ""),
        "momentum": metadata.get("momentum", ""),
        "weight_decay": metadata.get("weight_decay", ""),
        "label_smoothing": metadata.get("label_smoothing", ""),
        "lr_scheduler": metadata.get("lr_scheduler", ""),
        "lr_milestones": metadata.get("lr_milestones", ""),
        "lr_gamma": metadata.get("lr_gamma", ""),
        "augmentation": metadata.get("augmentation", ""),
        "eval_every": metadata.get("eval_every", ""),
        "grad_clip_norm": metadata.get("grad_clip_norm", ""),
        "noise_scale": metadata.get("noise_scale", ""),
        "stop_on_nonfinite": metadata.get("stop_on_nonfinite", ""),
        "max_train_batches": metadata.get("max_train_batches", ""),
        "max_eval_batches": metadata.get("max_eval_batches", ""),
        "stopped_early": any(item.get("stopped_early", False) for item in epochs),
        "final_metric": get_nested(final_epoch, metric),
        "best_metric": get_nested(best_epoch, metric),
        "best_epoch": best_epoch.get("epoch"),
        "final_eval_loss": get_nested(final_epoch, "eval.loss"),
        "final_eval_accuracy": get_nested(final_epoch, "eval.accuracy"),
    }


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def aggregate(rows: Iterable[dict[str, Any]], group_fields: list[str]):
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        groups[key].append(row)

    output_rows = []
    for key, group_rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        final_values = [row["final_metric"] for row in group_rows]
        best_values = [row["best_metric"] for row in group_rows]
        final_values = [value for value in final_values if value is not None]
        best_values = [value for value in best_values if value is not None]
        row = {field: key[index] for index, field in enumerate(group_fields)}
        row.update(
            {
                "runs": len(group_rows),
                "final_mean": mean(final_values) if final_values else "",
                "final_std": stdev(final_values) if len(final_values) > 1 else 0.0,
                "final_ci95": ci95(final_values),
                "best_mean": mean(best_values) if best_values else "",
                "best_std": stdev(best_values) if len(best_values) > 1 else 0.0,
                "best_ci95": ci95(best_values),
            }
        )
        output_rows.append(row)
    return output_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    group_fields = [
        field.strip() for field in args.group_by.split(",") if field.strip()
    ]
    run_rows = []
    for path in sorted(args.runs_dir.glob("*.jsonl")):
        row = flatten_run(path, args.metric)
        if row is not None:
            run_rows.append(row)

    summary_rows = aggregate(run_rows, group_fields)
    write_csv(args.output, summary_rows)
    print(
        json.dumps(
            {
                "runs": len(run_rows),
                "groups": len(summary_rows),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
