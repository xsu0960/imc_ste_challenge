#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize repeated checkpoint-evaluation CSV files."
    )
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Optional seed labels corresponding one-to-one with --inputs.",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds is not None and len(args.seeds) != len(args.inputs):
        raise ValueError("--seeds must have the same length as --inputs")
    rows = []
    for index, path in enumerate(args.inputs):
        with path.open(newline="") as handle:
            input_rows = list(csv.DictReader(handle))
        if args.seeds is not None:
            for row in input_rows:
                row["eval_seed"] = str(args.seeds[index])
        rows.extend(input_rows)
    if not rows:
        raise ValueError("no evaluation rows found")

    accuracies = [float(row["accuracy"]) for row in rows]
    if not all(math.isfinite(value) for value in accuracies):
        raise ValueError("all accuracy values must be finite")
    std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    mean = statistics.fmean(accuracies)
    ci95 = 1.96 * std / math.sqrt(len(accuracies))
    seeds = [row.get("eval_seed", "") for row in rows]
    result = {
        "label": args.label,
        "runs": len(accuracies),
        "mean_accuracy": mean,
        "std_accuracy": std,
        "ci95_accuracy": ci95,
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies),
        "eval_seeds": " ".join(str(seed) for seed in seeds),
        "examples_per_run": rows[0].get("examples", ""),
        "noise_scale": rows[0].get("noise_scale", ""),
        "checkpoint": rows[0].get("checkpoint", ""),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(result)
    print(f"repeated_eval_summary: {args.output}")


if __name__ == "__main__":
    main()
