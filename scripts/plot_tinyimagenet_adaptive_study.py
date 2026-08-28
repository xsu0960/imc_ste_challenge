#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot TinyImageNet adaptive-scale repeat and read-budget results."
    )
    parser.add_argument("--fixed-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--fixed-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--learned-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--learned-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--read-policy-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_accuracy(path: Path) -> float:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}")
    return float(rows[0]["accuracy"])


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def main() -> None:
    args = parse_args()
    if len(args.fixed_inputs) != len(args.fixed_seeds):
        raise ValueError("fixed inputs and seeds must have equal length")
    if len(args.learned_inputs) != len(args.learned_seeds):
        raise ValueError("learned inputs and seeds must have equal length")

    fixed = {
        seed: read_accuracy(path)
        for seed, path in zip(args.fixed_seeds, args.fixed_inputs)
    }
    learned = {
        seed: read_accuracy(path)
        for seed, path in zip(args.learned_seeds, args.learned_inputs)
    }
    with args.read_policy_summary.open(newline="") as handle:
        policy_rows = list(csv.DictReader(handle))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    fixed_values = list(fixed.values())
    learned_values = list(learned.values())
    axes[0].plot(
        list(fixed),
        [100 * value for value in fixed_values],
        marker="o",
        color="#0072B2",
        label=(
            f"Fixed P99: {100 * mean(fixed_values):.2f}% "
            f"+/- {100 * ci95(fixed_values):.2f}"
        ),
    )
    axes[0].plot(
        list(learned),
        [100 * value for value in learned_values],
        marker="s",
        color="#D55E00",
        label=(
            f"Learned bounded: {100 * mean(learned_values):.2f}% "
            f"+/- {100 * ci95(learned_values):.2f}"
        ),
    )
    axes[0].set_xlabel("Noisy evaluation seed")
    axes[0].set_ylabel("Full-validation top-1 (%)")
    axes[0].set_title("Strict-noise repeat stability")
    axes[0].set_xticks(sorted(set(fixed) | set(learned)))
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    label_map = {
        "uniform_read8": "Uniform 8",
        "compensated_base1_max8": "Adaptive 1-8",
        "compensated_base2_max8": "Adaptive 2-8",
        "compensated_base4_max8": "Adaptive 4-8",
        "compensated_base2_max16": "Adaptive 2-16",
    }
    for row in policy_rows:
        policy = row["policy"]
        x = float(row["mean_sensitive_layer_reads"])
        y = 100 * float(row["accuracy"])
        highlighted = policy in {"uniform_read8", "compensated_base4_max8"}
        axes[1].scatter(
            x,
            y,
            s=70 if highlighted else 45,
            color="#009E73" if highlighted else "#999999",
            zorder=3,
        )
        axes[1].annotate(
            label_map.get(policy, policy),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set_xlabel("Mean reads per sensitive layer")
    axes[1].set_ylabel("20-batch top-1 (%)")
    axes[1].set_title("Output-noise read-budget trade-off")
    axes[1].grid(alpha=0.25)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
