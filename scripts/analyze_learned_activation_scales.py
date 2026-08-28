#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export and plot fixed versus learned activation scales."
    )
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target", type=float, default=4.0)
    parser.add_argument("--scale-min", type=float, default=0.1)
    parser.add_argument("--scale-max", type=float, default=1.0)
    parser.add_argument(
        "--kinds", nargs="+", default=["depthwise", "pointwise"]
    )
    parser.add_argument("--read-base", type=int, default=4)
    parser.add_argument("--read-max", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    return parser.parse_args()


def allocated_reads(scale: float, base: int, maximum: int) -> int:
    return min(maximum, max(base, math.ceil(base / scale**2 - 1e-5)))


def main() -> None:
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu")
    with args.statistics.open(newline="") as handle:
        statistics_rows = list(csv.DictReader(handle))

    output_rows = []
    for row in statistics_rows:
        if row.get("kind") not in set(args.kinds):
            continue
        observed = float(row["clean_abs_p99_mean"])
        initial = max(
            args.scale_min,
            min(1.0, 1.0 if observed == 0 else args.target / observed),
        )
        key = f"{row['name']}.activation_scale_logit"
        enabled = key in state
        if enabled:
            normalized = float(torch.sigmoid(state[key]).item())
            learned = args.scale_min + (
                args.scale_max - args.scale_min
            ) * normalized
        else:
            learned = initial
        output_rows.append(
            {
                "name": row["name"],
                "kind": row["kind"],
                "clean_abs_p99_mean": observed,
                "initial_scale": initial,
                "learned_scale": learned,
                "scale_delta": learned - initial,
                "learnable": enabled,
                "initial_reads": allocated_reads(
                    initial, args.read_base, args.read_max
                ),
                "learned_reads": allocated_reads(
                    learned, args.read_base, args.read_max
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    enabled_rows = [row for row in output_rows if row["learnable"]]
    colors = {"depthwise": "#0072B2", "pointwise": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for kind in args.kinds:
        kind_rows = [row for row in enabled_rows if row["kind"] == kind]
        axes[0].scatter(
            [row["initial_scale"] for row in kind_rows],
            [row["learned_scale"] for row in kind_rows],
            label=kind,
            color=colors.get(kind),
            alpha=0.8,
            s=28,
        )
    axes[0].plot([0.1, 1.0], [0.1, 1.0], color="#444444", linestyle="--")
    axes[0].set_xlabel("Statistics-initialized scale")
    axes[0].set_ylabel("Learned scale")
    axes[0].set_title("Bounded layer-scale adaptation")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    deltas = [100 * row["scale_delta"] for row in enabled_rows]
    axes[1].hist(deltas, bins=16, color="#009E73", edgecolor="white")
    axes[1].axvline(0, color="#444444", linestyle="--")
    axes[1].set_xlabel("Scale change (percentage points)")
    axes[1].set_ylabel("Layer count")
    axes[1].set_title("Regularized scale updates")
    axes[1].grid(axis="y", alpha=0.25)

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=200)
    plt.close(fig)

    mean_delta = sum(row["scale_delta"] for row in enabled_rows) / len(enabled_rows)
    print(
        f"layers={len(enabled_rows)} mean_delta={mean_delta:.6f} "
        f"csv={args.output} figure={args.figure}"
    )


if __name__ == "__main__":
    main()
