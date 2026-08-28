#!/usr/bin/env python3
"""Plot release-facing full-validation and paired extended-task results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-validation",
        type=Path,
        default=PROJECT_ROOT / "runs" / "optional_full_validation_summary.csv",
    )
    parser.add_argument(
        "--paired",
        type=Path,
        default=PROJECT_ROOT / "runs" / "optional_paired_extension_summary.csv",
    )
    parser.add_argument(
        "--full-validation-figure",
        type=Path,
        default=PROJECT_ROOT / "docs" / "figures" / "optional_full_validation.png",
    )
    parser.add_argument(
        "--paired-figure",
        type=Path,
        default=PROJECT_ROOT / "docs" / "figures" / "optional_paired_extension.png",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_full_validation(path: Path, rows: list[dict[str, str]]) -> None:
    labels = ["Detection\nmAP50", "Segmentation\nmIoU"]
    clean = [float(row["clean_percent"]) for row in rows]
    direct = [float(row["direct_noisy_percent"]) for row in rows]
    ste = [float(row["ste_percent"]) for row in rows]
    positions = list(range(len(rows)))
    width = 0.24

    figure, axis = plt.subplots(figsize=(8.0, 4.6))
    series = (
        ("Clean", clean, -width, "#4C78A8"),
        ("Direct noisy", direct, 0.0, "#E45756"),
        ("1-epoch STE", ste, width, "#54A24B"),
    )
    for name, values, offset, color in series:
        bars = axis.bar(
            [position + offset for position in positions],
            values,
            width,
            label=name,
            color=color,
        )
        axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Metric (%)")
    axis.set_ylim(0, 80)
    axis.set_title("Shared-read full-validation under uniform noise", fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_paired(path: Path, rows: list[dict[str, str]]) -> None:
    labels = ["Detection\nonline profile", "Detection\nproposal aligned", "Segmentation\ncombined"]
    deltas = [float(row["paired_delta_mean_pp"]) for row in rows]
    intervals = [float(row["paired_delta_ci95_pp"]) for row in rows]
    colors = [
        "#54A24B" if row["conclusion"] == "significant" else "#9CA3AF"
        for row in rows
    ]

    figure, axis = plt.subplots(figsize=(8.0, 4.6))
    positions = list(range(len(rows)))
    for position, delta, interval, color in zip(
        positions, deltas, intervals, colors
    ):
        axis.errorbar(
            position,
            delta,
            yerr=interval,
            fmt="o",
            color=color,
            markersize=7,
            elinewidth=2.0,
            capsize=5,
            capthick=1.5,
            zorder=3,
        )
    axis.axhline(0.0, color="#374151", linewidth=1.0)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Paired improvement (pp)")
    axis.set_title("Paired extended-task evaluation (95% CI)", fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    for position, row in enumerate(rows):
        axis.annotate(
            f"p={float(row['paired_p_value']):.3g}",
            (position, deltas[position]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    plt.style.use("seaborn-v0_8-whitegrid")
    full_rows = read_rows(args.full_validation)
    paired_rows = read_rows(args.paired)
    if len(full_rows) != 2 or len(paired_rows) != 3:
        raise ValueError("release optional summaries have unexpected row counts")
    plot_full_validation(args.full_validation_figure, full_rows)
    plot_paired(args.paired_figure, paired_rows)
    print(f"figure: {args.full_validation_figure}")
    print(f"figure: {args.paired_figure}")


if __name__ == "__main__":
    main()
