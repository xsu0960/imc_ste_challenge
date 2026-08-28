#!/usr/bin/env python3
"""Summarize corrected shared-read head-aware screening results."""

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
CANDIDATES = ("control", "role_gradient", "task_consistency", "head_range")
DISPLAY_NAMES = {
    "control": "sat-aware control",
    "role_gradient": "head statistic STE",
    "task_consistency": "task-output consistency",
    "head_range": "head range control",
}
COLORS = {
    "control": "#4B5563",
    "role_gradient": "#2563EB",
    "task_consistency": "#059669",
    "head_range": "#D97706",
}


def read_run(path: Path, metric: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    metadata = rows[0]["metadata"]
    result_rows = [
        row for row in rows[1:] if isinstance(row.get("eval", {}).get(metric), (float, int))
    ]
    if not result_rows:
        raise ValueError(f"no {metric} result in {path}")
    return metadata, result_rows[-1]


def build_rows(seed: int, train_batches: int, eval_batches: int) -> list[dict[str, Any]]:
    rows = []
    for task, metric in (("detection", "mAP50"), ("segmentation", "mIoU")):
        for candidate in CANDIDATES:
            filename = (
                f"optional_{task}_shared_read_head_screen_{candidate}_seed{seed}_"
                f"{train_batches}tr{eval_batches}ev.jsonl"
            )
            path = RUNS_DIR / filename
            metadata, result = read_run(path, metric)
            value = float(result["eval"][metric])
            rows.append(
                {
                    "task": task,
                    "stage": "screen",
                    "candidate": candidate,
                    "display_name": DISPLAY_NAMES[candidate],
                    "metric": metric,
                    "seed": metadata["seed"],
                    "train_mode": metadata["train_mode"],
                    "noise_scale": metadata["noise_scale"],
                    "conv_weight_noise_scope": metadata["conv_weight_noise_scope"],
                    "train_examples": result["train"]["examples"],
                    "eval_examples": result["eval"]["examples"],
                    "value": value,
                    "value_percent": value * 100,
                    "delta_pp": 0.0,
                    "nonfinite": result["train"].get("nonfinite", False),
                    "task_output_consistency": result["train"].get(
                        "task_output_consistency", ""
                    ),
                    "path": str(path.relative_to(PROJECT_ROOT)),
                }
            )
    controls = {
        row["task"]: row["value"] for row in rows if row["candidate"] == "control"
    }
    for row in rows:
        row["delta_pp"] = (row["value"] - controls[row["task"]]) * 100
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def annotate(axis: plt.Axes, bars: Any, values: list[float], *, delta: bool) -> None:
    for bar, value in zip(bars, values):
        if delta and abs(value) < 0.1:
            axis.annotate(
                f"{value:+.2f}",
                (0, bar.get_y() + bar.get_height() / 2),
                xytext=(7, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
            )
            continue
        offset = 4
        axis.annotate(
            f"{value:+.2f}" if delta else f"{value:.2f}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(offset, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color="white" if value < 0 else "#111827",
        )


def write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, task in zip(axes, ("detection", "segmentation")):
        task_rows = [row for row in rows if row["task"] == task]
        labels = [row["display_name"] for row in task_rows]
        values = [float(row["delta_pp"]) for row in task_rows]
        bars = axis.barh(
            range(len(task_rows)),
            values,
            color=[COLORS[row["candidate"]] for row in task_rows],
        )
        axis.set_yticks(range(len(labels)), labels)
        axis.invert_yaxis()
        axis.axvline(0, color="#111827", linewidth=1)
        axis.set_xlabel("Delta vs matched control (pp)")
        axis.set_title("Detection mAP50" if task == "detection" else "Segmentation mIoU")
        axis.spines[["top", "right"]].set_visible(False)
        annotate(axis, bars, values, delta=True)
    figure.suptitle("Shared-read head-aware screening", fontweight="bold")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    seed = 131
    train_batches = 300
    eval_batches = 200
    rows = build_rows(seed, train_batches, eval_batches)
    output = RUNS_DIR / "optional_head_aware_screen_summary.csv"
    figure = PROJECT_ROOT / "docs/figures/optional_head_aware_screen.png"
    write_csv(output, rows)
    write_figure(figure, rows)
    print(f"summary: {output.relative_to(PROJECT_ROOT)}")
    print(f"figure: {figure.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
