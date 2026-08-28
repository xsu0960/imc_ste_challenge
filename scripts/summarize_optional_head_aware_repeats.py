#!/usr/bin/env python3
"""Summarize refined candidates and paired optional multi-seed repeats."""

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


REFINE_FILES = {
    "detection": {
        "control": "optional_detection_shared_read_head_screen_control_seed131_300tr200ev.jsonl",
        "gradient v0.25 b0.25": "optional_detection_shared_read_head_refine_gradient_v025_b025_seed131_300tr200ev.jsonl",
        "gradient v0.5 b0": "optional_detection_shared_read_head_refine_gradient_v05_b0_seed131_300tr200ev.jsonl",
        "gradient v0.5 b0.25": "optional_detection_shared_read_head_screen_role_gradient_seed131_300tr200ev.jsonl",
        "gradient v1 b0.25": "optional_detection_shared_read_head_refine_gradient_v1_b025_seed131_300tr200ev.jsonl",
    },
    "segmentation": {
        "control": "optional_segmentation_shared_read_head_screen_control_seed131_300tr200ev.jsonl",
        "task w0.005": "optional_segmentation_shared_read_head_refine_task_w0005_seed131_300tr200ev.jsonl",
        "task w0.01": "optional_segmentation_shared_read_head_refine_task_w001_seed131_300tr200ev.jsonl",
        "task w0.02": "optional_segmentation_shared_read_head_screen_task_consistency_seed131_300tr200ev.jsonl",
        "task w0.05": "optional_segmentation_shared_read_head_refine_task_w005_seed131_300tr200ev.jsonl",
        "range t10": "optional_segmentation_shared_read_head_screen_head_range_seed131_300tr200ev.jsonl",
        "range t12": "optional_segmentation_shared_read_head_refine_range_t12_seed131_300tr200ev.jsonl",
        "task w0.02 + range t10": "optional_segmentation_shared_read_head_refine_task_w002_range_t10_seed131_300tr200ev.jsonl",
        "task w0.005 + range t10": "optional_segmentation_shared_read_head_refine_task_w0005_range_t10_seed131_300tr200ev.jsonl",
    },
}


def read_run(path: Path, metric: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    metadata = rows[0]["metadata"]
    results = [
        row for row in rows[1:] if isinstance(row.get("eval", {}).get(metric), (int, float))
    ]
    if not results:
        raise ValueError(f"no {metric} result in {path}")
    return metadata, results[-1]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_refine_rows() -> list[dict[str, Any]]:
    rows = []
    for task, variants in REFINE_FILES.items():
        metric = "mAP50" if task == "detection" else "mIoU"
        values = {}
        for variant, filename in variants.items():
            path = RUNS_DIR / filename
            metadata, result = read_run(path, metric)
            value = float(result["eval"][metric])
            values[variant] = value
            rows.append(
                {
                    "task": task,
                    "variant": variant,
                    "metric": metric,
                    "seed": metadata["seed"],
                    "value": value,
                    "value_percent": value * 100,
                    "delta_pp": 0.0,
                    "nonfinite": result["train"].get("nonfinite", False),
                    "path": str(path.relative_to(PROJECT_ROOT)),
                }
            )
        for row in rows:
            if row["task"] == task:
                row["delta_pp"] = (row["value"] - values["control"]) * 100
    return rows


def repeat_path(task: str, variant: str, seed: int) -> Path:
    if seed == 131:
        filename = {
            ("detection", "control"): "optional_detection_shared_read_head_screen_control_seed131_300tr200ev.jsonl",
            ("detection", "winner"): "optional_detection_shared_read_head_screen_role_gradient_seed131_300tr200ev.jsonl",
            ("segmentation", "control"): "optional_segmentation_shared_read_head_screen_control_seed131_300tr200ev.jsonl",
            ("segmentation", "winner"): "optional_segmentation_shared_read_head_refine_task_w0005_range_t10_seed131_300tr200ev.jsonl",
        }[(task, variant)]
    else:
        filename = (
            f"optional_{task}_shared_read_head_repeat_{variant}_seed{seed}_"
            "300tr200ev.jsonl"
        )
    return RUNS_DIR / filename


def build_repeat_rows() -> list[dict[str, Any]]:
    rows = []
    for task in ("detection", "segmentation"):
        metric = "mAP50" if task == "detection" else "mIoU"
        for seed in (131, 137, 139):
            seed_values = {}
            seed_rows = []
            for variant in ("control", "winner"):
                path = repeat_path(task, variant, seed)
                metadata, result = read_run(path, metric)
                value = float(result["eval"][metric])
                seed_values[variant] = value
                seed_rows.append(
                    {
                        "task": task,
                        "seed": seed,
                        "variant": variant,
                        "metric": metric,
                        "train_mode": metadata["train_mode"],
                        "value": value,
                        "value_percent": value * 100,
                        "paired_delta_pp": 0.0,
                        "nonfinite": result["train"].get("nonfinite", False),
                        "path": str(path.relative_to(PROJECT_ROOT)),
                    }
                )
            delta = (seed_values["winner"] - seed_values["control"]) * 100
            for row in seed_rows:
                row["paired_delta_pp"] = delta
                rows.append(row)
    return rows


def build_repeat_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for task in ("detection", "segmentation"):
        task_rows = [row for row in rows if row["task"] == task]
        controls = [
            float(row["value_percent"])
            for row in task_rows
            if row["variant"] == "control"
        ]
        winners = [
            float(row["value_percent"])
            for row in task_rows
            if row["variant"] == "winner"
        ]
        deltas = [winner - control for control, winner in zip(controls, winners)]
        test = stats.ttest_rel(winners, controls)
        t_critical = stats.t.ppf(0.975, len(deltas) - 1)
        ci95 = t_critical * stdev(deltas) / math.sqrt(len(deltas))
        positive_seeds = sum(delta > 0 for delta in deltas)
        promoted = bool(
            mean(deltas) > 0
            and positive_seeds == len(deltas)
            and float(test.pvalue) < 0.05
        )
        summary.append(
            {
                "task": task,
                "runs": len(deltas),
                "control_mean_percent": mean(controls),
                "winner_mean_percent": mean(winners),
                "paired_delta_mean_pp": mean(deltas),
                "paired_delta_std_pp": stdev(deltas),
                "paired_delta_ci95_pp": ci95,
                "paired_t_statistic": float(test.statistic),
                "paired_p_value": float(test.pvalue),
                "positive_seeds": positive_seeds,
                "promoted_to_full_epoch": promoted,
                "promotion_decision": (
                    "promote_to_full_epoch" if promoted else "reject_unstable"
                ),
            }
        )
    return summary


def write_repeat_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, task in zip(axes, ("detection", "segmentation")):
        task_rows = [row for row in rows if row["task"] == task]
        for seed in (131, 137, 139):
            seed_rows = [row for row in task_rows if row["seed"] == seed]
            values = [
                next(
                    float(row["value_percent"])
                    for row in seed_rows
                    if row["variant"] == variant
                )
                for variant in ("control", "winner")
            ]
            axis.plot((0, 1), values, marker="o", linewidth=1.8, label=f"seed {seed}")
        axis.set_xticks((0, 1), ("control", "winner"))
        axis.set_ylabel("Metric (%)")
        axis.set_title("Detection mAP50" if task == "detection" else "Segmentation mIoU")
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Paired shared-read head-aware repeats", fontweight="bold")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    refine_rows = build_refine_rows()
    repeat_rows = build_repeat_rows()
    repeat_summary = build_repeat_summary(repeat_rows)
    write_csv(RUNS_DIR / "optional_head_aware_refine_summary.csv", refine_rows)
    write_csv(RUNS_DIR / "optional_head_aware_paired_repeats.csv", repeat_rows)
    write_csv(RUNS_DIR / "optional_head_aware_paired_summary.csv", repeat_summary)
    figure = PROJECT_ROOT / "docs/figures/optional_head_aware_paired_repeats.png"
    write_repeat_figure(figure, repeat_rows)
    print("summary: runs/optional_head_aware_paired_summary.csv")
    print(f"figure: {figure.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
