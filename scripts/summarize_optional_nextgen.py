#!/usr/bin/env python3
"""Summarize online-profile and proposal-aligned next-generation studies."""

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
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"

SCREEN_CANDIDATES = (
    "dualpass_control",
    "online_bias",
    "online_soft",
    "online_balanced",
    "online_staticmatch",
)
REFINE_FILES = {
    "detection_online": {
        "control": "optional_detection_online_profile_screen_dualpass_control_seed151_300tr200ev.jsonl",
        "online soft": "optional_detection_online_profile_screen_online_soft_seed151_300tr200ev.jsonl",
        "variance only": "optional_detection_nextgen_refine_online_variance_all_seed151_300tr200ev.jsonl",
        "soft floor 0.85": "optional_detection_nextgen_refine_online_soft_floor85_seed151_300tr200ev.jsonl",
        "soft shared conv": "optional_detection_nextgen_refine_online_soft_shared_conv_seed151_300tr200ev.jsonl",
    },
    "detection_roi": {
        "control": "optional_detection_nextgen_refine_roi_control_seed151_300tr200ev.jsonl",
        "ROI weight 0.0025": "optional_detection_nextgen_refine_roi_w0p0025_seed151_300tr200ev.jsonl",
        "ROI weight 0.005": "optional_detection_nextgen_refine_roi_w0p005_seed151_300tr200ev.jsonl",
        "ROI weight 0.01": "optional_detection_nextgen_refine_roi_w0p01_seed151_300tr200ev.jsonl",
    },
    "segmentation_combo": {
        "control": "optional_segmentation_online_profile_screen_dualpass_control_seed151_300tr200ev.jsonl",
        "online static match": "optional_segmentation_online_profile_screen_online_staticmatch_seed151_300tr200ev.jsonl",
        "online + teacher + range": "optional_segmentation_nextgen_refine_online_static_task_range_seed151_300tr200ev.jsonl",
        "output heads only": "optional_segmentation_nextgen_refine_online_output_heads_seed151_300tr200ev.jsonl",
        "exclude global pool": "optional_segmentation_nextgen_refine_online_no_global_pool_seed151_300tr200ev.jsonl",
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


def screen_rows() -> list[dict[str, Any]]:
    rows = []
    for task, metric in (("detection", "mAP50"), ("segmentation", "mIoU")):
        values = {}
        for candidate in SCREEN_CANDIDATES:
            path = RUNS_DIR / (
                f"optional_{task}_online_profile_screen_{candidate}_seed151_"
                "300tr200ev.jsonl"
            )
            metadata, result = read_run(path, metric)
            value = float(result["eval"][metric]) * 100
            values[candidate] = value
            profile = result["train"].get("online_gradient_profile", {})
            rows.append(
                {
                    "task": task,
                    "candidate": candidate,
                    "metric": metric,
                    "seed": metadata["seed"],
                    "value_percent": value,
                    "delta_pp": 0.0,
                    "gradient_scale_mean": profile.get("scale_mean", ""),
                    "gradient_scale_floor_fraction": profile.get("floor_fraction", ""),
                    "stochastic_ratio_mean": profile.get("stochastic_ratio_mean", ""),
                    "bias_ratio_mean": profile.get("bias_ratio_mean", ""),
                    "nonfinite": result["train"].get("nonfinite", False),
                    "path": str(path.relative_to(PROJECT_ROOT)),
                }
            )
        for row in rows:
            if row["task"] == task:
                row["delta_pp"] = row["value_percent"] - values["dualpass_control"]
    return rows


def refine_rows() -> list[dict[str, Any]]:
    rows = []
    for family, variants in REFINE_FILES.items():
        task = "segmentation" if family.startswith("segmentation") else "detection"
        metric = "mIoU" if task == "segmentation" else "mAP50"
        values = {}
        family_rows = []
        for variant, filename in variants.items():
            path = RUNS_DIR / filename
            metadata, result = read_run(path, metric)
            value = float(result["eval"][metric]) * 100
            values[variant] = value
            family_rows.append(
                {
                    "family": family,
                    "task": task,
                    "variant": variant,
                    "metric": metric,
                    "seed": metadata["seed"],
                    "value_percent": value,
                    "delta_pp": 0.0,
                    "nonfinite": result["train"].get("nonfinite", False),
                    "path": str(path.relative_to(PROJECT_ROOT)),
                }
            )
        for row in family_rows:
            row["delta_pp"] = row["value_percent"] - values["control"]
        rows.extend(family_rows)
    return rows


def repeat_path(family: str, variant: str, seed: int) -> Path:
    if seed == 151:
        filename = {
            ("detection_online", "control"): "optional_detection_online_profile_screen_dualpass_control_seed151_300tr200ev.jsonl",
            ("detection_online", "winner"): "optional_detection_online_profile_screen_online_soft_seed151_300tr200ev.jsonl",
            ("detection_roi", "control"): "optional_detection_nextgen_refine_roi_control_seed151_300tr200ev.jsonl",
            ("detection_roi", "winner"): "optional_detection_nextgen_refine_roi_w0p0025_seed151_300tr200ev.jsonl",
            ("segmentation_combo", "control"): "optional_segmentation_online_profile_screen_dualpass_control_seed151_300tr200ev.jsonl",
            ("segmentation_combo", "winner"): "optional_segmentation_nextgen_refine_online_static_task_range_seed151_300tr200ev.jsonl",
        }[(family, variant)]
    else:
        filename = (
            f"optional_{family}_nextgen_repeat_{variant}_seed{seed}_300tr200ev.jsonl"
        )
    return RUNS_DIR / filename


def repeat_rows() -> list[dict[str, Any]]:
    rows = []
    for family in ("detection_online", "detection_roi", "segmentation_combo"):
        task = "segmentation" if family.startswith("segmentation") else "detection"
        metric = "mIoU" if task == "segmentation" else "mAP50"
        for seed in (151, 157, 163):
            values = {}
            seed_rows = []
            for variant in ("control", "winner"):
                path = repeat_path(family, variant, seed)
                metadata, result = read_run(path, metric)
                value = float(result["eval"][metric]) * 100
                values[variant] = value
                seed_rows.append(
                    {
                        "family": family,
                        "task": task,
                        "seed": metadata["seed"],
                        "variant": variant,
                        "metric": metric,
                        "value_percent": value,
                        "paired_delta_pp": 0.0,
                        "nonfinite": result["train"].get("nonfinite", False),
                        "path": str(path.relative_to(PROJECT_ROOT)),
                    }
                )
            delta = values["winner"] - values["control"]
            for row in seed_rows:
                row["paired_delta_pp"] = delta
                rows.append(row)
    return rows


def repeat_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for family in ("detection_online", "detection_roi", "segmentation_combo"):
        family_rows = [row for row in rows if row["family"] == family]
        controls = [
            float(row["value_percent"])
            for row in family_rows
            if row["variant"] == "control"
        ]
        winners = [
            float(row["value_percent"])
            for row in family_rows
            if row["variant"] == "winner"
        ]
        deltas = [winner - control for control, winner in zip(controls, winners)]
        paired = stats.ttest_rel(winners, controls)
        t_critical = stats.t.ppf(0.975, len(deltas) - 1)
        ci95 = t_critical * stdev(deltas) / math.sqrt(len(deltas))
        positive = sum(delta > 0 for delta in deltas)
        promoted = bool(
            mean(deltas) > 0
            and positive == len(deltas)
            and float(paired.pvalue) < 0.05
        )
        summaries.append(
            {
                "family": family,
                "runs": len(deltas),
                "control_mean_percent": mean(controls),
                "winner_mean_percent": mean(winners),
                "paired_delta_mean_pp": mean(deltas),
                "paired_delta_std_pp": stdev(deltas),
                "paired_delta_ci95_pp": ci95,
                "paired_t_statistic": float(paired.statistic),
                "paired_p_value": float(paired.pvalue),
                "positive_seeds": positive,
                "promotion_decision": (
                    "promote_to_full_epoch" if promoted else "reject_unstable"
                ),
            }
        )
    return summaries


def plot_deltas(path: Path, rows: list[dict[str, Any]], title: str, group_key: str) -> None:
    groups = list(dict.fromkeys(str(row[group_key]) for row in rows))
    figure, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 4.8))
    if len(groups) == 1:
        axes = [axes]
    for axis, group in zip(axes, groups):
        group_rows = [row for row in rows if row[group_key] == group]
        labels = [str(row.get("variant", row.get("candidate"))) for row in group_rows]
        values = [float(row["delta_pp"]) for row in group_rows]
        colors = ["#4B5563" if abs(value) < 1e-12 else "#2563EB" for value in values]
        bars = axis.barh(range(len(values)), values, color=colors)
        axis.set_yticks(range(len(labels)), labels)
        axis.invert_yaxis()
        axis.axvline(0, color="#111827", linewidth=1)
        axis.set_xlabel("Delta vs matched control (pp)")
        axis.set_title(group.replace("_", " "))
        axis.spines[["top", "right"]].set_visible(False)
        extent = max((abs(value) for value in values), default=1.0)
        padding = max(0.08, 0.18 * extent)
        axis.set_xlim(
            min(0.0, min(values, default=0.0)) - padding,
            max(0.0, max(values, default=0.0)) + padding,
        )
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:+.2f}",
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(4 if value >= 0 else -4, 0),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=8,
            )
    figure.suptitle(title, fontweight="bold")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_repeats(path: Path, rows: list[dict[str, Any]]) -> None:
    families = ("detection_online", "detection_roi", "segmentation_combo")
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for axis, family in zip(axes, families):
        family_rows = [row for row in rows if row["family"] == family]
        for seed in (151, 157, 163):
            seed_rows = [row for row in family_rows if row["seed"] == seed]
            values = [
                next(
                    float(row["value_percent"])
                    for row in seed_rows
                    if row["variant"] == variant
                )
                for variant in ("control", "winner")
            ]
            axis.plot((0, 1), values, marker="o", linewidth=1.8, label=f"seed {seed}")
        axis.set_xticks((0, 1), ("control", "candidate"))
        axis.set_ylabel("Metric (%)")
        axis.set_title(family.replace("_", " "))
        axis.legend(frameon=False, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Next-generation paired repeats", fontweight="bold")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def full_epoch_rows() -> list[dict[str, Any]]:
    rows = []
    values = {}
    for variant in ("control", "winner"):
        path = RUNS_DIR / (
            f"optional_segmentation_nextgen_full_epoch_{variant}_seed167.jsonl"
        )
        metadata, result = read_run(path, "mIoU")
        value = float(result["eval"]["mIoU"]) * 100
        values[variant] = value
        profile = result["train"].get("online_gradient_profile", {})
        rows.append(
            {
                "variant": variant,
                "seed": metadata["seed"],
                "train_examples": result["train"]["examples"],
                "eval_examples": result["eval"]["examples"],
                "mIoU_percent": value,
                "paired_delta_pp": 0.0,
                "gradient_scale_mean": profile.get("scale_mean", ""),
                "gradient_scale_floor_fraction": profile.get("floor_fraction", ""),
                "shape_mismatches": profile.get("shape_mismatches", ""),
                "nonfinite": result["train"].get("nonfinite", False),
                "path": str(path.relative_to(PROJECT_ROOT)),
            }
        )
    delta = values["winner"] - values["control"]
    for row in rows:
        row["paired_delta_pp"] = delta
    return rows


def plot_full_epoch(path: Path, rows: list[dict[str, Any]]) -> None:
    values = [float(row["mIoU_percent"]) for row in rows]
    figure, axis = plt.subplots(figsize=(5.4, 4.4))
    axis.plot((0, 1), values, marker="o", linewidth=2.2, color="#2563EB")
    axis.set_xticks((0, 1), ("matched control", "candidate"))
    axis.set_ylabel("mIoU (%)")
    axis.set_title("Full-epoch segmentation validation", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    delta = values[1] - values[0]
    axis.annotate(
        f"{delta:+.3f} pp",
        (1, values[1]),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    screen = screen_rows()
    refine = refine_rows()
    repeats = repeat_rows()
    summaries = repeat_summary(repeats)
    full_epoch = full_epoch_rows()
    write_csv(RUNS_DIR / "optional_nextgen_screen_summary.csv", screen)
    write_csv(RUNS_DIR / "optional_nextgen_refine_summary.csv", refine)
    write_csv(RUNS_DIR / "optional_nextgen_paired_repeats.csv", repeats)
    write_csv(RUNS_DIR / "optional_nextgen_paired_summary.csv", summaries)
    write_csv(RUNS_DIR / "optional_nextgen_full_epoch.csv", full_epoch)
    plot_deltas(
        FIGURES_DIR / "optional_nextgen_screen.png",
        screen,
        "Online gradient profile screen",
        "task",
    )
    plot_deltas(
        FIGURES_DIR / "optional_nextgen_refine.png",
        refine,
        "Next-generation refinement",
        "family",
    )
    plot_repeats(FIGURES_DIR / "optional_nextgen_paired_repeats.png", repeats)
    plot_full_epoch(
        FIGURES_DIR / "optional_nextgen_full_epoch.png",
        full_epoch,
    )
    print("summary: runs/optional_nextgen_paired_summary.csv")
    print("figure: docs/figures/optional_nextgen_paired_repeats.png")


if __name__ == "__main__":
    main()
