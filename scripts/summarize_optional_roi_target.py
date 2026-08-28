#!/usr/bin/env python3
"""Summarize target-supervised clean-proposal detection experiments."""

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
SEEDS = (173, 179, 181)
WEIGHTS = (0.025, 0.05)


def read_run(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    results = [
        row
        for row in rows[1:]
        if isinstance(row.get("eval", {}).get("mAP50"), (int, float))
    ]
    if not results:
        raise ValueError(f"no mAP50 result in {path}")
    return rows[0]["metadata"], results[-1]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def control_path(seed: int) -> Path:
    if seed == 173:
        return RUNS_DIR / (
            "optional_detection_roi_target_screen_control_seed173_200tr120ev.jsonl"
        )
    return RUNS_DIR / (
        f"optional_detection_roi_target_repeat_control_seed{seed}_200tr120ev.jsonl"
    )


def winner_path(seed: int, weight: float) -> Path:
    if seed == 173:
        tag = "w0p025" if weight == 0.025 else "w0p05"
        return RUNS_DIR / (
            f"optional_detection_roi_target_screen_target_{tag}_seed173_"
            "200tr120ev.jsonl"
        )
    if weight == 0.05:
        return RUNS_DIR / (
            f"optional_detection_roi_target_repeat_winner_seed{seed}_200tr120ev.jsonl"
        )
    return RUNS_DIR / (
        f"optional_detection_roi_target_repeat_winner_w0p025_seed{seed}_"
        "200tr120ev.jsonl"
    )


def screen_rows() -> list[dict[str, Any]]:
    variants = (
        ("control", 0.0),
        ("target_w0p025", 0.025),
        ("target_w0p05", 0.05),
        ("target_w0p1", 0.1),
    )
    rows = []
    values = {}
    for variant, weight in variants:
        path = RUNS_DIR / (
            f"optional_detection_roi_target_screen_{variant}_seed173_200tr120ev.jsonl"
        )
        metadata, result = read_run(path)
        value = float(result["eval"]["mAP50"]) * 100
        values[variant] = value
        train = result["train"]
        rows.append(
            {
                "variant": variant,
                "weight": weight,
                "seed": metadata["seed"],
                "mAP50_percent": value,
                "delta_pp": 0.0,
                "auxiliary_loss": train.get("proposal_roi_consistency", ""),
                "foreground_proposals": train.get("proposal_roi_foreground", ""),
                "nonfinite": train.get("nonfinite", False),
                "path": str(path.relative_to(PROJECT_ROOT)),
            }
        )
    for row in rows:
        row["delta_pp"] = row["mAP50_percent"] - values["control"]
    return rows


def paired_rows() -> list[dict[str, Any]]:
    rows = []
    for weight in WEIGHTS:
        for seed in SEEDS:
            control_file = control_path(seed)
            winner_file = winner_path(seed, weight)
            _, control_result = read_run(control_file)
            _, winner_result = read_run(winner_file)
            control = float(control_result["eval"]["mAP50"]) * 100
            winner = float(winner_result["eval"]["mAP50"]) * 100
            rows.append(
                {
                    "weight": weight,
                    "seed": seed,
                    "control_percent": control,
                    "winner_percent": winner,
                    "paired_delta_pp": winner - control,
                    "control_path": str(control_file.relative_to(PROJECT_ROOT)),
                    "winner_path": str(winner_file.relative_to(PROJECT_ROOT)),
                }
            )
    return rows


def paired_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for weight in WEIGHTS:
        selected = [row for row in rows if row["weight"] == weight]
        controls = [float(row["control_percent"]) for row in selected]
        winners = [float(row["winner_percent"]) for row in selected]
        deltas = [float(row["paired_delta_pp"]) for row in selected]
        paired = stats.ttest_rel(winners, controls)
        ci95 = stats.t.ppf(0.975, len(deltas) - 1) * stdev(deltas) / math.sqrt(
            len(deltas)
        )
        positive = sum(delta > 0 for delta in deltas)
        summaries.append(
            {
                "weight": weight,
                "runs": len(deltas),
                "control_mean_percent": mean(controls),
                "winner_mean_percent": mean(winners),
                "paired_delta_mean_pp": mean(deltas),
                "paired_delta_std_pp": stdev(deltas),
                "paired_delta_ci95_pp": ci95,
                "paired_t_statistic": float(paired.statistic),
                "paired_p_value": float(paired.pvalue),
                "positive_seeds": positive,
                "decision": (
                    "promote"
                    if positive == len(deltas) and float(paired.pvalue) < 0.05
                    else "reject_unstable"
                ),
            }
        )
    return summaries


def foreground_rows() -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        candidate_path = RUNS_DIR / (
            f"optional_detection_roi_foreground_w0p05_seed{seed}_200tr120ev.jsonl"
        )
        baseline_path = control_path(seed)
        _, baseline_result = read_run(baseline_path)
        _, candidate_result = read_run(candidate_path)
        baseline = float(baseline_result["eval"]["mAP50"]) * 100
        candidate = float(candidate_result["eval"]["mAP50"]) * 100
        rows.append(
            {
                "seed": seed,
                "control_percent": baseline,
                "winner_percent": candidate,
                "paired_delta_pp": candidate - baseline,
                "foreground_proposals": candidate_result["train"].get(
                    "proposal_roi_foreground", ""
                ),
                "auxiliary_loss": candidate_result["train"].get(
                    "proposal_roi_consistency", ""
                ),
                "control_path": str(baseline_path.relative_to(PROJECT_ROOT)),
                "winner_path": str(candidate_path.relative_to(PROJECT_ROOT)),
            }
        )
    return rows


def foreground_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    controls = [float(row["control_percent"]) for row in rows]
    winners = [float(row["winner_percent"]) for row in rows]
    deltas = [float(row["paired_delta_pp"]) for row in rows]
    paired = stats.ttest_rel(winners, controls)
    ci95 = stats.t.ppf(0.975, len(deltas) - 1) * stdev(deltas) / math.sqrt(
        len(deltas)
    )
    positive = sum(delta > 0 for delta in deltas)
    return [
        {
            "objective": "foreground_target",
            "weight": 0.05,
            "runs": len(rows),
            "control_mean_percent": mean(controls),
            "winner_mean_percent": mean(winners),
            "paired_delta_mean_pp": mean(deltas),
            "paired_delta_std_pp": stdev(deltas),
            "paired_delta_ci95_pp": ci95,
            "paired_t_statistic": float(paired.statistic),
            "paired_p_value": float(paired.pvalue),
            "positive_seeds": positive,
            "decision": (
                "promote"
                if positive == len(deltas) and float(paired.pvalue) < 0.05
                else "reject_unstable"
            ),
        }
    ]


def plot_results(
    path: Path,
    screen: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    screen_labels = [str(row["variant"]) for row in screen]
    screen_deltas = [float(row["delta_pp"]) for row in screen]
    axes[0].barh(screen_labels, screen_deltas, color="#2563EB")
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#111827", linewidth=1)
    axes[0].set_title("weight screen, seed 173")
    axes[0].set_xlabel("Delta mAP50 (pp)")
    for axis, weight in zip(axes[1:], WEIGHTS):
        selected = [row for row in paired if row["weight"] == weight]
        for row in selected:
            axis.plot(
                (0, 1),
                (row["control_percent"], row["winner_percent"]),
                marker="o",
                label=f"seed {row['seed']}",
            )
        axis.set_xticks((0, 1), ("control", "candidate"))
        axis.set_title(f"target weight {weight}")
        axis.set_ylabel("mAP50 (%)")
        axis.legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Ground-truth aligned ROI supervision", fontweight="bold")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_foreground(path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(6.0, 4.4))
    for row in rows:
        axis.plot(
            (0, 1),
            (row["control_percent"], row["winner_percent"]),
            marker="o",
            linewidth=1.8,
            label=f"seed {row['seed']} ({row['paired_delta_pp']:+.2f} pp)",
        )
    axis.set_xticks((0, 1), ("matched control", "foreground candidate"))
    axis.set_ylabel("mAP50 (%)")
    axis.set_title("Foreground-only ROI supervision", fontweight="bold")
    axis.legend(frameon=False, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    screen = screen_rows()
    paired = paired_rows()
    summaries = paired_summary(paired)
    foreground = foreground_rows()
    foreground_summaries = foreground_summary(foreground)
    write_csv(RUNS_DIR / "optional_roi_target_screen_summary.csv", screen)
    write_csv(RUNS_DIR / "optional_roi_target_paired_repeats.csv", paired)
    write_csv(RUNS_DIR / "optional_roi_target_paired_summary.csv", summaries)
    write_csv(RUNS_DIR / "optional_roi_foreground_paired_repeats.csv", foreground)
    write_csv(
        RUNS_DIR / "optional_roi_foreground_paired_summary.csv",
        foreground_summaries,
    )
    plot_results(
        FIGURES_DIR / "optional_roi_target_study.png",
        screen,
        paired,
    )
    plot_foreground(
        FIGURES_DIR / "optional_roi_foreground_paired.png",
        foreground,
    )
    print("summary: runs/optional_roi_target_paired_summary.csv")
    print("figure: docs/figures/optional_roi_target_study.png")


if __name__ == "__main__":
    main()
