#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


EXPERIMENTS = [
    {
        "task": "detection",
        "group": "full_val_seed1",
        "protocol": "clean",
        "path": "optional_detection_internal_clean_eval_full.jsonl",
        "metric": "mAP50",
    },
    {
        "task": "detection",
        "group": "full_val_seed1",
        "protocol": "direct noisy",
        "path": "optional_detection_internal_noisy_eval_full.jsonl",
        "metric": "mAP50",
    },
    {
        "task": "detection",
        "group": "full_val_seed1",
        "protocol": "1-epoch sat-aware STE",
        "path": "optional_detection_internal_ste_full_1epoch.jsonl",
        "metric": "mAP50",
    },
    {
        "task": "detection",
        "group": "full_val_seed1",
        "protocol": "+300 batch continuation",
        "path": "optional_detection_ste1ep_plus300_sat_aware_fullval_noise_seed1.jsonl",
        "metric": "mAP50",
        "reference": "1-epoch sat-aware STE",
    },
    {
        "task": "segmentation",
        "group": "full_val_seed1",
        "protocol": "clean",
        "path": "optional_segmentation_internal_clean_eval_full.jsonl",
        "metric": "mIoU",
    },
    {
        "task": "segmentation",
        "group": "full_val_seed1",
        "protocol": "direct noisy",
        "path": "optional_segmentation_internal_noisy_eval_full.jsonl",
        "metric": "mIoU",
    },
    {
        "task": "segmentation",
        "group": "full_val_seed1",
        "protocol": "1-epoch sat-aware STE",
        "path": "optional_segmentation_internal_ste_full_1epoch.jsonl",
        "metric": "mIoU",
    },
    {
        "task": "detection",
        "group": "posthoc_200_seed43",
        "protocol": "baseline",
        "path": "optional_detection_ste1ep_baseline_noise_seed43_200ev.jsonl",
        "metric": "mAP50",
    },
    {
        "task": "detection",
        "group": "posthoc_200_seed43",
        "protocol": "activation target 4",
        "path": "optional_detection_ste1ep_activation_target4_noise_seed43_200ev.jsonl",
        "metric": "mAP50",
        "reference": "baseline",
    },
    {
        "task": "detection",
        "group": "posthoc_200_seed43",
        "protocol": "activation target 6",
        "path": "optional_detection_ste1ep_activation_target6_noise_seed43_200ev.jsonl",
        "metric": "mAP50",
        "reference": "baseline",
    },
    {
        "task": "detection",
        "group": "posthoc_200_seed43",
        "protocol": "activation target 8",
        "path": "optional_detection_ste1ep_activation_target8_noise_seed43_200ev.jsonl",
        "metric": "mAP50",
        "reference": "baseline",
    },
    {
        "task": "detection",
        "group": "posthoc_200_seed43",
        "protocol": "ROI read 4",
        "path": "optional_detection_ste1ep_roi_read4_noise_seed43_200ev.jsonl",
        "metric": "mAP50",
        "reference": "baseline",
    },
    {
        "task": "detection",
        "group": "posthoc_200_seed43",
        "protocol": "RPN/ROI target 6 + adaptive read 8",
        "path": "optional_detection_ste1ep_rpn_roi_target6_adaptive_read8_noise_seed43_200ev.jsonl",
        "metric": "mAP50",
        "reference": "baseline",
    },
    {
        "task": "detection",
        "group": "train_300_eval_200_seed47",
        "protocol": "sat-aware control",
        "path": "optional_detection_ste1ep_continue_sat_aware_300tr200ev_seed47.jsonl",
        "metric": "mAP50",
    },
    {
        "task": "detection",
        "group": "train_300_eval_200_seed47",
        "protocol": "variance sat-aware",
        "path": "optional_detection_ste1ep_continue_variance_sat_aware_300tr200ev_seed47.jsonl",
        "metric": "mAP50",
        "reference": "sat-aware control",
    },
    {
        "task": "detection",
        "group": "train_300_eval_200_seed47",
        "protocol": "adaptive sat-aware",
        "path": "optional_detection_ste1ep_continue_adaptive_sat_aware_300tr200ev_seed47.jsonl",
        "metric": "mAP50",
        "reference": "sat-aware control",
    },
    {
        "task": "detection",
        "group": "train_300_eval_200_seed47",
        "protocol": "FPN clean consistency",
        "path": "optional_detection_ste1ep_continue_fpn_clean_consistency_w0p02_300tr200ev_seed47.jsonl",
        "metric": "mAP50",
        "reference": "sat-aware control",
    },
    {
        "task": "segmentation",
        "group": "train_300_eval_200_seed47",
        "protocol": "sat-aware control",
        "path": "optional_segmentation_ste1ep_continue_sat_aware_300tr200ev_seed47.jsonl",
        "metric": "mIoU",
    },
    {
        "task": "segmentation",
        "group": "train_300_eval_200_seed47",
        "protocol": "adaptive sat-aware",
        "path": "optional_segmentation_ste1ep_continue_adaptive_sat_aware_300tr200ev_seed47.jsonl",
        "metric": "mIoU",
        "reference": "sat-aware control",
    },
    {
        "task": "segmentation",
        "group": "train_300_eval_200_seed47",
        "protocol": "classifier clean consistency",
        "path": "optional_segmentation_ste1ep_continue_classifier_clean_consistency_w0p02_300tr200ev_seed47.jsonl",
        "metric": "mIoU",
        "reference": "sat-aware control",
    },
]


ROLE_SUMMARIES = {
    "detection": "optional_detection_layer_noise_stats_seed41_8b_summary.json",
    "segmentation": "optional_segmentation_layer_noise_stats_seed41_8b_summary.json",
}

DISPLAY_LABELS = {
    "activation target 4": "activation target 4",
    "activation target 6": "activation target 6",
    "activation target 8": "activation target 8",
    "ROI read 4": "ROI read4",
    "RPN/ROI target 6 + adaptive read 8": "RPN/ROI target6 + read8",
    "variance sat-aware": "variance saturation",
    "adaptive sat-aware": "adaptive saturation",
    "FPN clean consistency": "FPN consistency",
    "classifier clean consistency": "classifier consistency",
}


def read_run(path: Path, metric: str) -> tuple[dict[str, Any], float]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    metadata = rows[0].get("metadata", rows[0])
    values = [
        row.get("eval", {}).get(metric)
        for row in rows[1:]
        if isinstance(row.get("eval", {}).get(metric), (int, float))
    ]
    if not values:
        raise ValueError(f"no {metric} value in {path}")
    return metadata, float(values[-1])


def build_rows() -> list[dict[str, Any]]:
    rows = []
    for experiment in EXPERIMENTS:
        path = RUNS_DIR / experiment["path"]
        metadata, value = read_run(path, experiment["metric"])
        rows.append(
            {
                **experiment,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "seed": metadata.get("seed"),
                "noise_state_semantics": "legacy_per_chunk",
                "train_batches": metadata.get("max_train_batches"),
                "eval_batches": metadata.get("max_eval_batches") or "full",
                "value": value,
                "value_percent": value * 100,
                "reference": experiment.get("reference", ""),
                "delta_pp": 0.0,
            }
        )

    lookup = {
        (row["task"], row["group"], row["protocol"]): row["value"] for row in rows
    }
    for row in rows:
        if row["reference"]:
            reference_value = lookup[(row["task"], row["group"], row["reference"])]
            row["delta_pp"] = (row["value"] - reference_value) * 100
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "group",
        "protocol",
        "metric",
        "seed",
        "noise_state_semantics",
        "train_batches",
        "eval_batches",
        "value",
        "value_percent",
        "reference",
        "delta_pp",
        "path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def write_role_summary(path: Path) -> None:
    rows = []
    for task, filename in ROLE_SUMMARIES.items():
        summary = json.loads((RUNS_DIR / filename).read_text())
        for role, layers in summary["layers_by_role"].items():
            rows.append(
                {
                    "task": task,
                    "role": role,
                    "layers": layers,
                    "residual_to_signal": summary[
                        "residual_to_signal_mean_by_role"
                    ][role],
                    "stochastic_to_signal": summary[
                        "stochastic_to_signal_mean_by_role"
                    ][role],
                    "bias_to_signal": summary["bias_to_signal_mean_by_role"][role],
                    "shape_mismatch_calls": summary["shape_mismatch_calls"],
                    "nonfinite_outputs": summary["nonfinite_outputs"],
                }
            )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bar_colors(values: list[float]) -> list[str]:
    return ["#59A14F" if value > 0 else "#E15759" if value < 0 else "#9D9D9D" for value in values]


def label_delta_bars(ax, bars, values: list[float]) -> None:
    for bar, value in zip(bars, values):
        if value < -0.25:
            ax.text(
                value + 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.2f}",
                va="center",
                ha="left",
                color="white",
                fontsize=9,
            )
        elif value < 0:
            ax.text(
                value - 0.03,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.2f}",
                va="center",
                ha="right",
                color="#333333",
                fontsize=9,
            )
        else:
            ax.text(
                value + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.2f}",
                va="center",
                ha="left",
                color="#333333",
                fontsize=9,
            )


def plot(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.0))

    for ax, task, title in (
        (axes[0, 0], "detection", "Detection full validation"),
        (axes[0, 1], "segmentation", "Segmentation full validation"),
    ):
        selected = [
            row
            for row in rows
            if row["task"] == task and row["group"] == "full_val_seed1"
        ]
        labels = [row["protocol"] for row in selected]
        values = [row["value_percent"] for row in selected]
        colors = ["#4C78A8", "#E15759", "#59A14F", "#F28E2B"][: len(values)]
        bars = ax.bar(range(len(values)), values, color=colors)
        ax.set_xticks(range(len(labels)), labels, rotation=18, ha="right")
        ax.set_ylabel(f"{selected[0]['metric']} (%)")
        ax.set_title(title)
        ax.bar_label(bars, fmt="%.2f", padding=3)

    detection_deltas = [
        row
        for row in rows
        if row["task"] == "detection" and row["reference"]
        and row["group"] != "full_val_seed1"
    ]
    ax = axes[1, 0]
    labels = [DISPLAY_LABELS.get(row["protocol"], row["protocol"]) for row in detection_deltas]
    values = [row["delta_pp"] for row in detection_deltas]
    bars = ax.barh(range(len(values)), values, color=bar_colors(values))
    ax.set_yticks(range(len(labels)), labels)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Delta vs matched control (pp)")
    ax.set_title("Detection algorithm screening")
    label_delta_bars(ax, bars, values)

    segmentation_deltas = [
        row
        for row in rows
        if row["task"] == "segmentation" and row["reference"]
    ]
    ax = axes[1, 1]
    labels = [DISPLAY_LABELS.get(row["protocol"], row["protocol"]) for row in segmentation_deltas]
    values = [row["delta_pp"] for row in segmentation_deltas]
    bars = ax.barh(range(len(values)), values, color=bar_colors(values))
    ax.set_yticks(range(len(labels)), labels)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Delta vs matched control (pp)")
    ax.set_title("Segmentation algorithm screening")
    label_delta_bars(ax, bars, values)

    for ax in axes.flat:
        ax.grid(axis="x" if ax in axes[1] else "y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    rows = build_rows()
    output = RUNS_DIR / "optional_algorithm_study_summary.csv"
    role_output = RUNS_DIR / "optional_role_noise_summary.csv"
    figure = PROJECT_ROOT / "docs" / "figures" / "optional_algorithm_study.png"
    write_csv(output, rows)
    write_role_summary(role_output)
    plot(rows, figure)
    print("note: all experiments in this table use legacy_per_chunk noise-state semantics")
    print(f"summary: {output}")
    print(f"role_summary: {role_output}")
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
