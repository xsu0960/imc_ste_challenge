#!/usr/bin/env python3
"""Summarize the large-image convolution noise-state correction study."""

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
OUTPUT_CSV = RUNS_DIR / "optional_conv_noise_state_study.csv"
OUTPUT_FIGURE = PROJECT_ROOT / "docs/figures/optional_conv_noise_state_study.png"


EXPERIMENTS = [
    {
        "task": "detection",
        "group": "scope_check_200_seed73",
        "protocol": "legacy chunk",
        "scope": "chunk",
        "checkpoint_stage": "legacy 1-epoch checkpoint",
        "checkpoint_train_examples": 2501,
        "path": "optional_detection_ste1ep_legacy_chunk_state_seed73_200ev.jsonl",
        "metric": "mAP50",
    },
    {
        "task": "detection",
        "group": "scope_check_200_seed73",
        "protocol": "shared read",
        "scope": "read",
        "checkpoint_stage": "legacy 1-epoch checkpoint",
        "checkpoint_train_examples": 2501,
        "path": "optional_detection_ste1ep_shared_read_state_seed73_200ev.jsonl",
        "metric": "mAP50",
        "reference": "legacy chunk",
    },
    {
        "task": "segmentation",
        "group": "scope_check_200_seed73",
        "protocol": "legacy chunk",
        "scope": "chunk",
        "checkpoint_stage": "legacy 1-epoch checkpoint",
        "checkpoint_train_examples": 1464,
        "path": "optional_segmentation_ste1ep_legacy_chunk_state_seed73_200ev.jsonl",
        "metric": "mIoU",
    },
    {
        "task": "segmentation",
        "group": "scope_check_200_seed73",
        "protocol": "shared read",
        "scope": "read",
        "checkpoint_stage": "legacy 1-epoch checkpoint",
        "checkpoint_train_examples": 1464,
        "path": "optional_segmentation_ste1ep_shared_read_state_seed73_200ev.jsonl",
        "metric": "mIoU",
        "reference": "legacy chunk",
    },
    {
        "task": "detection",
        "group": "corrected_full_val",
        "protocol": "clean",
        "scope": "none",
        "checkpoint_stage": "clean checkpoint",
        "checkpoint_train_examples": 0,
        "path": "optional_detection_internal_clean_eval_full.jsonl",
        "metric": "mAP50",
    },
    {
        "task": "detection",
        "group": "corrected_full_val",
        "protocol": "direct noisy",
        "scope": "read",
        "checkpoint_stage": "clean checkpoint",
        "checkpoint_train_examples": 0,
        "path": "optional_detection_shared_read_state_direct_fullval_seed89.jsonl",
        "metric": "mAP50",
    },
    {
        "task": "detection",
        "group": "corrected_full_val",
        "protocol": "1-epoch STE",
        "scope": "read",
        "checkpoint_stage": "shared-read 1-epoch checkpoint",
        "checkpoint_train_examples": 2501,
        "path": "optional_detection_shared_read_state_ste_full_1epoch_paired_eval_seed89.jsonl",
        "metric": "mAP50",
        "reference": "direct noisy",
    },
    {
        "task": "segmentation",
        "group": "corrected_full_val",
        "protocol": "clean",
        "scope": "none",
        "checkpoint_stage": "clean checkpoint",
        "checkpoint_train_examples": 0,
        "path": "optional_segmentation_internal_clean_eval_full.jsonl",
        "metric": "mIoU",
    },
    {
        "task": "segmentation",
        "group": "corrected_full_val",
        "protocol": "direct noisy",
        "scope": "read",
        "checkpoint_stage": "clean checkpoint",
        "checkpoint_train_examples": 0,
        "path": "optional_segmentation_shared_read_state_direct_fullval_seed89.jsonl",
        "metric": "mIoU",
    },
    {
        "task": "segmentation",
        "group": "corrected_full_val",
        "protocol": "1-epoch STE",
        "scope": "read",
        "checkpoint_stage": "shared-read 1-epoch checkpoint",
        "checkpoint_train_examples": 1464,
        "path": "optional_segmentation_shared_read_state_ste_full_1epoch_paired_eval_seed89.jsonl",
        "metric": "mIoU",
        "reference": "direct noisy",
    },
]


def read_run(path: Path, metric: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty run file: {path}")
    metadata = rows[0].get("metadata", rows[0])
    eval_rows = [
        row for row in rows[1:] if isinstance(row.get("eval", {}).get(metric), (int, float))
    ]
    if not eval_rows:
        raise ValueError(f"no {metric} value in {path}")
    return metadata, eval_rows[-1]


def build_rows() -> list[dict[str, Any]]:
    rows = []
    for experiment in EXPERIMENTS:
        path = RUNS_DIR / experiment["path"]
        metadata, result = read_run(path, experiment["metric"])
        value = float(result["eval"][experiment["metric"]])
        rows.append(
            {
                **experiment,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "seed": metadata.get("seed", ""),
                "checkpoint_stage": experiment["checkpoint_stage"],
                "checkpoint_train_examples": experiment["checkpoint_train_examples"],
                "eval_examples": result.get("eval", {}).get("examples", ""),
                "noise_scale": (
                    0.0
                    if experiment["scope"] == "none"
                    else metadata.get("noise_scale", "")
                ),
                "conv_chunk_rows": (
                    ""
                    if experiment["scope"] == "none"
                    else metadata.get("conv_chunk_rows", "")
                ),
                "value": value,
                "value_percent": value * 100,
                "reference": experiment.get("reference", ""),
                "delta_pp": 0.0,
                "retained_clean_percent": "",
                "clean_gap_recovery_percent": "",
            }
        )

    lookup = {
        (row["task"], row["group"], row["protocol"]): row["value"] for row in rows
    }
    for row in rows:
        if row["reference"]:
            reference = lookup[(row["task"], row["group"], row["reference"])]
            row["delta_pp"] = (row["value"] - reference) * 100
        if row["group"] == "corrected_full_val":
            clean = lookup[(row["task"], row["group"], "clean")]
            row["retained_clean_percent"] = row["value"] / clean * 100
            if row["protocol"] == "1-epoch STE":
                direct = lookup[(row["task"], row["group"], "direct noisy")]
                row["clean_gap_recovery_percent"] = (
                    (row["value"] - direct) / (clean - direct) * 100
                )
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "task",
        "group",
        "protocol",
        "scope",
        "metric",
        "seed",
        "checkpoint_stage",
        "checkpoint_train_examples",
        "eval_examples",
        "noise_scale",
        "conv_chunk_rows",
        "value",
        "value_percent",
        "reference",
        "delta_pp",
        "retained_clean_percent",
        "clean_gap_recovery_percent",
        "path",
    ]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def annotate_bars(axis: plt.Axes, bars: Any) -> None:
    for bar in bars:
        height = bar.get_height()
        axis.annotate(
            f"{height:.2f}",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def write_figure(rows: list[dict[str, Any]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    tasks = ["detection", "segmentation"]
    scope_rows = {
        (row["task"], row["protocol"]): row
        for row in rows
        if row["group"] == "scope_check_200_seed73"
    }
    x = range(len(tasks))
    width = 0.34
    legacy = axes[0].bar(
        [index - width / 2 for index in x],
        [scope_rows[(task, "legacy chunk")]["value_percent"] for task in tasks],
        width,
        label="Legacy per-chunk state",
        color="#D97706",
    )
    shared = axes[0].bar(
        [index + width / 2 for index in x],
        [scope_rows[(task, "shared read")]["value_percent"] for task in tasks],
        width,
        label="Shared per-read state",
        color="#2563EB",
    )
    axes[0].set_title("Same checkpoint, 200-image scope check")
    axes[0].set_xticks(list(x), ["Detection mAP50", "Segmentation mIoU"])
    axes[0].set_ylabel("Metric (%)")
    axes[0].legend(frameon=False)
    annotate_bars(axes[0], legacy)
    annotate_bars(axes[0], shared)

    full_rows = {
        (row["task"], row["protocol"]): row
        for row in rows
        if row["group"] == "corrected_full_val"
    }
    protocols = ["clean", "direct noisy", "1-epoch STE"]
    colors = ["#4B5563", "#DC2626", "#059669"]
    bar_width = 0.24
    for offset, (protocol, color) in enumerate(zip(protocols, colors)):
        bars = axes[1].bar(
            [index + (offset - 1) * bar_width for index in x],
            [full_rows[(task, protocol)]["value_percent"] for task in tasks],
            bar_width,
            label=protocol,
            color=color,
        )
        annotate_bars(axes[1], bars)
    axes[1].set_title("Corrected shared-read full validation")
    axes[1].set_xticks(list(x), ["Detection mAP50", "Segmentation mIoU"])
    axes[1].set_ylabel("Metric (%)")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylim(bottom=0)
    figure.suptitle("Large-image convolution noise-state correction", fontweight="bold")
    figure.tight_layout()
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    write_figure(rows)
    print(f"summary: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"figure: {OUTPUT_FIGURE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
