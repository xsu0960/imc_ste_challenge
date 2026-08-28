#!/usr/bin/env python3
"""Rebuild submission-facing result tables and the main comparison figure."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_PREFIXES = ("CIFAR-10", "CIFAR-100", "TinyImageNet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-source",
        type=Path,
        default=PROJECT_ROOT / "runs" / "main_uniform_noise_summary.csv",
    )
    parser.add_argument(
        "--optional-source",
        type=Path,
        default=PROJECT_ROOT / "runs" / "optional_full_validation_summary.csv",
    )
    parser.add_argument(
        "--paired-source",
        type=Path,
        default=PROJECT_ROOT / "runs" / "optional_paired_extension_summary.csv",
    )
    parser.add_argument(
        "--efficiency-source",
        type=Path,
        default=PROJECT_ROOT / "runs" / "efficiency_benchmark.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "submission_key_results.csv",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "generated" / "key_results.md",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=PROJECT_ROOT / "docs" / "figures" / "submission_key_results.png",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing result source: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close_enough(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"inconsistent {label}: computed {actual}, stored {expected}")


def main_results(path: Path) -> list[dict[str, str | float]]:
    results: list[dict[str, str | float]] = []
    for row in read_rows(path):
        task = row["task"]
        if not task.startswith(CLASSIFICATION_PREFIXES):
            continue
        clean = float(row["clean_eval"])
        direct = float(row["direct_uniform_noisy"])
        best = float(row["best_uniform_ste"])
        recovery = (best - direct) * 100.0
        retained = best / clean * 100.0
        close_enough(recovery, float(row["recovery_pp"]), 0.002, f"{task} recovery")
        close_enough(retained / 100.0, float(row["retained_clean_ratio"]), 0.0002, f"{task} retained ratio")
        results.append(
            {
                "section": "mandatory_classification",
                "task": task,
                "metric": "accuracy",
                "clean_percent": clean * 100.0,
                "baseline_percent": direct * 100.0,
                "candidate_percent": best * 100.0,
                "delta_pp": recovery,
                "retained_clean_percent": retained,
                "ci95_pp": "0.247" if "5-noise-seed" in row["notes"] else "",
                "p_value": "",
                "decision": "main_result",
                "protocol": row["best_uniform_protocol"],
            }
        )
    if len(results) != 6:
        raise ValueError(f"expected 6 mandatory classification rows, found {len(results)}")
    return results


def optional_results(path: Path) -> list[dict[str, str | float]]:
    rows = read_rows(path)
    results: list[dict[str, str | float]] = []
    for task in ("detection", "segmentation"):
        task_rows = [row for row in rows if row["task"] == task]
        if len(task_rows) != 1:
            raise ValueError(f"expected one optional full-validation row for {task}")
        row = task_rows[0]
        clean = float(row["clean_percent"])
        direct = float(row["direct_noisy_percent"])
        best = float(row["ste_percent"])
        delta = best - direct
        close_enough(delta, float(row["recovery_pp"]), 1e-9, f"optional {task} delta")
        results.append(
            {
                "section": "optional_full_val",
                "task": "VOC2007 + Faster R-CNN" if task == "detection" else "VOC2012 + DeepLabV3",
                "metric": "mAP50" if task == "detection" else "mIoU",
                "clean_percent": clean,
                "baseline_percent": direct,
                "candidate_percent": best,
                "delta_pp": delta,
                "retained_clean_percent": best / clean * 100.0,
                "ci95_pp": "",
                "p_value": "",
                "decision": "mechanism_validation",
                "protocol": "shared-read uniform noise, one full epoch",
            }
        )
    return results


def paired_extension_results(path: Path) -> list[dict[str, str | float]]:
    rows = read_rows(path)
    selected = [
        row for row in rows
        if row["family"] == "segmentation_online_consistency_range"
    ]
    if len(selected) != 1:
        raise ValueError("expected one segmentation paired-extension row")
    row = selected[0]
    control = float(row["control_mean_percent"])
    candidate = float(row["candidate_mean_percent"])
    delta = candidate - control
    close_enough(delta, float(row["paired_delta_mean_pp"]), 1e-9, "paired-extension delta")
    return [
        {
            "section": "optional_paired_extension",
            "task": "VOC2012 + DeepLabV3 online extension",
            "metric": "mIoU",
            "clean_percent": "",
            "baseline_percent": control,
            "candidate_percent": candidate,
            "delta_pp": delta,
            "retained_clean_percent": "",
            "ci95_pp": float(row["paired_delta_ci95_pp"]),
            "p_value": float(row["paired_p_value"]),
            "decision": row["conclusion"],
            "protocol": f"{row['runs']}-seed paired evaluation",
        }
    ]


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(value: str | float) -> str:
    return "-" if value == "" else f"{float(value):.2f}%"


def build_markdown(
    rows: list[dict[str, str | float]], efficiency_rows: list[dict[str, str]]
) -> str:
    main = [row for row in rows if row["section"] == "mandatory_classification"]
    optional = [row for row in rows if row["section"] == "optional_full_val"]
    paired = [row for row in rows if row["section"] == "optional_paired_extension"]
    lines = [
        "# Key Results",
        "",
        "## Mandatory Classification",
        "",
        "| Task | Clean | Direct noisy | Best STE | Recovery | Retained clean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in main:
        lines.append(
            f"| {row['task']} | {percent(row['clean_percent'])} | "
            f"{percent(row['baseline_percent'])} | {percent(row['candidate_percent'])} | "
            f"{float(row['delta_pp']):+.2f} pp | {percent(row['retained_clean_percent'])} |"
        )
    lines.extend(
        [
            "",
            "## Optional Full Validation",
            "",
            "| Task | Metric | Clean | Direct noisy | 1-epoch STE | Recovery |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in optional:
        lines.append(
            f"| {row['task']} | {row['metric']} | {percent(row['clean_percent'])} | "
            f"{percent(row['baseline_percent'])} | {percent(row['candidate_percent'])} | "
            f"{float(row['delta_pp']):+.2f} pp |"
        )
    row = paired[0]
    lines.extend(
        [
            "",
            "## Optional Paired Extension",
            "",
            "| Task | Control | Candidate | Delta | CI95 | p-value | Decision |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            f"| {row['task']} | {percent(row['baseline_percent'])} | "
            f"{percent(row['candidate_percent'])} | {float(row['delta_pp']):+.3f} pp | "
            f"+/- {float(row['ci95_pp']):.3f} pp | {float(row['p_value']):.4f} | "
            f"{row['decision']} |",
        ]
    )
    if efficiency_rows:
        lines.extend(
            [
                "",
                "## Engineering Efficiency",
                "",
                "| Path | Step time | Relative to clean | Throughput | Incremental peak memory |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for efficiency in efficiency_rows:
            lines.append(
                f"| {efficiency['mode']} | {float(efficiency['mean_step_ms']):.2f} ms | "
                f"{float(efficiency['time_vs_clean']):.2f}x | "
                f"{float(efficiency['throughput_images_s']):.2f} images/s | "
                f"{float(efficiency['incremental_peak_allocated_mib']):.2f} MiB |"
            )
    return "\n".join(lines) + "\n"


def plot_main(path: Path, rows: list[dict[str, str | float]]) -> None:
    main = [row for row in rows if row["section"] == "mandatory_classification"]
    labels = [str(row["task"]).replace(" + ", "\n") for row in main]
    clean = [float(row["clean_percent"]) for row in main]
    direct = [float(row["baseline_percent"]) for row in main]
    best = [float(row["candidate_percent"]) for row in main]
    positions = list(range(len(main)))
    width = 0.25
    figure, axis = plt.subplots(figsize=(12, 5.2))
    axis.bar([value - width for value in positions], clean, width, label="Clean", color="#4C78A8")
    axis.bar(positions, direct, width, label="Direct noisy", color="#E45756")
    axis.bar([value + width for value in positions], best, width, label="Best STE", color="#54A24B")
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(0, 105)
    axis.set_xticks(positions, labels, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, loc="upper center")
    axis.set_title("Mandatory classification under uniform noise scale 1.0")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = main_results(args.main_source)
    rows.extend(optional_results(args.optional_source))
    rows.extend(paired_extension_results(args.paired_source))
    efficiency_rows = read_rows(args.efficiency_source) if args.efficiency_source.exists() else []
    write_csv(args.output, rows)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(
        build_markdown(rows, efficiency_rows), encoding="utf-8"
    )
    plot_main(args.figure, rows)
    print(f"validated rows: {len(rows)}")
    print(f"csv: {args.output}")
    print(f"markdown: {args.markdown_output}")
    print(f"figure: {args.figure}")


if __name__ == "__main__":
    main()
