#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"

TASK_PATTERNS = (
    ("cifar10_resnet18", "CIFAR-10 + ResNet18"),
    ("cifar10_efficientnet_b0", "CIFAR-10 + EfficientNet-B0"),
    ("cifar100_resnet18", "CIFAR-100 + ResNet18"),
    ("cifar100_efficientnet_b0", "CIFAR-100 + EfficientNet-B0"),
    ("tinyimagenet_resnet18_imagenet224", "TinyImageNet + ResNet18 ImageNet-224"),
    (
        "tinyimagenet_efficientnet_b0_imagenet224",
        "TinyImageNet + EfficientNet-B0 ImageNet-224",
    ),
    ("tinyimagenet_resnet18", "TinyImageNet + ResNet18"),
    ("tinyimagenet_efficientnet_b0", "TinyImageNet + EfficientNet-B0"),
)

BASELINE_HINTS = (
    "clean120",
    "clean15",
    "clean10",
    "clean20",
    "clean30",
    "dwclean_clean120",
    "clean40",
    "clean120_noise_eval",
    "clean_checkpoint_noise_eval",
    "clean_direct_noise",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize repeat-eval CSVs with CIs, effect sizes, t-tests, and ANOVA."
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="*",
        default=sorted(RUNS_DIR.glob("*repeats.csv")),
        help="Repeat-eval CSV files. Defaults to runs/*repeats.csv.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=RUNS_DIR / "statistical_summary.csv",
    )
    parser.add_argument(
        "--comparisons-output",
        type=Path,
        default=RUNS_DIR / "statistical_pairwise_comparisons.csv",
    )
    parser.add_argument(
        "--anova-output",
        type=Path,
        default=RUNS_DIR / "statistical_anova.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=PROJECT_ROOT / "docs/figures/statistical_pairwise_improvements.png",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def finite_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def task_from_path(path: Path) -> tuple[str, str]:
    stem = path.stem
    for key, display_name in TASK_PATTERNS:
        if stem.startswith(key):
            return key, display_name
    return "unknown", stem


def protocol_from_path(path: Path, task_key: str, rows: list[dict[str, str]]) -> str:
    stem = path.stem
    if task_key != "unknown" and stem.startswith(task_key):
        protocol = stem[len(task_key) :].strip("_")
    else:
        protocol = stem
    protocol = protocol.removesuffix("_repeats")
    protocol = protocol.removesuffix("_noise_eval")
    protocol = protocol.removesuffix("_twostage")

    checkpoint_modes = {
        row.get("checkpoint_train_mode", "")
        for row in rows
        if row.get("checkpoint_train_mode", "")
    }
    if len(checkpoint_modes) == 1:
        mode = next(iter(checkpoint_modes))
        if mode and mode not in protocol:
            protocol = f"{protocol}__{mode}"
    return protocol or stem


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def hedges_g(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled_n = len(a) + len(b) - 2
    if pooled_n <= 0:
        return 0.0
    pooled_var = ((len(a) - 1) * stdev(a) ** 2 + (len(b) - 1) * stdev(b) ** 2) / pooled_n
    if pooled_var <= 0:
        return 0.0
    cohen_d = (mean(a) - mean(b)) / math.sqrt(pooled_var)
    correction = 1 - 3 / (4 * (len(a) + len(b)) - 9)
    return cohen_d * correction


def p_to_stars(p_value: float | None) -> str:
    if p_value is None or not math.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def is_baseline(protocol: str) -> bool:
    return any(hint in protocol for hint in BASELINE_HINTS)


def is_preferred_protocol(protocol: str) -> bool:
    preferred = (
        "ste",
        "sat_aware",
        "adaptive",
        "dw_clean",
        "pointwise",
        "calibrated",
    )
    return any(item in protocol for item in preferred)


def analyze_file(path: Path) -> dict[str, Any] | None:
    rows = read_csv(path)
    values = [
        value
        for row in rows
        if (value := finite_float(row.get("accuracy"))) is not None
    ]
    if not values:
        return None

    task_key, task_display = task_from_path(path)
    protocol = protocol_from_path(path, task_key, rows)
    dataset = next((row.get("dataset", "") for row in rows if row.get("dataset")), "")
    model_name = next((row.get("model_name", "") for row in rows if row.get("model_name")), "")
    if not dataset and task_key != "unknown":
        dataset = task_key.split("_")[0]
    if not model_name:
        model_name = "efficientnet_b0" if "efficientnet_b0" in task_key else "resnet18"

    normality_p = None
    if 3 <= len(values) <= 5000:
        try:
            normality_p = float(stats.shapiro(values).pvalue)
        except ValueError:
            normality_p = None

    return {
        "task_key": task_key,
        "task": task_display,
        "dataset": dataset,
        "model_name": model_name,
        "protocol": protocol,
        "runs": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "se": standard_error(values),
        "ci95": ci95(values),
        "min": min(values),
        "max": max(values),
        "normality_shapiro_p": normality_p if normality_p is not None else "",
        "source_csv": str(path),
        "_values": values,
    }


def best_nonbaseline(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if not is_baseline(row["protocol"]) and is_preferred_protocol(row["protocol"])
    ]
    if not candidates:
        candidates = [row for row in rows if not is_baseline(row["protocol"])]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row["mean"])


def baseline_for(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    baselines = [row for row in rows if is_baseline(row["protocol"])]
    if not baselines:
        return None
    return max(baselines, key=lambda row: row["mean"])


def build_pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_key"]].append(row)

    output = []
    for task_key, task_rows in sorted(grouped.items()):
        baseline = baseline_for(task_rows)
        best = best_nonbaseline(task_rows)
        if baseline is None or best is None:
            continue
        if len(baseline["_values"]) < 2 or len(best["_values"]) < 2:
            continue
        test = stats.ttest_ind(
            best["_values"], baseline["_values"], equal_var=False, nan_policy="omit"
        )
        output.append(
            {
                "task_key": task_key,
                "task": best["task"],
                "baseline_protocol": baseline["protocol"],
                "candidate_protocol": best["protocol"],
                "baseline_mean": baseline["mean"],
                "candidate_mean": best["mean"],
                "absolute_improvement": best["mean"] - baseline["mean"],
                "relative_improvement_percent": (
                    (best["mean"] - baseline["mean"]) / baseline["mean"] * 100
                    if baseline["mean"]
                    else ""
                ),
                "welch_t": float(test.statistic),
                "welch_p": float(test.pvalue),
                "significance": p_to_stars(float(test.pvalue)),
                "hedges_g": hedges_g(best["_values"], baseline["_values"]),
                "baseline_source": baseline["source_csv"],
                "candidate_source": best["source_csv"],
            }
        )
    return output


def build_anova(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["runs"] >= 2:
            grouped[row["task_key"]].append(row)

    output = []
    for task_key, task_rows in sorted(grouped.items()):
        if len(task_rows) < 3:
            continue
        groups = [row["_values"] for row in task_rows]
        try:
            result = stats.f_oneway(*groups)
        except ValueError:
            continue

        all_values = [value for group in groups for value in group]
        grand_mean = mean(all_values)
        ss_between = sum(len(group) * (mean(group) - grand_mean) ** 2 for group in groups)
        ss_total = sum((value - grand_mean) ** 2 for value in all_values)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
        output.append(
            {
                "task_key": task_key,
                "task": task_rows[0]["task"],
                "groups": len(task_rows),
                "observations": len(all_values),
                "anova_f": float(result.statistic),
                "anova_p": float(result.pvalue),
                "significance": p_to_stars(float(result.pvalue)),
                "eta_squared": eta_squared,
                "protocols": ";".join(row["protocol"] for row in task_rows),
            }
        )
    return output


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "_values"}


def plot_pairwise_improvements(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: row["absolute_improvement"], reverse=True)
    labels = [row["task"].replace(" + ", "\n") for row in rows]
    improvements = [100 * row["absolute_improvement"] for row in rows]
    stars = [row["significance"] for row in rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(range(len(rows)), improvements, color="#4C78A8", width=0.62)
    ax.set_title("Noisy accuracy improvement over clean checkpoint")
    ax.set_ylabel("Absolute improvement (percentage points)")
    ax.set_xticks(range(len(rows)), labels)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, star, value in zip(bars, stars, improvements):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{value:.1f} pp\n{star}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    summaries = [
        summary
        for path in args.inputs
        if (summary := analyze_file(path)) is not None
    ]
    summaries.sort(key=lambda row: (row["task_key"], row["protocol"], row["source_csv"]))

    pairwise = build_pairwise(summaries)
    anova = build_anova(summaries)
    write_csv(args.summary_output, [public_row(row) for row in summaries])
    write_csv(args.comparisons_output, pairwise)
    write_csv(args.anova_output, anova)
    plot_pairwise_improvements(args.figure, pairwise)
    print(
        {
            "groups": len(summaries),
            "pairwise_comparisons": len(pairwise),
            "anova_rows": len(anova),
        }
    )


if __name__ == "__main__":
    main()
