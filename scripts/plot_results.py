#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate report figures from aggregated experiment CSV files."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "runs/summary.csv",
    )
    parser.add_argument(
        "--gradient-diagnostics",
        type=Path,
        default=PROJECT_ROOT / "runs/gradient_diagnostics.csv",
    )
    parser.add_argument(
        "--checkpoint-sweep",
        type=Path,
        default=PROJECT_ROOT / "runs/checkpoint_noise_sweep.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs/figures",
    )
    parser.add_argument(
        "--checkpoint-sweep-summary",
        type=Path,
        default=PROJECT_ROOT / "runs/checkpoint_noise_sweep_summary.csv",
    )
    parser.add_argument(
        "--layerwise-sweep-summary",
        type=Path,
        default=PROJECT_ROOT / "runs/cifar10_efficientnet_b0_layerwise_eval_sweep_summary.csv",
    )
    parser.add_argument(
        "--tinyimagenet-layerwise-sweep-summary",
        type=Path,
        default=PROJECT_ROOT
        / "runs/tinyimagenet_efficientnet_b0_imagenet224_layerwise_eval_sweep_summary.csv",
    )
    parser.add_argument(
        "--current-best",
        type=Path,
        default=PROJECT_ROOT / "runs/current_best_noisy_summary.csv",
    )
    parser.add_argument(
        "--main-uniform-summary",
        type=Path,
        default=PROJECT_ROOT / "runs/main_uniform_noise_summary.csv",
    )
    parser.add_argument(
        "--tinyimagenet-summary",
        type=Path,
        default=PROJECT_ROOT / "runs/tinyimagenet_imagenet224_summary.csv",
    )
    parser.add_argument(
        "--optional-domain-summaries",
        type=Path,
        nargs="*",
        default=[
            PROJECT_ROOT / "runs/optional_detection_voc100_summary.csv",
            PROJECT_ROOT / "runs/optional_segmentation_voc50_summary.csv",
        ],
    )
    parser.add_argument(
        "--optional-internal-summary",
        type=Path,
        default=PROJECT_ROOT / "runs/optional_internal_ste_pilot_summary.csv",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def close_float(value: str, target: float) -> bool:
    parsed = to_float(value)
    return parsed is not None and math.isclose(parsed, target, rel_tol=0.0, abs_tol=1e-9)


def style_axes(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"wrote {path}")


def short_task_label(task: str) -> str:
    replacements = {
        "CIFAR-10 + ResNet18": "C10\nR18",
        "CIFAR-10 + EfficientNet-B0": "C10\nB0",
        "CIFAR-100 + ResNet18": "C100\nR18",
        "CIFAR-100 + EfficientNet-B0": "C100\nB0",
        "TinyImageNet + ResNet18 ImageNet-224": "Tiny\nR18",
        "TinyImageNet + EfficientNet-B0 ImageNet-224": "Tiny\nB0",
    }
    return replacements.get(task, task)


def find_summary_row(
    rows: list[dict[str, str]],
    *,
    mode: str,
    epochs: int,
    learning_rate: float,
    max_train_batches: int,
    max_eval_batches: int,
) -> dict[str, str] | None:
    for row in rows:
        if row.get("dataset") != "cifar10":
            continue
        if row.get("model_name") != "resnet18":
            continue
        if row.get("train_mode") != mode or row.get("eval_mode") != "noise":
            continue
        if row.get("epochs") != str(epochs):
            continue
        if not close_float(row.get("learning_rate", ""), learning_rate):
            continue
        if not close_float(row.get("grad_clip_norm", ""), 1.0):
            continue
        if row.get("max_train_batches") != str(max_train_batches):
            continue
        if row.get("max_eval_batches") != str(max_eval_batches):
            continue
        return row
    return None


def plot_long_training(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    modes = ["clean", "sat_aware_ste"]
    labels = ["Clean", "Sat-aware STE"]
    selected = [
        find_summary_row(
            summary_rows,
            mode=mode,
            epochs=20,
            learning_rate=0.06,
            max_train_batches=200,
            max_eval_batches=40,
        )
        for mode in modes
    ]
    if any(row is None for row in selected):
        print("skipped long-training plot: required rows not found")
        return

    final_means = [to_float(row["final_mean"], 0.0) for row in selected]
    final_ci = [to_float(row["final_ci95"], 0.0) for row in selected]
    best_means = [to_float(row["best_mean"], 0.0) for row in selected]
    best_ci = [to_float(row["best_ci95"], 0.0) for row in selected]

    x = range(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(
        [item - width / 2 for item in x],
        final_means,
        width,
        yerr=final_ci,
        capsize=4,
        label="Final",
        color="#4C78A8",
    )
    ax.bar(
        [item + width / 2 for item in x],
        best_means,
        width,
        yerr=best_ci,
        capsize=4,
        label="Best",
        color="#F58518",
    )
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0.70, 0.84)
    ax.set_title("CIFAR-10 / ResNet18, 20 epochs")
    style_axes(ax, "Noisy eval accuracy")
    ax.legend(frameon=False)
    save_current(output_dir / "resnet18_20epoch_comparison.png")


def plot_lr_sweep(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    modes = {
        "clean": ("Clean", "#4C78A8"),
        "sat_aware_ste": ("Sat-aware STE", "#F58518"),
        "adaptive_sat_aware_ste": ("Adaptive sat-aware STE", "#54A24B"),
    }
    series: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row in summary_rows:
        if row.get("dataset") != "cifar10" or row.get("model_name") != "resnet18":
            continue
        if row.get("eval_mode") != "noise" or row.get("epochs") != "10":
            continue
        if row.get("max_train_batches") != "100" or row.get("max_eval_batches") != "40":
            continue
        if not close_float(row.get("grad_clip_norm", ""), 1.0):
            continue
        mode = row.get("train_mode", "")
        if mode not in modes:
            continue
        lr = to_float(row.get("learning_rate"))
        final = to_float(row.get("final_mean"))
        ci = to_float(row.get("final_ci95"), 0.0)
        if lr is None or final is None:
            continue
        series[mode].append((lr, final, ci or 0.0))

    if not series:
        print("skipped learning-rate plot: required rows not found")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for mode, (label, color) in modes.items():
        points = sorted(series.get(mode, []))
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        yerr = [point[2] for point in points]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker="o",
            linewidth=2.0,
            capsize=3,
            label=label,
            color=color,
        )
    ax.set_title("Learning-rate sweep, 10 epochs")
    ax.set_xlabel("Learning rate")
    ax.set_ylim(0.35, 0.72)
    style_axes(ax, "Final noisy eval accuracy")
    ax.legend(frameon=False)
    save_current(output_dir / "lr_sweep_10epoch.png")


def plot_gradient_diagnostics(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        print("skipped gradient plot: diagnostics CSV not found")
        return

    labels = [row["method"] for row in rows]
    bias = [to_float(row.get("relative_bias"), 0.0) for row in rows]
    variance = [to_float(row.get("relative_variance"), 0.0) for row in rows]
    cosine = [to_float(row.get("cosine_of_mean"), 0.0) for row in rows]
    x = list(range(len(labels)))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar([item - width for item in x], bias, width, label="Relative bias", color="#E45756")
    ax.bar(x, variance, width, label="Relative variance", color="#72B7B2")
    ax.bar([item + width for item in x], cosine, width, label="Cosine of mean", color="#54A24B")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_title("Gradient estimator diagnostics")
    style_axes(ax, "Value")
    ax.legend(frameon=False, ncols=3, loc="upper center")
    save_current(output_dir / "gradient_diagnostics.png")


def summarize_checkpoint_sweep(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        mode = row.get("checkpoint_train_mode", "")
        noise_scale = to_float(row.get("noise_scale"))
        accuracy = to_float(row.get("accuracy"))
        if not mode or noise_scale is None or accuracy is None:
            continue
        groups[(mode, noise_scale)].append(accuracy)

    output = []
    for (mode, noise_scale), values in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        output.append(
            {
                "checkpoint_train_mode": mode,
                "noise_scale": noise_scale,
                "runs": len(values),
                "accuracy_mean": mean(values),
                "accuracy_std": stdev(values) if len(values) > 1 else 0.0,
                "accuracy_ci95": ci95(values),
            }
        )
    return output


def write_checkpoint_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def plot_checkpoint_sweep(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        print("skipped checkpoint sweep plot: sweep CSV not found")
        return
    modes = {
        "clean": ("Clean", "#4C78A8"),
        "sat_aware_ste": ("Sat-aware STE", "#F58518"),
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for mode, (label, color) in modes.items():
        points = [
            row
            for row in rows
            if row.get("checkpoint_train_mode") == mode
        ]
        points = sorted(points, key=lambda row: row["noise_scale"])
        if not points:
            continue
        xs = [row["noise_scale"] for row in points]
        ys = [row["accuracy_mean"] for row in points]
        yerr = [row["accuracy_ci95"] for row in points]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=2.0, capsize=3, label=label, color=color)
    ax.set_title("Checkpoint robustness vs. noise scale")
    ax.set_xlabel("Noise scale")
    ax.set_ylim(0.30, 0.86)
    style_axes(ax, "Noisy eval accuracy")
    ax.legend(frameon=False)
    save_current(output_dir / "noise_scale_sweep.png")


def plot_current_best(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        print("skipped current-best plot: CSV not found")
        return

    labels = [short_task_label(row.get("task", "")) for row in rows]
    clean = [to_float(row.get("clean_eval_accuracy"), 0.0) or 0.0 for row in rows]
    noisy = [to_float(row.get("best_noisy_mean"), 0.0) or 0.0 for row in rows]
    noisy_ci = [to_float(row.get("best_noisy_ci95"), 0.0) or 0.0 for row in rows]

    x = list(range(len(rows)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(
        [item - width / 2 for item in x],
        clean,
        width,
        label="Clean eval",
        color="#4C78A8",
    )
    ax.bar(
        [item + width / 2 for item in x],
        noisy,
        width,
        yerr=noisy_ci,
        capsize=3,
        label="Best noisy eval",
        color="#F58518",
    )
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Current best noisy accuracy by task")
    style_axes(ax, "Accuracy")
    ax.legend(frameon=False, ncols=2, loc="upper center")
    for item, value in enumerate(noisy):
        ax.text(item + width / 2, value + 0.025, f"{100 * value:.1f}%", ha="center", fontsize=8)
    save_current(output_dir / "current_best_noisy_summary.png")


def plot_main_uniform_noise(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        print("skipped main-uniform plot: CSV not found")
        return

    classification_rows = [
        row for row in rows if not row.get("task", "").startswith("VOC")
    ]
    if not classification_rows:
        print("skipped main-uniform plot: no classification rows found")
        return

    labels = [short_task_label(row.get("task", "")) for row in classification_rows]
    clean = [to_float(row.get("clean_eval"), float("nan")) for row in classification_rows]
    direct = [
        to_float(row.get("direct_uniform_noisy"), float("nan"))
        for row in classification_rows
    ]
    ste = [
        to_float(row.get("best_uniform_ste"), float("nan"))
        for row in classification_rows
    ]

    x = list(range(len(classification_rows)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    series = [
        ("Clean", clean, -width, "#4C78A8"),
        ("Direct uniform noisy", direct, 0.0, "#E45756"),
        ("Uniform STE", ste, width, "#54A24B"),
    ]
    for label, values, offset, color in series:
        ax.bar(
            [item + offset for item in x],
            values,
            width,
            label=label,
            color=color,
        )
        for item, value in enumerate(values):
            if value is None or not math.isfinite(value):
                continue
            ax.text(
                item + offset,
                value + 0.025,
                f"{100 * value:.1f}%",
                ha="center",
                fontsize=8,
            )

    for item, value in enumerate(ste):
        if value is None or not math.isfinite(value):
            ax.text(item + width, 0.035, "pending", ha="center", fontsize=8)

    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Strict uniform noise main results")
    style_axes(ax, "Accuracy")
    ax.legend(frameon=False, ncols=3, loc="upper center")
    save_current(output_dir / "main_uniform_noise_summary.png")


def plot_tinyimagenet_summary(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        print("skipped TinyImageNet summary plot: CSV not found")
        return

    grouped: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for row in rows:
        task = row.get("task", "")
        protocol = row.get("protocol", "")
        value = to_float(row.get("clean_accuracy"))
        if value is None:
            value = to_float(row.get("repeat_mean"))
        if value is None:
            continue
        grouped[task].append((protocol, value, to_float(row.get("ci95"), 0.0) or 0.0))

    protocol_labels = {
        "clean eval": "Clean",
        "clean checkpoint noisy eval": "Direct noisy",
        "strict uniform STE fine-tune": "Uniform STE",
        "sat-aware STE fine-tune": "Sat-aware STE",
        "depthwise-clean noisy eval": "DW clean",
        "layerwise calibrated depthwise-clean noisy eval": "Layerwise cal.",
    }
    colors = {
        "Clean": "#4C78A8",
        "Direct noisy": "#E45756",
        "Uniform STE": "#54A24B",
        "Sat-aware STE": "#F58518",
        "DW clean": "#B279A2",
        "Layerwise cal.": "#54A24B",
    }

    fig, axes = plt.subplots(1, len(grouped), figsize=(11.0, 4.6), sharey=True)
    if len(grouped) == 1:
        axes = [axes]
    for ax, (task, items) in zip(axes, grouped.items()):
        labels = [protocol_labels.get(protocol, protocol) for protocol, _, _ in items]
        values = [value for _, value, _ in items]
        errors = [error for _, _, error in items]
        bar_colors = [colors.get(label, "#72B7B2") for label in labels]
        ax.bar(range(len(labels)), values, yerr=errors, capsize=3, color=bar_colors)
        ax.set_title(short_task_label(task).replace("\n", " "))
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        ax.set_ylim(0.0, 0.88)
        style_axes(ax, "Accuracy")
        for index, value in enumerate(values):
            ax.text(index, value + 0.025, f"{100 * value:.1f}%", ha="center", fontsize=8)
    fig.suptitle("TinyImageNet ImageNet-224 results", y=0.98)
    save_current(output_dir / "tinyimagenet_imagenet224_summary.png")


def plot_optional_domain(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        print("skipped optional-domain plot: summary CSVs not found")
        return

    task_labels = {
        "detection": "Detection\nmAP50",
        "segmentation": "Segmentation\nmIoU",
    }
    protocol_labels = {
        "clean": "Clean",
        "output_noise": "Output noise",
    }
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        task = row.get("task", "")
        protocol = row.get("protocol", "")
        value = to_float(row.get("mean"))
        if task not in task_labels or protocol not in protocol_labels or value is None:
            continue
        grouped[task][protocol] = value

    tasks = [task for task in ["detection", "segmentation"] if task in grouped]
    if not tasks:
        print("skipped optional-domain plot: no supported rows found")
        return

    x = list(range(len(tasks)))
    width = 0.34
    clean = [grouped[task].get("clean", 0.0) for task in tasks]
    noisy = [grouped[task].get("output_noise", 0.0) for task in tasks]

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.bar(
        [item - width / 2 for item in x],
        clean,
        width,
        label="Clean",
        color="#4C78A8",
    )
    ax.bar(
        [item + width / 2 for item in x],
        noisy,
        width,
        label="Output noise",
        color="#E45756",
    )
    ax.set_xticks(x, [task_labels[task] for task in tasks])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Optional domain pilot evaluation")
    style_axes(ax, "Metric value")
    ax.legend(frameon=False, loc="upper center", ncols=2)
    for index, value in enumerate(clean):
        ax.text(index - width / 2, value + 0.025, f"{100 * value:.1f}%", ha="center", fontsize=8)
    for index, value in enumerate(noisy):
        ax.text(index + width / 2, value + 0.025, f"{100 * value:.1f}%", ha="center", fontsize=8)
    save_current(output_dir / "optional_domain_pilot.png")


def plot_optional_internal(rows: list[dict[str, str]], output_dir: Path) -> None:
    if not rows:
        print("skipped optional-internal plot: summary CSV not found")
        return

    task_labels = {
        "detection": "Detection\nmAP50",
        "segmentation": "Segmentation\nmIoU",
    }
    protocol_labels = {
        "clean_internal_full_val": "Clean",
        "direct_internal_noisy_full_val": "Direct noisy",
        "sat_aware_ste_internal_finetune_1epoch_full": "STE 1 epoch",
    }
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        task = row.get("task", "")
        protocol = row.get("protocol", "")
        value = to_float(row.get("value"))
        if task not in task_labels or protocol not in protocol_labels or value is None:
            continue
        grouped[task][protocol] = value

    tasks = [task for task in ["detection", "segmentation"] if task in grouped]
    if not tasks:
        print("skipped optional-internal plot: no full rows found")
        return

    x = list(range(len(tasks)))
    width = 0.25
    clean = [grouped[task].get("clean_internal_full_val", 0.0) for task in tasks]
    direct = [grouped[task].get("direct_internal_noisy_full_val", 0.0) for task in tasks]
    ste = [
        grouped[task].get("sat_aware_ste_internal_finetune_1epoch_full", 0.0)
        for task in tasks
    ]

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.bar(
        [item - width for item in x],
        clean,
        width,
        label="Clean",
        color="#4C78A8",
    )
    ax.bar(
        x,
        direct,
        width,
        label="Direct internal noisy",
        color="#E45756",
    )
    ax.bar(
        [item + width for item in x],
        ste,
        width,
        label="Internal STE, 1 epoch",
        color="#54A24B",
    )
    ax.set_xticks(x, [task_labels[task] for task in tasks])
    ax.set_ylim(0.0, max([*clean, *direct, *ste, 0.1]) * 1.22)
    ax.set_title("Optional internal STE full validation")
    style_axes(ax, "Metric value")
    ax.legend(frameon=False, loc="upper center", ncols=3)
    for index, value in enumerate(clean):
        ax.text(index - width, value + 0.012, f"{100 * value:.1f}%", ha="center", fontsize=8)
    for index, value in enumerate(direct):
        ax.text(index, value + 0.012, f"{100 * value:.1f}%", ha="center", fontsize=8)
    for index, value in enumerate(ste):
        ax.text(index + width, value + 0.012, f"{100 * value:.1f}%", ha="center", fontsize=8)
    save_current(output_dir / "optional_internal_ste_full.png")


def plot_layerwise_sweep(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    output_name: str = "efficientnet_b0_layerwise_noise_sweep.png",
    title: str = "EfficientNet-B0 layer-wise noise calibration",
    vmin: float | None = 0.87,
    vmax: float | None = 0.905,
) -> None:
    if not rows:
        print("skipped layerwise sweep plot: sweep summary CSV not found")
        return

    pointwise_scales = sorted(
        {
            value
            for row in rows
            if (value := to_float(row.get("pointwise_noise_scale"))) is not None
        },
        reverse=True,
    )
    linear_scales = sorted(
        {
            value
            for row in rows
            if (value := to_float(row.get("linear_noise_scale"))) is not None
        },
        reverse=True,
    )
    if not pointwise_scales or not linear_scales:
        print("skipped layerwise sweep plot: missing scale columns")
        return

    values: dict[tuple[float, float], float] = {}
    for row in rows:
        pointwise = to_float(row.get("pointwise_noise_scale"))
        linear = to_float(row.get("linear_noise_scale"))
        accuracy = to_float(row.get("mean"))
        if pointwise is None or linear is None or accuracy is None:
            continue
        values[(pointwise, linear)] = accuracy

    matrix = [
        [values.get((pointwise, linear), float("nan")) for linear in linear_scales]
        for pointwise in pointwise_scales
    ]
    finite_values = [value for value in values.values() if math.isfinite(value)]
    if vmin is None:
        vmin = min(finite_values) if finite_values else 0.0
    if vmax is None:
        vmax = max(finite_values) if finite_values else 1.0
    text_threshold = vmin + 0.55 * (vmax - vmin)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.87, vmax=0.905)
    image.set_clim(vmin, vmax)
    ax.set_title(title)
    ax.set_xlabel("Linear noise scale")
    ax.set_ylabel("Pointwise conv noise scale")
    ax.set_xticks(range(len(linear_scales)), [f"{value:g}" for value in linear_scales])
    ax.set_yticks(
        range(len(pointwise_scales)), [f"{value:g}" for value in pointwise_scales]
    )

    for row_index, pointwise in enumerate(pointwise_scales):
        for col_index, linear in enumerate(linear_scales):
            value = values.get((pointwise, linear))
            if value is None:
                continue
            color = "white" if value < text_threshold else "black"
            ax.text(
                col_index,
                row_index,
                f"{100 * value:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Noisy eval accuracy")
    save_current(output_dir / output_name)


def main() -> None:
    args = parse_args()
    summary_rows = read_csv(args.summary)
    gradient_rows = read_csv(args.gradient_diagnostics)
    checkpoint_rows = read_csv(args.checkpoint_sweep)
    layerwise_rows = read_csv(args.layerwise_sweep_summary)
    tinyimagenet_layerwise_rows = read_csv(args.tinyimagenet_layerwise_sweep_summary)
    current_best_rows = read_csv(args.current_best)
    main_uniform_rows = read_csv(args.main_uniform_summary)
    tinyimagenet_rows = read_csv(args.tinyimagenet_summary)
    optional_domain_rows: list[dict[str, str]] = []
    for path in args.optional_domain_summaries:
        optional_domain_rows.extend(read_csv(path))
    optional_internal_rows = read_csv(args.optional_internal_summary)

    plot_long_training(summary_rows, args.output_dir)
    plot_lr_sweep(summary_rows, args.output_dir)
    plot_gradient_diagnostics(gradient_rows, args.output_dir)
    checkpoint_summary = summarize_checkpoint_sweep(checkpoint_rows)
    write_checkpoint_summary(args.checkpoint_sweep_summary, checkpoint_summary)
    plot_checkpoint_sweep(checkpoint_summary, args.output_dir)
    plot_layerwise_sweep(layerwise_rows, args.output_dir)
    plot_layerwise_sweep(
        tinyimagenet_layerwise_rows,
        args.output_dir,
        output_name="tinyimagenet_efficientnet_b0_layerwise_noise_sweep.png",
        title="TinyImageNet EfficientNet-B0 layer-wise calibration",
        vmin=None,
        vmax=None,
    )
    plot_current_best(current_best_rows, args.output_dir)
    plot_main_uniform_noise(main_uniform_rows, args.output_dir)
    plot_tinyimagenet_summary(tinyimagenet_rows, args.output_dir)
    plot_optional_domain(optional_domain_rows, args.output_dir)
    plot_optional_internal(optional_internal_rows, args.output_dir)


if __name__ == "__main__":
    main()
