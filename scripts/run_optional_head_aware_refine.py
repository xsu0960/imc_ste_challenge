#!/usr/bin/env python3
"""Refine the positive shared-read head-aware screening candidates."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


DETECTION_VARIANTS = {
    "gradient_v025_b025": ("0.25", "0.25"),
    "gradient_v05_b0": ("0.5", "0.0"),
    "gradient_v1_b025": ("1.0", "0.25"),
}

SEGMENTATION_VARIANTS = {
    "task_w0005": ["--task-output-consistency-weight", "0.005"],
    "task_w001": ["--task-output-consistency-weight", "0.01"],
    "task_w005": ["--task-output-consistency-weight", "0.05"],
    "range_t12": [
        "--activation-stat-csv",
        str(RUNS_DIR / "optional_segmentation_shared_read_state_layer_noise_stats_seed101_8b.csv"),
        "--activation-stat-roles",
        "classifier",
        "aux_classifier",
        "--activation-target",
        "12",
        "--activation-scale-floor",
        "0.5",
    ],
    "task_w002_range_t10": [
        "--task-output-consistency-weight",
        "0.02",
        "--activation-stat-csv",
        str(RUNS_DIR / "optional_segmentation_shared_read_state_layer_noise_stats_seed101_8b.csv"),
        "--activation-stat-roles",
        "classifier",
        "aux_classifier",
        "--activation-target",
        "10",
        "--activation-scale-floor",
        "0.5",
    ],
    "task_w0005_range_t10": [
        "--task-output-consistency-weight",
        "0.005",
        "--activation-stat-csv",
        str(RUNS_DIR / "optional_segmentation_shared_read_state_layer_noise_stats_seed101_8b.csv"),
        "--activation-stat-roles",
        "classifier",
        "aux_classifier",
        "--activation-target",
        "10",
        "--activation-scale-floor",
        "0.5",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--max-train-batches", type=int, default=300)
    parser.add_argument("--max-eval-batches", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def common_command(
    task: str, variant: str, args: argparse.Namespace, extra: list[str]
) -> tuple[list[str], Path]:
    checkpoint = RUNS_DIR / (
        "optional_detection_shared_read_state_ste_full_1epoch_seed89_best.pt"
        if task == "detection"
        else "optional_segmentation_shared_read_state_ste_full_1epoch_seed89_best.pt"
    )
    output = RUNS_DIR / (
        f"optional_{task}_shared_read_head_refine_{variant}_seed{args.seed}_"
        f"{args.max_train_batches}tr{args.max_eval_batches}ev.jsonl"
    )
    shape_args = (
        ["--detection-min-size", "320", "--detection-max-size", "512"]
        if task == "detection"
        else ["--segmentation-image-size", "256"]
    )
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_optional_ste.py"),
        "--task",
        task,
        "--mode",
        "variance_sat_aware_ste" if task == "detection" else "sat_aware_ste",
        "--eval-mode",
        "noise",
        "--epochs",
        "1",
        "--checkpoint",
        str(checkpoint),
        "--noise-scale",
        "1.0",
        "--batch-size",
        "1",
        "--seed",
        str(args.seed),
        "--learning-rate",
        str(args.learning_rate),
        "--grad-clip-norm",
        "1.0",
        "--conv-chunk-rows",
        "8",
        "--conv-weight-noise-scope",
        "read",
        "--max-train-batches",
        str(args.max_train_batches),
        "--max-eval-batches",
        str(args.max_eval_batches),
        "--no-download",
        "--no-save-checkpoint",
        "--stop-on-nonfinite",
        "--output",
        str(output),
        *shape_args,
        *extra,
    ]
    return command, output


def run(task: str, variant: str, args: argparse.Namespace, extra: list[str]) -> None:
    command, output = common_command(task, variant, args, extra)
    if args.skip_existing and output.exists() and len(output.read_text().splitlines()) >= 2:
        print(f"skip existing: {output.relative_to(PROJECT_ROOT)}")
        return
    print(f"\n=== {task}: {variant} ===")
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    statistics = RUNS_DIR / (
        "optional_detection_shared_read_state_layer_noise_stats_seed101_8b.csv"
    )
    for variant, (variance_strength, bias_strength) in DETECTION_VARIANTS.items():
        run(
            "detection",
            variant,
            args,
            [
                "--gradient-stat-csv",
                str(statistics),
                "--gradient-stat-roles",
                "rpn",
                "roi",
                "--gradient-stat-strength",
                variance_strength,
                "--gradient-bias-strength",
                bias_strength,
                "--gradient-stat-floor",
                "0.5",
            ],
        )
    for variant, extra in SEGMENTATION_VARIANTS.items():
        run("segmentation", variant, args, extra)


if __name__ == "__main__":
    main()
