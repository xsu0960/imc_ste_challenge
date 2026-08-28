#!/usr/bin/env python3
"""Run paired multi-seed repeats for promoted optional head-aware candidates."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[137, 139])
    parser.add_argument("--max-train-batches", type=int, default=300)
    parser.add_argument("--max-eval-batches", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def task_configuration(task: str, variant: str) -> tuple[str, list[str], list[str]]:
    if task == "detection":
        checkpoint = "optional_detection_shared_read_state_ste_full_1epoch_seed89_best.pt"
        shape_args = ["--detection-min-size", "320", "--detection-max-size", "512"]
        if variant == "winner":
            extra = [
                "--mode",
                "variance_sat_aware_ste",
                "--gradient-stat-csv",
                str(
                    RUNS_DIR
                    / "optional_detection_shared_read_state_layer_noise_stats_seed101_8b.csv"
                ),
                "--gradient-stat-roles",
                "rpn",
                "roi",
                "--gradient-stat-strength",
                "0.5",
                "--gradient-bias-strength",
                "0.25",
                "--gradient-stat-floor",
                "0.5",
            ]
        else:
            extra = ["--mode", "sat_aware_ste"]
        return checkpoint, shape_args, extra

    checkpoint = "optional_segmentation_shared_read_state_ste_full_1epoch_seed89_best.pt"
    shape_args = ["--segmentation-image-size", "256"]
    if variant == "winner":
        extra = [
            "--mode",
            "sat_aware_ste",
            "--task-output-consistency-weight",
            "0.005",
            "--activation-stat-csv",
            str(
                RUNS_DIR
                / "optional_segmentation_shared_read_state_layer_noise_stats_seed101_8b.csv"
            ),
            "--activation-stat-roles",
            "classifier",
            "aux_classifier",
            "--activation-target",
            "10",
            "--activation-scale-floor",
            "0.5",
        ]
    else:
        extra = ["--mode", "sat_aware_ste"]
    return checkpoint, shape_args, extra


def run(task: str, variant: str, seed: int, args: argparse.Namespace) -> None:
    checkpoint, shape_args, extra = task_configuration(task, variant)
    output = RUNS_DIR / (
        f"optional_{task}_shared_read_head_repeat_{variant}_seed{seed}_"
        f"{args.max_train_batches}tr{args.max_eval_batches}ev.jsonl"
    )
    if args.skip_existing and output.exists() and len(output.read_text().splitlines()) >= 2:
        print(f"skip existing: {output.relative_to(PROJECT_ROOT)}")
        return
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_optional_ste.py"),
        "--task",
        task,
        "--eval-mode",
        "noise",
        "--epochs",
        "1",
        "--checkpoint",
        str(RUNS_DIR / checkpoint),
        "--noise-scale",
        "1.0",
        "--batch-size",
        "1",
        "--seed",
        str(seed),
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
    print(f"\n=== {task}: {variant}, seed {seed} ===")
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    for seed in args.seeds:
        for task in ("detection", "segmentation"):
            for variant in ("control", "winner"):
                run(task, variant, seed, args)


if __name__ == "__main__":
    main()
