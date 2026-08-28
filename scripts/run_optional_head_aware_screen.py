#!/usr/bin/env python3
"""Run matched shared-read head-aware screening experiments."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


TASK_CONFIG = {
    "detection": {
        "checkpoint": "optional_detection_shared_read_state_ste_full_1epoch_seed89_best.pt",
        "statistics": "optional_detection_shared_read_state_layer_noise_stats_seed101_8b.csv",
        "roles": ("rpn", "roi"),
        "shape_args": ("--detection-min-size", "320", "--detection-max-size", "512"),
    },
    "segmentation": {
        "checkpoint": "optional_segmentation_shared_read_state_ste_full_1epoch_seed89_best.pt",
        "statistics": "optional_segmentation_shared_read_state_layer_noise_stats_seed101_8b.csv",
        "roles": ("classifier", "aux_classifier"),
        "shape_args": ("--segmentation-image-size", "256"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASK_CONFIG),
        default=list(TASK_CONFIG),
    )
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--max-train-batches", type=int, default=300)
    parser.add_argument("--max-eval-batches", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--task-consistency-weight", type=float, default=0.02)
    parser.add_argument("--activation-target", type=float, default=10.0)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def candidate_args(task: str, candidate: str, args: argparse.Namespace) -> list[str]:
    config = TASK_CONFIG[task]
    statistics = RUNS_DIR / config["statistics"]
    roles = list(config["roles"])
    if candidate == "control":
        return ["--mode", "sat_aware_ste"]
    if candidate == "role_gradient":
        return [
            "--mode",
            "variance_sat_aware_ste",
            "--gradient-stat-csv",
            str(statistics),
            "--gradient-stat-roles",
            *roles,
            "--gradient-stat-strength",
            "0.5",
            "--gradient-bias-strength",
            "0.25",
            "--gradient-stat-floor",
            "0.5",
        ]
    if candidate == "task_consistency":
        return [
            "--mode",
            "sat_aware_ste",
            "--task-output-consistency-weight",
            str(args.task_consistency_weight),
            "--task-output-consistency-temperature",
            "2.0",
            "--task-output-consistency-box-weight",
            "0.25",
            "--task-output-consistency-aux-weight",
            "0.4",
        ]
    if candidate == "head_range":
        return [
            "--mode",
            "sat_aware_ste",
            "--activation-stat-csv",
            str(statistics),
            "--activation-stat-roles",
            *roles,
            "--activation-target",
            str(args.activation_target),
            "--activation-scale-floor",
            "0.5",
        ]
    raise ValueError(f"unknown candidate: {candidate}")


def run_candidate(task: str, candidate: str, args: argparse.Namespace) -> None:
    config = TASK_CONFIG[task]
    output = RUNS_DIR / (
        f"optional_{task}_shared_read_head_screen_{candidate}_seed{args.seed}_"
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
        str(RUNS_DIR / config["checkpoint"]),
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
        *config["shape_args"],
        *candidate_args(task, candidate, args),
    ]
    print(f"\n=== {task}: {candidate} ===")
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    for task in args.tasks:
        for candidate in ("control", "role_gradient", "task_consistency", "head_range"):
            run_candidate(task, candidate, args)


if __name__ == "__main__":
    main()
