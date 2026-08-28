#!/usr/bin/env python3
"""Run matched next-generation online-gradient-profile screening experiments."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"

TASK_CONFIG = {
    "detection": {
        "checkpoint": "optional_detection_shared_read_state_ste_full_1epoch_seed89_best.pt",
        "roles": ("rpn",),
        "shape_args": ("--detection-min-size", "320", "--detection-max-size", "512"),
    },
    "segmentation": {
        "checkpoint": "optional_segmentation_shared_read_state_ste_full_1epoch_seed89_best.pt",
        "roles": ("classifier", "aux_classifier"),
        "shape_args": ("--segmentation-image-size", "256"),
    },
}

CANDIDATES = {
    "dualpass_control": {
        "variance": 0.0,
        "bias": 0.0,
        "floor": 1.0,
    },
    "online_bias": {
        "variance": 0.0,
        "bias": 0.25,
        "floor": 0.5,
    },
    "online_soft": {
        "variance": 0.1,
        "bias": 0.1,
        "floor": 0.75,
    },
    "online_balanced": {
        "variance": 0.25,
        "bias": 0.25,
        "floor": 0.5,
    },
    "online_staticmatch": {
        "variance": 0.5,
        "bias": 0.25,
        "floor": 0.5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+", choices=tuple(TASK_CONFIG), default=list(TASK_CONFIG)
    )
    parser.add_argument(
        "--candidates", nargs="+", choices=tuple(CANDIDATES), default=list(CANDIDATES)
    )
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--max-train-batches", type=int, default=300)
    parser.add_argument("--max-eval-batches", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.95)
    parser.add_argument("--warmup-updates", type=int, default=16)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def run_candidate(task: str, candidate: str, args: argparse.Namespace) -> None:
    config = TASK_CONFIG[task]
    profile = CANDIDATES[candidate]
    output = RUNS_DIR / (
        f"optional_{task}_online_profile_screen_{candidate}_seed{args.seed}_"
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
        "--mode",
        "variance_sat_aware_ste",
        "--eval-mode",
        "noise",
        "--epochs",
        "1",
        "--checkpoint",
        str(RUNS_DIR / config["checkpoint"]),
        "--online-gradient-profile",
        "--online-gradient-roles",
        *config["roles"],
        "--online-gradient-ema-decay",
        str(args.ema_decay),
        "--online-gradient-variance-strength",
        str(profile["variance"]),
        "--online-gradient-bias-strength",
        str(profile["bias"]),
        "--online-gradient-scale-floor",
        str(profile["floor"]),
        "--online-gradient-scale-ceiling",
        "1.0",
        "--online-gradient-warmup-updates",
        str(args.warmup_updates),
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
    ]
    print(f"\n=== {task}: {candidate} ===")
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    for task in args.tasks:
        for candidate in args.candidates:
            run_candidate(task, candidate, args)


if __name__ == "__main__":
    main()
