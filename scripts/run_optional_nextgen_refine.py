#!/usr/bin/env python3
"""Refine online profiles and proposal-aligned ROI consistency."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"

CHECKPOINTS = {
    "detection": RUNS_DIR
    / "optional_detection_shared_read_state_ste_full_1epoch_seed89_best.pt",
    "segmentation": RUNS_DIR
    / "optional_segmentation_shared_read_state_ste_full_1epoch_seed89_best.pt",
}
STATISTICS = {
    "segmentation": RUNS_DIR
    / "optional_segmentation_shared_read_state_layer_noise_stats_seed101_8b.csv",
}

DETECTION_CANDIDATES = (
    "online_variance_all",
    "online_soft_floor85",
    "online_soft_shared_conv",
    "roi_control",
    "roi_w0p0025",
    "roi_w0p005",
    "roi_w0p01",
)
SEGMENTATION_CANDIDATES = (
    "online_static_task_range",
    "online_output_heads",
    "online_no_global_pool",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("detection", "segmentation"),
        default=["detection", "segmentation"],
    )
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--max-train-batches", type=int, default=300)
    parser.add_argument("--max-eval-batches", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def online_args(
    *,
    variance: float,
    bias: float,
    floor: float,
    roles: tuple[str, ...] = (),
    prefixes: tuple[str, ...] = (),
) -> list[str]:
    selection = (
        ["--online-gradient-name-prefixes", *prefixes]
        if prefixes
        else ["--online-gradient-roles", *roles]
    )
    return [
        "--mode",
        "variance_sat_aware_ste",
        "--online-gradient-profile",
        *selection,
        "--online-gradient-ema-decay",
        "0.95",
        "--online-gradient-variance-strength",
        str(variance),
        "--online-gradient-bias-strength",
        str(bias),
        "--online-gradient-scale-floor",
        str(floor),
        "--online-gradient-warmup-updates",
        "16",
    ]


def candidate_args(task: str, candidate: str) -> list[str]:
    if task == "detection":
        if candidate == "online_variance_all":
            return online_args(variance=0.1, bias=0.0, floor=0.75, roles=("rpn",))
        if candidate == "online_soft_floor85":
            return online_args(variance=0.1, bias=0.1, floor=0.85, roles=("rpn",))
        if candidate == "online_soft_shared_conv":
            return online_args(
                variance=0.1,
                bias=0.1,
                floor=0.75,
                prefixes=("rpn.head.conv.0.0",),
            )
        if candidate == "roi_control":
            weight = "0.0"
        elif candidate == "roi_w0p0025":
            weight = "0.0025"
        elif candidate == "roi_w0p005":
            weight = "0.005"
        elif candidate == "roi_w0p01":
            weight = "0.01"
        else:
            raise ValueError(f"unknown detection candidate: {candidate}")
        return [
            "--mode",
            "sat_aware_ste",
            "--proposal-roi-consistency",
            "--proposal-roi-consistency-weight",
            weight,
            "--proposal-roi-consistency-temperature",
            "2.0",
            "--proposal-roi-consistency-box-weight",
            "0.25",
        ]

    if candidate == "online_static_task_range":
        return [
            *online_args(
                variance=0.5,
                bias=0.25,
                floor=0.5,
                roles=("classifier", "aux_classifier"),
            ),
            "--task-output-consistency-weight",
            "0.005",
            "--activation-stat-csv",
            str(STATISTICS["segmentation"]),
            "--activation-stat-roles",
            "classifier",
            "aux_classifier",
            "--activation-target",
            "10.0",
            "--activation-scale-floor",
            "0.5",
        ]
    if candidate == "online_output_heads":
        return online_args(
            variance=0.5,
            bias=0.25,
            floor=0.5,
            prefixes=("classifier.4", "aux_classifier.4"),
        )
    if candidate == "online_no_global_pool":
        return online_args(
            variance=0.5,
            bias=0.25,
            floor=0.5,
            prefixes=(
                "classifier.0.convs.0.0",
                "classifier.0.convs.1.0",
                "classifier.0.convs.2.0",
                "classifier.0.convs.3.0",
                "classifier.0.project.0",
                "classifier.1",
                "classifier.4",
                "aux_classifier.",
            ),
        )
    raise ValueError(f"unknown segmentation candidate: {candidate}")


def run_candidate(task: str, candidate: str, args: argparse.Namespace) -> None:
    output = RUNS_DIR / (
        f"optional_{task}_nextgen_refine_{candidate}_seed{args.seed}_"
        f"{args.max_train_batches}tr{args.max_eval_batches}ev.jsonl"
    )
    if args.skip_existing and output.exists() and len(output.read_text().splitlines()) >= 2:
        print(f"skip existing: {output.relative_to(PROJECT_ROOT)}")
        return
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
        "--eval-mode",
        "noise",
        "--epochs",
        "1",
        "--checkpoint",
        str(CHECKPOINTS[task]),
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
        *candidate_args(task, candidate),
    ]
    print(f"\n=== {task}: {candidate} ===")
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    for task in args.tasks:
        candidates = (
            DETECTION_CANDIDATES if task == "detection" else SEGMENTATION_CANDIDATES
        )
        for candidate in candidates:
            run_candidate(task, candidate, args)


if __name__ == "__main__":
    main()
