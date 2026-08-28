#!/usr/bin/env python3
"""Run paired multi-seed repeats for promoted next-generation candidates."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
FAMILIES = ("detection_online", "detection_roi", "segmentation_combo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[157, 163])
    parser.add_argument("--max-train-batches", type=int, default=300)
    parser.add_argument("--max-eval-batches", type=int, default=200)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def family_config(family: str) -> dict[str, object]:
    if family == "detection_online":
        return {
            "task": "detection",
            "checkpoint": "optional_detection_shared_read_state_ste_full_1epoch_seed89_best.pt",
            "shape": ("--detection-min-size", "320", "--detection-max-size", "512"),
            "control": (
                "--mode",
                "variance_sat_aware_ste",
                "--online-gradient-profile",
                "--online-gradient-roles",
                "rpn",
                "--online-gradient-variance-strength",
                "0.0",
                "--online-gradient-bias-strength",
                "0.0",
                "--online-gradient-scale-floor",
                "1.0",
                "--online-gradient-warmup-updates",
                "16",
            ),
            "winner": (
                "--mode",
                "variance_sat_aware_ste",
                "--online-gradient-profile",
                "--online-gradient-roles",
                "rpn",
                "--online-gradient-variance-strength",
                "0.1",
                "--online-gradient-bias-strength",
                "0.1",
                "--online-gradient-scale-floor",
                "0.75",
                "--online-gradient-warmup-updates",
                "16",
            ),
        }
    if family == "detection_roi":
        return {
            "task": "detection",
            "checkpoint": "optional_detection_shared_read_state_ste_full_1epoch_seed89_best.pt",
            "shape": ("--detection-min-size", "320", "--detection-max-size", "512"),
            "control": (
                "--mode",
                "sat_aware_ste",
                "--proposal-roi-consistency",
                "--proposal-roi-consistency-weight",
                "0.0",
            ),
            "winner": (
                "--mode",
                "sat_aware_ste",
                "--proposal-roi-consistency",
                "--proposal-roi-consistency-weight",
                "0.0025",
            ),
        }
    return {
        "task": "segmentation",
        "checkpoint": "optional_segmentation_shared_read_state_ste_full_1epoch_seed89_best.pt",
        "shape": ("--segmentation-image-size", "256"),
        "control": (
            "--mode",
            "variance_sat_aware_ste",
            "--online-gradient-profile",
            "--online-gradient-roles",
            "classifier",
            "aux_classifier",
            "--online-gradient-variance-strength",
            "0.0",
            "--online-gradient-bias-strength",
            "0.0",
            "--online-gradient-scale-floor",
            "1.0",
            "--online-gradient-warmup-updates",
            "16",
        ),
        "winner": (
            "--mode",
            "variance_sat_aware_ste",
            "--online-gradient-profile",
            "--online-gradient-roles",
            "classifier",
            "aux_classifier",
            "--online-gradient-variance-strength",
            "0.5",
            "--online-gradient-bias-strength",
            "0.25",
            "--online-gradient-scale-floor",
            "0.5",
            "--online-gradient-warmup-updates",
            "16",
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
            "10.0",
            "--activation-scale-floor",
            "0.5",
        ),
    }


def run_variant(
    family: str, variant: str, seed: int, args: argparse.Namespace
) -> None:
    config = family_config(family)
    task = str(config["task"])
    output = RUNS_DIR / (
        f"optional_{family}_nextgen_repeat_{variant}_seed{seed}_"
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
        str(RUNS_DIR / str(config["checkpoint"])),
        "--noise-scale",
        "1.0",
        "--batch-size",
        "1",
        "--seed",
        str(seed),
        "--learning-rate",
        "0.0001",
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
        *config["shape"],
        *config[variant],
    ]
    print(f"\n=== {family}: {variant}, seed {seed} ===")
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    for family in args.families:
        for seed in args.seeds:
            for variant in ("control", "winner"):
                run_variant(family, variant, seed, args)


if __name__ == "__main__":
    main()
