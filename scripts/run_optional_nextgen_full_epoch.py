#!/usr/bin/env python3
"""Run the promoted segmentation next-generation candidate for a full epoch."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=167)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def run_variant(variant: str, seed: int, skip_existing: bool) -> None:
    output = RUNS_DIR / (
        f"optional_segmentation_nextgen_full_epoch_{variant}_seed{seed}.jsonl"
    )
    if skip_existing and output.exists() and len(output.read_text().splitlines()) >= 2:
        print(f"skip existing: {output.relative_to(PROJECT_ROOT)}")
        return

    shared = [
        "--mode",
        "variance_sat_aware_ste",
        "--online-gradient-profile",
        "--online-gradient-roles",
        "classifier",
        "aux_classifier",
        "--online-gradient-warmup-updates",
        "16",
    ]
    if variant == "control":
        variant_args = [
            *shared,
            "--online-gradient-variance-strength",
            "0.0",
            "--online-gradient-bias-strength",
            "0.0",
            "--online-gradient-scale-floor",
            "1.0",
        ]
    else:
        variant_args = [
            *shared,
            "--online-gradient-variance-strength",
            "0.5",
            "--online-gradient-bias-strength",
            "0.25",
            "--online-gradient-scale-floor",
            "0.5",
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
        ]

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_optional_ste.py"),
        "--task",
        "segmentation",
        "--eval-mode",
        "noise",
        "--epochs",
        "1",
        "--checkpoint",
        str(
            RUNS_DIR
            / "optional_segmentation_shared_read_state_ste_full_1epoch_seed89_best.pt"
        ),
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
        "--segmentation-image-size",
        "256",
        "--no-download",
        "--stop-on-nonfinite",
        "--output",
        str(output),
        *(["--no-save-checkpoint"] if variant == "control" else []),
        *variant_args,
    ]
    print(f"\n=== full epoch segmentation: {variant}, seed {seed} ===")
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    for variant in ("control", "winner"):
        run_variant(variant, args.seed, args.skip_existing)


if __name__ == "__main__":
    main()
