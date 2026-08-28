#!/usr/bin/env python3
"""Run paired repeats for target-supervised clean-aligned ROI training."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[179, 181])
    parser.add_argument("--max-train-batches", type=int, default=200)
    parser.add_argument("--max-eval-batches", type=int, default=120)
    parser.add_argument("--weight", type=float, default=0.05)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("control", "winner"),
        default=["control", "winner"],
    )
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def run_variant(
    variant: str,
    seed: int,
    weight: float,
    max_train_batches: int,
    max_eval_batches: int,
    skip_existing: bool,
) -> None:
    weight_tag = "" if weight == 0.05 else f"_w{str(weight).replace('.', 'p')}"
    output = RUNS_DIR / (
        f"optional_detection_roi_target_repeat_{variant}{weight_tag}_seed{seed}_"
        f"{max_train_batches}tr{max_eval_batches}ev.jsonl"
    )
    if skip_existing and output.exists() and len(output.read_text().splitlines()) >= 2:
        print(f"skip existing: {output.relative_to(PROJECT_ROOT)}")
        return
    auxiliary_weight = 0.0 if variant == "control" else weight
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_optional_ste.py"),
        "--task",
        "detection",
        "--mode",
        "sat_aware_ste",
        "--eval-mode",
        "noise",
        "--epochs",
        "1",
        "--checkpoint",
        str(
            RUNS_DIR
            / "optional_detection_shared_read_state_ste_full_1epoch_seed89_best.pt"
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
        "--max-train-batches",
        str(max_train_batches),
        "--max-eval-batches",
        str(max_eval_batches),
        "--detection-min-size",
        "320",
        "--detection-max-size",
        "512",
        "--proposal-roi-consistency",
        "--proposal-roi-objective",
        "target",
        "--proposal-roi-consistency-weight",
        str(auxiliary_weight),
        "--proposal-roi-consistency-box-weight",
        "1.0",
        "--no-download",
        "--no-save-checkpoint",
        "--stop-on-nonfinite",
        "--output",
        str(output),
    ]
    print(f"\n=== ROI target repeat: {variant}, seed {seed} ===")
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    for seed in args.seeds:
        for variant in args.variants:
            run_variant(
                variant,
                seed,
                args.weight,
                args.max_train_batches,
                args.max_eval_batches,
                args.skip_existing,
            )


if __name__ == "__main__":
    main()
