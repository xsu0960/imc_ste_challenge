#!/usr/bin/env python3
"""Screen foreground-only supervision on clean-aligned ROI proposals."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[179])
    parser.add_argument("--weights", nargs="+", type=float, default=[0.025, 0.05])
    parser.add_argument("--max-train-batches", type=int, default=200)
    parser.add_argument("--max-eval-batches", type=int, default=120)
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for seed in args.seeds:
        for weight in args.weights:
            weight_tag = str(weight).replace(".", "p")
            output = RUNS_DIR / (
                f"optional_detection_roi_foreground_w{weight_tag}_seed{seed}_"
                f"{args.max_train_batches}tr{args.max_eval_batches}ev.jsonl"
            )
            if (
                args.skip_existing
                and output.exists()
                and len(output.read_text().splitlines()) >= 2
            ):
                print(f"skip existing: {output.relative_to(PROJECT_ROOT)}")
                continue
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
                str(args.max_train_batches),
                "--max-eval-batches",
                str(args.max_eval_batches),
                "--detection-min-size",
                "320",
                "--detection-max-size",
                "512",
                "--proposal-roi-consistency",
                "--proposal-roi-objective",
                "foreground_target",
                "--proposal-roi-consistency-weight",
                str(weight),
                "--proposal-roi-consistency-box-weight",
                "1.0",
                "--no-download",
                "--no-save-checkpoint",
                "--stop-on-nonfinite",
                "--output",
                str(output),
            ]
            print(f"\n=== foreground ROI: weight {weight}, seed {seed} ===")
            print("$ " + " ".join(command))
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
