#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from imc_ste import COMPUTE_MODES


DEFAULT_MODES = ("clean", "sat_aware_ste", "adaptive_sat_aware_ste")
DATASETS = ("fake", "cifar10", "cifar100", "tinyimagenet")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a repeatable training matrix and aggregate the results."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/cifar10_resnet18.yaml",
    )
    parser.add_argument("--dataset", choices=DATASETS, default="cifar10")
    parser.add_argument("--modes", nargs="+", choices=COMPUTE_MODES, default=DEFAULT_MODES)
    parser.add_argument("--eval-mode", choices=COMPUTE_MODES, default="noise")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=100)
    parser.add_argument("--max-eval-batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine", "multistep"],
    )
    parser.add_argument("--lr-milestones", nargs="+", type=int)
    parser.add_argument("--lr-gamma", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--augmentation", choices=["standard", "autoaugment"])
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--noise-scale", type=float)
    parser.add_argument("--depthwise-noise-scale", type=float)
    parser.add_argument("--pointwise-noise-scale", type=float)
    parser.add_argument("--linear-noise-scale", type=float)
    parser.add_argument("--stop-on-nonfinite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later jobs if one job fails.",
    )
    return parser.parse_args()


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_ROOT) if not existing else f"{SRC_ROOT}:{existing}"
    return env


def build_train_command(args, mode: str, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train.py"),
        "--config",
        str(args.config),
        "--dataset",
        args.dataset,
        "--mode",
        mode,
        "--eval-mode",
        args.eval_mode,
        "--epochs",
        str(args.epochs),
        "--max-train-batches",
        str(args.max_train_batches),
        "--max-eval-batches",
        str(args.max_eval_batches),
        "--seed",
        str(seed),
    ]
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.learning_rate is not None:
        command.extend(["--learning-rate", str(args.learning_rate)])
    if args.momentum is not None:
        command.extend(["--momentum", str(args.momentum)])
    if args.weight_decay is not None:
        command.extend(["--weight-decay", str(args.weight_decay)])
    if args.label_smoothing is not None:
        command.extend(["--label-smoothing", str(args.label_smoothing)])
    if args.lr_scheduler is not None:
        command.extend(["--lr-scheduler", args.lr_scheduler])
    if args.lr_milestones is not None:
        command.append("--lr-milestones")
        command.extend(str(milestone) for milestone in args.lr_milestones)
    if args.lr_gamma is not None:
        command.extend(["--lr-gamma", str(args.lr_gamma)])
    if args.num_workers is not None:
        command.extend(["--num-workers", str(args.num_workers)])
    if args.augmentation is not None:
        command.extend(["--augmentation", args.augmentation])
    if args.eval_every is not None:
        command.extend(["--eval-every", str(args.eval_every)])
    if args.grad_clip_norm is not None:
        command.extend(["--grad-clip-norm", str(args.grad_clip_norm)])
    if args.noise_scale is not None:
        command.extend(["--noise-scale", str(args.noise_scale)])
    if args.depthwise_noise_scale is not None:
        command.extend(["--depthwise-noise-scale", str(args.depthwise_noise_scale)])
    if args.pointwise_noise_scale is not None:
        command.extend(["--pointwise-noise-scale", str(args.pointwise_noise_scale)])
    if args.linear_noise_scale is not None:
        command.extend(["--linear-noise-scale", str(args.linear_noise_scale)])
    if args.stop_on_nonfinite:
        command.append("--stop-on-nonfinite")
    return command


def run_command(command: list[str], env: dict[str, str], dry_run: bool) -> int:
    printable = " ".join(command)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    return completed.returncode


def main() -> int:
    args = parse_args()
    env = build_env()
    total = len(args.modes) * len(args.seeds)
    failures = 0

    print(
        f"Running {total} jobs: dataset={args.dataset}, "
        f"eval_mode={args.eval_mode}, epochs={args.epochs}",
        flush=True,
    )
    for seed in args.seeds:
        for mode in args.modes:
            print(f"\n=== mode={mode} seed={seed} ===", flush=True)
            command = build_train_command(args, mode, seed)
            returncode = run_command(command, env, args.dry_run)
            if returncode != 0:
                failures += 1
                print(
                    f"Job failed with exit code {returncode}: mode={mode} seed={seed}",
                    flush=True,
                )
                if not args.continue_on_error:
                    return returncode

    if not args.no_aggregate:
        print("\n=== aggregate ===", flush=True)
        aggregate_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/aggregate_runs.py"),
        ]
        returncode = run_command(aggregate_command, env, args.dry_run)
        if returncode != 0:
            return returncode

    if failures:
        print(f"\nCompleted with {failures} failed job(s).", flush=True)
        return 1
    print("\nAll jobs completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
