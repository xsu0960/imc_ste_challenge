#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

TASKS = {
    "cifar10_resnet18": {
        "dataset": "cifar10",
        "config": PROJECT_ROOT / "configs/cifar10_resnet18_formal.yaml",
    },
    "cifar10_efficientnet_b0": {
        "dataset": "cifar10",
        "config": PROJECT_ROOT / "configs/cifar10_efficientnet_b0_formal.yaml",
    },
    "cifar100_resnet18": {
        "dataset": "cifar100",
        "config": PROJECT_ROOT / "configs/cifar100_resnet18.yaml",
    },
    "cifar100_efficientnet_b0": {
        "dataset": "cifar100",
        "config": PROJECT_ROOT / "configs/cifar100_efficientnet_b0.yaml",
    },
    "tinyimagenet_resnet18": {
        "dataset": "tinyimagenet",
        "config": PROJECT_ROOT / "configs/tinyimagenet_resnet18.yaml",
    },
    "tinyimagenet_resnet18_imagenet224": {
        "dataset": "tinyimagenet",
        "config": PROJECT_ROOT / "configs/tinyimagenet_resnet18_imagenet224.yaml",
    },
    "tinyimagenet_efficientnet_b0": {
        "dataset": "tinyimagenet",
        "config": PROJECT_ROOT / "configs/tinyimagenet_efficientnet_b0.yaml",
    },
    "tinyimagenet_efficientnet_b0_imagenet224": {
        "dataset": "tinyimagenet",
        "config": PROJECT_ROOT / "configs/tinyimagenet_efficientnet_b0_imagenet224.yaml",
    },
}

PROFILES = {
    "smoke": {
        "epochs": 1,
        "max_train_batches": 2,
        "max_eval_batches": 2,
        "seeds": [1],
        "eval_every": 1,
    },
    "pilot": {
        "epochs": 10,
        "max_train_batches": 100,
        "max_eval_batches": 40,
        "seeds": [1],
        "eval_every": 1,
    },
    "formal": {
        "epochs": 120,
        "max_train_batches": 0,
        "max_eval_batches": 0,
        "seeds": [1, 2, 3],
        "eval_every": 5,
    },
}

DEFAULT_MODES = (
    "clean",
    "noise",
    "ste",
    "sat_aware_ste",
    "adaptive_sat_aware_ste",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the formal CIFAR classification task matrix."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS.keys(),
        default=list(TASKS.keys()),
    )
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--profile", choices=PROFILES.keys(), default="formal")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    return parser.parse_args()


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_ROOT) if not existing else f"{SRC_ROOT}:{existing}"
    return env


def profile_value(args, key: str):
    override_name = key.replace("_", "-")
    value = getattr(args, key, None)
    if value is not None:
        return value
    return PROFILES[args.profile][key]


def run_command(command: list[str], env: dict[str, str], dry_run: bool) -> int:
    print("\n$ " + " ".join(str(item) for item in command), flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    return completed.returncode


def main() -> int:
    args = parse_args()
    env = build_env()
    seeds = args.seeds if args.seeds is not None else PROFILES[args.profile]["seeds"]
    epochs = profile_value(args, "epochs")
    max_train_batches = profile_value(args, "max_train_batches")
    max_eval_batches = profile_value(args, "max_eval_batches")
    eval_every = profile_value(args, "eval_every")
    failures = 0

    for task_name in args.tasks:
        task = TASKS[task_name]
        print(f"\n=== formal task: {task_name} ===", flush=True)
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_matrix.py"),
            "--config",
            str(task["config"]),
            "--dataset",
            task["dataset"],
            "--modes",
            *args.modes,
            "--seeds",
            *(str(seed) for seed in seeds),
            "--epochs",
            str(epochs),
            "--max-train-batches",
            str(max_train_batches),
            "--max-eval-batches",
            str(max_eval_batches),
            "--eval-every",
            str(eval_every),
            "--grad-clip-norm",
            str(args.grad_clip_norm),
            "--noise-scale",
            str(args.noise_scale),
            "--stop-on-nonfinite",
            "--continue-on-error",
        ]
        if args.learning_rate is not None:
            command.extend(["--learning-rate", str(args.learning_rate)])
        if args.no_aggregate:
            command.append("--no-aggregate")
        returncode = run_command(command, env, args.dry_run)
        if returncode != 0:
            failures += 1
            if not args.continue_on_error:
                return returncode

    if not args.no_aggregate:
        returncode = run_command(
            [sys.executable, str(PROJECT_ROOT / "scripts/aggregate_runs.py")],
            env,
            args.dry_run,
        )
        if returncode != 0:
            return returncode

    if failures:
        print(f"\nCompleted with {failures} failed formal task(s).", flush=True)
        return 1
    print("\nFormal task matrix completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
