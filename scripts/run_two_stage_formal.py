#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
RUNS_DIR = PROJECT_ROOT / "runs"

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run clean pretraining followed by noisy STE fine-tuning."
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=TASKS.keys(), default=list(TASKS.keys())
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--clean-epochs", type=int, default=120)
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument(
        "--finetune-modes",
        nargs="+",
        default=["sat_aware_ste", "adaptive_sat_aware_ste"],
    )
    parser.add_argument("--clean-learning-rate", type=float)
    parser.add_argument("--finetune-learning-rate", type=float, default=0.005)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--depthwise-noise-scale", type=float)
    parser.add_argument("--pointwise-noise-scale", type=float)
    parser.add_argument("--linear-noise-scale", type=float)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--clean-eval-every", type=int, default=5)
    parser.add_argument("--finetune-eval-every", type=int, default=2)
    parser.add_argument("--eval-repeats", type=int, default=5)
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix for repeat-eval CSV names, useful for LR sweeps.",
    )
    parser.add_argument("--reuse-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    return parser.parse_args()


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_ROOT) if not existing else f"{SRC_ROOT}:{existing}"
    return env


def run_command(command: list[str], env: dict[str, str], dry_run: bool) -> int:
    print("\n$ " + " ".join(str(item) for item in command), flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    return completed.returncode


def read_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {}
    epochs: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "metadata" in record:
            metadata = record["metadata"]
        elif "epoch" in record:
            epochs.append(record)
    return metadata, epochs


def metric_value(record: dict[str, Any], path: str = "eval.accuracy") -> float:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return float("-inf")
        current = current[part]
    if current is None:
        return float("-inf")
    try:
        return float(current)
    except (TypeError, ValueError):
        return float("-inf")


def best_metric(path: Path) -> float:
    _, epochs = read_jsonl(path)
    if not epochs:
        return float("-inf")
    return max(metric_value(epoch) for epoch in epochs)


def model_name_from_config(config_path: Path) -> str:
    for line in config_path.read_text().splitlines():
        if line.strip().startswith("name:"):
            return line.split(":", 1)[1].strip()
    return ""


def best_checkpoint_for_run(metrics_path: Path) -> Path:
    best_path = metrics_path.with_name(f"{metrics_path.stem}_best.pt")
    if best_path.exists():
        return best_path
    return metrics_path.with_suffix(".pt")


def eval_mode_for_finetune_mode(mode: str) -> str:
    if mode.startswith("dw_clean_"):
        return "dw_clean_noise"
    return "noise"


def same_config_path(recorded_config: str, config_path: Path) -> bool:
    if not recorded_config:
        return False
    recorded = Path(recorded_config)
    if not recorded.is_absolute():
        recorded = PROJECT_ROOT / recorded
    try:
        return recorded.resolve() == config_path.resolve()
    except FileNotFoundError:
        return recorded.absolute() == config_path.absolute()


def find_best_clean_checkpoint(
    dataset: str, model_name: str, seed: int, min_epochs: int, config_path: Path
) -> Path | None:
    candidates: list[tuple[float, float, Path]] = []
    for metrics_path in RUNS_DIR.glob("*.jsonl"):
        metadata, epochs = read_jsonl(metrics_path)
        model = metadata.get("model", {})
        if metadata.get("dataset") != dataset:
            continue
        if not same_config_path(str(metadata.get("config", "")), config_path):
            continue
        if model.get("name") != model_name:
            continue
        if metadata.get("seed") != seed:
            continue
        if metadata.get("train_mode") != "clean" or metadata.get("eval_mode") != "clean":
            continue
        if len(epochs) < min_epochs:
            continue
        checkpoint = best_checkpoint_for_run(metrics_path)
        if checkpoint.exists():
            candidates.append((best_metric(metrics_path), metrics_path.stat().st_mtime, checkpoint))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def latest_metrics_path(since: float, dataset: str, mode: str, seed: int) -> Path | None:
    candidates: list[tuple[float, Path]] = []
    for metrics_path in RUNS_DIR.glob("*.jsonl"):
        if metrics_path.stat().st_mtime < since:
            continue
        metadata, _ = read_jsonl(metrics_path)
        if metadata.get("dataset") == dataset and metadata.get("train_mode") == mode and metadata.get("seed") == seed:
            candidates.append((metrics_path.stat().st_mtime, metrics_path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def train_clean(args, task: dict[str, Any], env: dict[str, str]) -> Path | None:
    dataset = task["dataset"]
    config = task["config"]
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train.py"),
        "--config",
        str(config),
        "--dataset",
        dataset,
        "--mode",
        "clean",
        "--eval-mode",
        "clean",
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.clean_epochs),
        "--max-train-batches",
        str(args.max_train_batches),
        "--max-eval-batches",
        str(args.max_eval_batches),
        "--eval-every",
        str(args.clean_eval_every),
        "--noise-scale",
        str(args.noise_scale),
        "--stop-on-nonfinite",
    ]
    if args.clean_learning_rate is not None:
        command.extend(["--learning-rate", str(args.clean_learning_rate)])
    since = time.time()
    returncode = run_command(command, env, args.dry_run)
    if returncode != 0:
        return None
    if args.dry_run:
        return Path("DRY_RUN_CLEAN_CHECKPOINT.pt")
    metrics_path = latest_metrics_path(since, dataset, "clean", args.seed)
    return best_checkpoint_for_run(metrics_path) if metrics_path else None


def train_finetune(
    args,
    task: dict[str, Any],
    clean_checkpoint: Path,
    mode: str,
    env: dict[str, str],
) -> Path | None:
    dataset = task["dataset"]
    config = task["config"]
    eval_mode = eval_mode_for_finetune_mode(mode)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train.py"),
        "--config",
        str(config),
        "--dataset",
        dataset,
        "--mode",
        mode,
        "--eval-mode",
        eval_mode,
        "--checkpoint",
        str(clean_checkpoint),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.finetune_epochs),
        "--learning-rate",
        str(args.finetune_learning_rate),
        "--max-train-batches",
        str(args.max_train_batches),
        "--max-eval-batches",
        str(args.max_eval_batches),
        "--eval-every",
        str(args.finetune_eval_every),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--noise-scale",
        str(args.noise_scale),
        "--stop-on-nonfinite",
    ]
    if args.depthwise_noise_scale is not None:
        command.extend(["--depthwise-noise-scale", str(args.depthwise_noise_scale)])
    if args.pointwise_noise_scale is not None:
        command.extend(["--pointwise-noise-scale", str(args.pointwise_noise_scale)])
    if args.linear_noise_scale is not None:
        command.extend(["--linear-noise-scale", str(args.linear_noise_scale)])
    since = time.time()
    returncode = run_command(command, env, args.dry_run)
    if returncode != 0:
        return None
    if args.dry_run:
        return Path(f"DRY_RUN_{mode}_CHECKPOINT.pt")
    metrics_path = latest_metrics_path(since, dataset, mode, args.seed)
    return best_checkpoint_for_run(metrics_path) if metrics_path else None


def evaluate_repeats(
    args,
    task_name: str,
    task: dict[str, Any],
    mode: str,
    checkpoint: Path,
    env: dict[str, str],
) -> int:
    tag = f"_{args.tag}" if args.tag else ""
    output = RUNS_DIR / f"{task_name}_{mode}{tag}_twostage_noise_eval_repeats.csv"
    repeated = [str(checkpoint)] * args.eval_repeats
    eval_mode = eval_mode_for_finetune_mode(mode)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/evaluate_checkpoint.py"),
        "--config",
        str(task["config"]),
        "--dataset",
        task["dataset"],
        "--eval-mode",
        eval_mode,
        "--checkpoints",
        *repeated,
        "--noise-scales",
        str(args.noise_scale),
        "--max-eval-batches",
        str(args.max_eval_batches),
        "--seed",
        str(args.seed),
        "--output",
        str(output),
    ]
    if args.depthwise_noise_scale is not None:
        command.extend(["--depthwise-noise-scale", str(args.depthwise_noise_scale)])
    if args.pointwise_noise_scale is not None:
        command.extend(["--pointwise-noise-scale", str(args.pointwise_noise_scale)])
    if args.linear_noise_scale is not None:
        command.extend(["--linear-noise-scale", str(args.linear_noise_scale)])
    return run_command(command, env, args.dry_run)


def main() -> int:
    args = parse_args()
    env = build_env()
    failures = 0
    RUNS_DIR.mkdir(exist_ok=True)

    for task_name in args.tasks:
        task = TASKS[task_name]
        dataset = task["dataset"]
        model_name = model_name_from_config(task["config"])
        print(f"\n=== two-stage task: {task_name} ===", flush=True)

        clean_checkpoint = None
        if args.reuse_clean:
            clean_checkpoint = find_best_clean_checkpoint(
                dataset, model_name, args.seed, args.clean_epochs, task["config"]
            )
            if clean_checkpoint is not None:
                print(f"reusing clean checkpoint: {clean_checkpoint}", flush=True)
        if clean_checkpoint is None:
            clean_checkpoint = train_clean(args, task, env)
        if clean_checkpoint is None:
            failures += 1
            if not args.continue_on_error:
                return 1
            continue

        for mode in args.finetune_modes:
            checkpoint = train_finetune(args, task, clean_checkpoint, mode, env)
            if checkpoint is None:
                failures += 1
                if not args.continue_on_error:
                    return 1
                continue
            returncode = evaluate_repeats(args, task_name, task, mode, checkpoint, env)
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
        print(f"\nCompleted with {failures} failed task(s).", flush=True)
        return 1
    print("\nTwo-stage formal tasks completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
