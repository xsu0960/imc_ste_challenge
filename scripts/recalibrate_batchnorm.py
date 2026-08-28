#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imc_ste import COMPUTE_MODES, NoiseConfig, convert_model, scale_noise_config, set_compute_mode
from train import (
    DATASETS,
    build_loaders,
    build_model,
    infer_num_classes,
    limit_or_none,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-estimate BatchNorm running statistics under a selected noisy forward path."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=COMPUTE_MODES, default="noise")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def reset_batchnorm_stats(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            count += 1
    return count


def recalibrate(model: nn.Module, loader, device: torch.device, max_batches: int | None) -> int:
    model.train()
    examples = 0
    with torch.no_grad():
        for batch_index, (images, _) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device)
            model(images)
            examples += images.shape[0]
    return examples


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)
    training_config = config["training"]
    batch_size = args.batch_size or training_config["batch_size"]
    num_classes = infer_num_classes(args.dataset, config["model"]["num_classes"])
    train_loader, _ = build_loaders(
        args.dataset,
        batch_size,
        training_config.get("num_workers", 0),
        args.seed,
        num_classes,
        training_config.get("augmentation", "standard"),
        training_config.get("image_size"),
    )

    noise_config = scale_noise_config(NoiseConfig(**config["noise"]), args.noise_scale)
    model = build_model(config["model"], num_classes)
    model = convert_model(model, noise_config, args.mode).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    set_compute_mode(model, args.mode)

    bn_layers = reset_batchnorm_stats(model)
    examples = recalibrate(model, train_loader, device, limit_or_none(args.max_batches))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "output": str(args.output),
                "dataset": args.dataset,
                "mode": args.mode,
                "noise_scale": args.noise_scale,
                "batch_size": batch_size,
                "max_batches": limit_or_none(args.max_batches),
                "examples": examples,
                "batchnorm_layers": bn_layers,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
