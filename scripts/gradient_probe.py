#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from imc_ste.noise import NoiseConfig, noisy_matmul
from imc_ste.ste import ste_matmul


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/cifar10_resnet18.yaml"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=32)
    return parser.parse_args()


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        left.flatten(), right.flatten(), dim=0
    ).item()


def get_gradient(mode, input_value, weight_value, target, config):
    input = input_value.detach().clone().requires_grad_()
    weight = weight_value.detach().clone().requires_grad_()

    if mode == "clean":
        output = torch.matmul(input, weight)
    elif mode == "noise":
        output = noisy_matmul(input, weight, config)
    elif mode == "ste":
        output = ste_matmul(input, weight, config, strategy="identity")
    elif mode == "sat_aware_ste":
        output = ste_matmul(input, weight, config, strategy="saturation_aware")
    elif mode == "adaptive_sat_aware_ste":
        output = ste_matmul(input, weight, config, strategy="adaptive_saturation_aware")
    else:
        raise ValueError(mode)

    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    return weight.grad.detach(), loss.item()


def main():
    args = parse_args()
    raw_config = yaml.safe_load(args.config.read_text())
    config = NoiseConfig(**raw_config["noise"])
    torch.manual_seed(args.seed)

    input = torch.randn(64, 128)
    weight = torch.randn(128, 32) * 0.1
    target = torch.randn(64, 32)
    clean_gradient, clean_loss = get_gradient("clean", input, weight, target, config)

    results = {
        "clean": {
            "loss": clean_loss,
            "gradient_norm": clean_gradient.norm().item(),
            "cosine_to_clean": 1.0,
        }
    }
    for mode in ("noise", "ste", "sat_aware_ste", "adaptive_sat_aware_ste"):
        gradients = []
        losses = []
        for _ in range(args.samples):
            gradient, loss = get_gradient(mode, input, weight, target, config)
            gradients.append(gradient)
            losses.append(loss)

        stacked = torch.stack(gradients)
        mean_gradient = stacked.mean(dim=0)
        results[mode] = {
            "loss_mean": torch.tensor(losses).mean().item(),
            "gradient_norm_mean": stacked.flatten(1).norm(dim=1).mean().item(),
            "gradient_variance": stacked.var(dim=0).mean().item(),
            "mean_gradient_cosine_to_clean": cosine_similarity(
                mean_gradient, clean_gradient
            ),
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
