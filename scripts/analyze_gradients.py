#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from imc_ste import NoiseConfig, noisy_matmul, ste_matmul


METHODS = (
    "noise_autograd",
    "ste",
    "sat_aware_ste",
    "adaptive_sat_aware_ste",
)

STRATEGIES = {
    "ste": "identity",
    "sat_aware_ste": "saturation_aware",
    "adaptive_sat_aware_ste": "adaptive_saturation_aware",
}

NOISE_SCALE_FIELDS = (
    "prog_noise_std",
    "drift_factor",
    "nonlinear_alpha",
    "nonlinear_beta",
    "output_noise_std",
    "crosstalk_factor",
    "temperature_factor",
    "retention_loss",
    "supply_variation_std",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate gradient bias and variance for noisy matmul estimators."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/cifar10_resnet18.yaml",
    )
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--in-features", type=int, default=128)
    parser.add_argument("--out-features", type=int, default=64)
    parser.add_argument("--target-noise", type=float, default=0.5)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs/gradient_diagnostics.csv",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scale_noise_config(config: NoiseConfig, scale: float) -> NoiseConfig:
    if scale < 0:
        raise ValueError("--noise-scale must be non-negative")
    values = asdict(config)
    for field in NOISE_SCALE_FIELDS:
        values[field] *= scale
    return NoiseConfig(**values)


def asymmetric_saturation(
    result: torch.Tensor, alpha: float, beta: float
) -> torch.Tensor:
    pos_scale = alpha
    neg_scale = alpha + beta
    positive = result if pos_scale == 0 else torch.tanh(pos_scale * result) / pos_scale
    negative = result if neg_scale == 0 else torch.tanh(neg_scale * result) / neg_scale
    return torch.where(result >= 0, positive, negative)


def reference_output(
    input: torch.Tensor, weight: torch.Tensor, config: NoiseConfig
) -> torch.Tensor:
    result = torch.matmul(input, weight)
    return asymmetric_saturation(result, config.nonlinear_alpha, config.nonlinear_beta)


def weight_gradient(
    method: str,
    input_base: torch.Tensor,
    weight_base: torch.Tensor,
    target: torch.Tensor,
    config: NoiseConfig,
) -> tuple[torch.Tensor, float]:
    input = input_base.detach().clone().requires_grad_(True)
    weight = weight_base.detach().clone().requires_grad_(True)
    if method == "reference":
        output = reference_output(input, weight, config)
    elif method == "noise_autograd":
        output = noisy_matmul(input, weight, config)
    else:
        output = ste_matmul(input, weight, config, strategy=STRATEGIES[method])
    loss = F.mse_loss(output, target)
    loss.backward()
    return weight.grad.detach().flatten(), float(loss.detach().cpu())


def summarize(
    method: str,
    gradients: list[torch.Tensor],
    losses: list[float],
    ref_gradient: torch.Tensor,
) -> dict[str, Any]:
    eps = torch.finfo(ref_gradient.dtype).eps
    finite_gradients = [
        gradient for gradient in gradients if torch.isfinite(gradient).all().item()
    ]
    if not finite_gradients:
        return {
            "method": method,
            "finite_ratio": 0.0,
            "loss_mean": "",
            "grad_norm_mean": "",
            "grad_norm_std": "",
            "cosine_mean": "",
            "cosine_of_mean": "",
            "relative_error_mean": "",
            "relative_bias": "",
            "relative_variance": "",
        }

    stack = torch.stack(finite_gradients)
    ref = ref_gradient.detach()
    ref_norm = torch.linalg.vector_norm(ref).clamp_min(eps)
    ref_mse = ref.square().mean().clamp_min(eps)
    grad_norms = torch.linalg.vector_norm(stack, dim=1)
    cosines = F.cosine_similarity(stack, ref.unsqueeze(0), dim=1, eps=float(eps))
    relative_errors = torch.linalg.vector_norm(stack - ref.unsqueeze(0), dim=1) / ref_norm
    mean_gradient = stack.mean(dim=0)
    centered = stack - mean_gradient.unsqueeze(0)

    return {
        "method": method,
        "finite_ratio": len(finite_gradients) / len(gradients),
        "loss_mean": sum(losses) / len(losses),
        "grad_norm_mean": float(grad_norms.mean().cpu()),
        "grad_norm_std": float(grad_norms.std(unbiased=False).cpu()),
        "cosine_mean": float(cosines.mean().cpu()),
        "cosine_of_mean": float(
            F.cosine_similarity(
                mean_gradient.unsqueeze(0), ref.unsqueeze(0), dim=1, eps=float(eps)
            )[0].cpu()
        ),
        "relative_error_mean": float(relative_errors.mean().cpu()),
        "relative_bias": float(
            (torch.linalg.vector_norm(mean_gradient - ref) / ref_norm).cpu()
        ),
        "relative_variance": float((centered.square().mean() / ref_mse).cpu()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    device = select_device(args.device)
    raw_config = yaml.safe_load(args.config.read_text())["noise"]
    config = scale_noise_config(NoiseConfig(**raw_config), args.noise_scale)

    torch.manual_seed(args.seed)
    input_base = torch.randn(args.batch_size, args.in_features, device=device)
    weight_base = (
        torch.randn(args.in_features, args.out_features, device=device)
        / math.sqrt(args.in_features)
    )
    with torch.no_grad():
        clean_target = reference_output(input_base, weight_base, config)
        target = clean_target + args.target_noise * torch.randn_like(clean_target)

    ref_gradient, ref_loss = weight_gradient(
        "reference", input_base, weight_base, target, config
    )
    rows = []
    for method in METHODS:
        gradients = []
        losses = []
        for trial in range(args.trials):
            torch.manual_seed(args.seed + trial)
            gradient, loss = weight_gradient(
                method, input_base, weight_base, target, config
            )
            gradients.append(gradient)
            losses.append(loss)
        rows.append(summarize(method, gradients, losses, ref_gradient))

    metadata = {
        "config": str(args.config),
        "trials": args.trials,
        "batch_size": args.batch_size,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "target_noise": args.target_noise,
        "noise_scale": args.noise_scale,
        "seed": args.seed,
        "device": str(device),
        "reference_loss": ref_loss,
        "reference_grad_norm": float(torch.linalg.vector_norm(ref_gradient).cpu()),
        "noise": asdict(config),
    }

    write_csv(args.output, rows)
    json_path = args.output.with_suffix(".json")
    json_path.write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n"
    )
    print(json.dumps({"metadata": metadata, "rows": rows}, ensure_ascii=False))
    print(f"gradient_csv: {args.output}")
    print(f"gradient_json: {json_path}")


if __name__ == "__main__":
    main()
