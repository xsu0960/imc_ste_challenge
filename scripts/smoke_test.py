#!/usr/bin/env python3
"""Dataset-free end-to-end smoke test for noisy layers and online STE."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from imc_ste import (  # noqa: E402
    NoiseConfig,
    NoisyConv2d,
    NoisyLinear,
    OnlineGradientProfile,
    convert_model,
    set_compute_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4, bias=False),
        nn.ReLU(),
        nn.Conv2d(4, 6, kernel_size=1, bias=False),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(6, 3),
    )


def assert_finite_gradients(model: nn.Module) -> tuple[int, float]:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise AssertionError("smoke test produced no gradients")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("smoke test produced a non-finite gradient")
    norm = torch.sqrt(sum(gradient.detach().float().square().sum() for gradient in gradients))
    return len(gradients), float(norm)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    config = NoiseConfig(quantization_bits=8)
    model = convert_model(
        build_model(), config, compute_mode="variance_sat_aware_ste"
    ).to(device)
    model.train()
    inputs = torch.randn(2, 3, 8, 8, device=device)
    targets = torch.tensor([0, 2], device=device)
    criterion = nn.CrossEntropyLoss()

    profile = OnlineGradientProfile(
        model,
        ("",),
        ema_decay=0.0,
        variance_strength=0.5,
        bias_strength=0.25,
        scale_floor=0.5,
        warmup_updates=0,
    )
    profile.begin_clean()
    set_compute_mode(model, "clean")
    with torch.no_grad():
        clean_logits = model(inputs)
    profile.begin_noisy()
    set_compute_mode(model, "variance_sat_aware_ste")
    noisy_logits = model(inputs)
    profile.finalize_batch()
    loss = criterion(noisy_logits, targets)
    loss.backward()

    if clean_logits.shape != (2, 3) or noisy_logits.shape != clean_logits.shape:
        raise AssertionError("smoke test output shape mismatch")
    if not torch.isfinite(clean_logits).all() or not torch.isfinite(noisy_logits).all():
        raise AssertionError("smoke test produced a non-finite output")
    gradient_tensors, gradient_norm = assert_finite_gradients(model)
    noisy_layers = sum(
        isinstance(module, (NoisyConv2d, NoisyLinear)) for module in model.modules()
    )
    profile_summary = profile.summary()
    if profile_summary["initialized_layers"] != noisy_layers:
        raise AssertionError("online profile did not initialize every noisy layer")
    if profile_summary["shape_mismatches"] or profile_summary["skipped_calls"]:
        raise AssertionError("online profile failed to pair clean and noisy layer calls")
    profile.close()

    result = {
        "status": "passed",
        "device": str(device),
        "torch": torch.__version__,
        "noisy_layers": noisy_layers,
        "gradient_tensors": gradient_tensors,
        "gradient_norm": gradient_norm,
        "loss": float(loss.detach()),
        "profile": profile_summary,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
