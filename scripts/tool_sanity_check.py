#!/usr/bin/env python3
import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from imc_ste import NoiseConfig, noisy_matmul  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local noisy_matmul with organizer-recommended IMC tools."
    )
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--inner", type=int, default=128)
    parser.add_argument("--cols", type=int, default=96)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs/tool_sanity_check.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=PROJECT_ROOT / "docs/figures/tool_sanity_check_error_hist.png",
    )
    return parser.parse_args()


def snr_db(actual: np.ndarray, ideal: np.ndarray) -> float:
    signal = float(np.sum(ideal**2))
    noise = float(np.sum((ideal - actual) ** 2))
    if noise <= 0:
        return float("inf")
    return 10.0 * math.log10(signal / noise)


def relative_l2(actual: np.ndarray, ideal: np.ndarray) -> float:
    denom = float(np.linalg.norm(ideal))
    if denom <= 0:
        return 0.0
    return float(np.linalg.norm(actual - ideal) / denom)


def mean_abs_error(actual: np.ndarray, ideal: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - ideal)))


def flatten_error(actual: np.ndarray, ideal: np.ndarray) -> np.ndarray:
    return (actual - ideal).reshape(-1)


def run_local_noisy(input_np: np.ndarray, weight_np: np.ndarray) -> np.ndarray:
    input_tensor = torch.from_numpy(input_np).float()
    weight_tensor = torch.from_numpy(weight_np).float()
    with torch.no_grad():
        output = noisy_matmul(input_tensor, weight_tensor, NoiseConfig())
    return output.cpu().numpy()


def run_memintelli(input_np: np.ndarray, weight_np: np.ndarray) -> np.ndarray:
    import memintelli

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.from_numpy(input_np).float().to(device)
    weight_tensor = torch.from_numpy(weight_np).float().to(device)
    engine = memintelli.DPETensor(
        HGS=1e-5,
        LGS=1e-8,
        g_level=2**2,
        var=0.05,
        vnoise=0.05,
        rdac=2**2,
        radc=2**12,
        weight_paral_size=(32, 32),
        input_paral_size=(1, 32),
        device=device,
    )
    input_slice = torch.tensor([1, 1, 2, 2, 2], device=device)
    weight_slice = torch.tensor([1, 1, 2, 2, 2], device=device)
    input_data = memintelli.SlicedData(input_slice, device=device)
    weight_data = memintelli.SlicedData(weight_slice, device=device)
    input_data.slice_data_imp(engine, input_tensor)
    weight_data.slice_data_imp(engine, weight_tensor)
    with torch.no_grad():
        output = engine(input_data, weight_data)
    return output.detach().cpu().numpy()


def run_mpimpy(input_np: np.ndarray, weight_np: np.ndarray) -> np.ndarray:
    from mpimpy.memmat import bitslicedpe

    engine = bitslicedpe(
        HGS=1e-5,
        LGS=1e-8,
        g_level=2**2,
        var=0.05,
        vnoise=0.05,
        rdac=2**2,
        radc=2**12,
        array_size=(32, 32),
    )
    return engine.MapReduceDot(
        input_np,
        weight_np,
        xblk=[1, 1, 2, 2, 2],
        mblk=[1, 1, 2, 2, 2],
    )


def summarize_tool(
    name: str,
    runner,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    rows: list[dict[str, Any]] = []
    errors: list[np.ndarray] = []
    for trial in range(1, args.trials + 1):
        input_np = rng.normal(size=(args.rows, args.inner)).astype(np.float32)
        weight_np = rng.normal(size=(args.inner, args.cols)).astype(np.float32)
        ideal = input_np @ weight_np
        try:
            actual = runner(input_np, weight_np)
            status = "ok"
            error = ""
            errors.append(flatten_error(actual, ideal))
            metrics = {
                "snr_db": snr_db(actual, ideal),
                "relative_l2": relative_l2(actual, ideal),
                "mean_abs_error": mean_abs_error(actual, ideal),
            }
        except Exception as exc:  # pragma: no cover - tool compatibility varies
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            metrics = {"snr_db": "", "relative_l2": "", "mean_abs_error": ""}

        rows.append(
            {
                "tool": name,
                "trial": trial,
                "status": status,
                "rows": args.rows,
                "inner": args.inner,
                "cols": args.cols,
                **metrics,
                "error": error,
            }
        )
    return rows, errors


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def plot_errors(path: Path, tool_errors: dict[str, list[np.ndarray]]) -> None:
    available = {
        name: np.concatenate(errors)
        for name, errors in tool_errors.items()
        if errors
    }
    if not available:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for name, errors in available.items():
        clipped = np.clip(errors, np.percentile(errors, 1), np.percentile(errors, 99))
        ax.hist(clipped, bins=60, density=True, histtype="step", linewidth=1.8, label=name)
    ax.set_title("Matrix multiplication error distribution")
    ax.set_xlabel("Actual - ideal")
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    runners = {
        "local_noisy_matmul": run_local_noisy,
        "memintelli": run_memintelli,
        "mpimpy": run_mpimpy,
    }
    rows: list[dict[str, Any]] = []
    tool_errors: dict[str, list[np.ndarray]] = {}
    for name, runner in runners.items():
        tool_rows, errors = summarize_tool(name, runner, rng, args)
        rows.extend(tool_rows)
        tool_errors[name] = errors

    write_csv(args.output, rows)
    plot_errors(args.figure, tool_errors)


if __name__ == "__main__":
    main()
