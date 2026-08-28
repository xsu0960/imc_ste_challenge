#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev

from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare paired noisy-evaluation CSVs with a paired t-test."
    )
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def accuracy(path: Path) -> float:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one evaluation row in {path}")
    return float(rows[0]["accuracy"])


def main() -> None:
    args = parse_args()
    if not len(args.baseline) == len(args.candidate) == len(args.seeds):
        raise ValueError("baseline, candidate, and seeds must have equal length")
    baseline = [accuracy(path) for path in args.baseline]
    candidate = [accuracy(path) for path in args.candidate]
    differences = [new - old for old, new in zip(baseline, candidate)]
    difference_std = stdev(differences) if len(differences) > 1 else 0.0
    t_result = stats.ttest_rel(candidate, baseline) if len(differences) > 1 else None
    result = {
        "baseline": args.baseline_label,
        "candidate": args.candidate_label,
        "pairs": len(differences),
        "seeds": " ".join(str(seed) for seed in args.seeds),
        "baseline_mean": mean(baseline),
        "candidate_mean": mean(candidate),
        "mean_difference": mean(differences),
        "difference_std": difference_std,
        "difference_ci95": 1.96 * difference_std / math.sqrt(len(differences))
        if len(differences) > 1
        else 0.0,
        "paired_t": float(t_result.statistic) if t_result is not None else "",
        "paired_p": float(t_result.pvalue) if t_result is not None else "",
        "paired_effect_dz": mean(differences) / difference_std
        if difference_std > 0
        else "",
        "differences": " ".join(f"{value:.6f}" for value in differences),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(result)
    print(f"paired_comparison: {args.output}")


if __name__ == "__main__":
    main()
