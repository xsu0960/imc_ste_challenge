#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the TinyImageNet adaptive-scale study summary."
    )
    parser.add_argument("--fixed-summary", type=Path, required=True)
    parser.add_argument("--learned-summary", type=Path, required=True)
    parser.add_argument("--output-comp-summary", type=Path, required=True)
    parser.add_argument("--learned-comparison", type=Path, required=True)
    parser.add_argument("--output-comp-comparison", type=Path, required=True)
    parser.add_argument("--read-policy-summary", type=Path, required=True)
    parser.add_argument("--learned-run", type=Path, required=True)
    parser.add_argument("--moment-fixed-run", type=Path, required=True)
    parser.add_argument("--moment-learned-run", type=Path, required=True)
    parser.add_argument("--moment-eval", type=Path, required=True)
    parser.add_argument("--uniform-timed-eval", type=Path, required=True)
    parser.add_argument("--output-comp-timed-eval", type=Path, required=True)
    parser.add_argument("--exact-read-benchmark", type=Path, required=True)
    parser.add_argument("--moment-read-benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def one_csv(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}")
    return rows[0]


def run_timing(path: Path) -> tuple[float, int]:
    seconds = 0.0
    batches = 0
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if "epoch" not in record:
            continue
        seconds += float(record.get("train_seconds", 0.0))
        batches += int(record["train"]["examples"]) // 64
    return seconds, batches


def main() -> None:
    args = parse_args()
    fixed = one_csv(args.fixed_summary)
    learned = one_csv(args.learned_summary)
    output_comp = one_csv(args.output_comp_summary)
    learned_comparison = one_csv(args.learned_comparison)
    output_comp_comparison = one_csv(args.output_comp_comparison)
    policies = {}
    with args.read_policy_summary.open(newline="") as handle:
        for row in csv.DictReader(handle):
            policies[row["policy"]] = row
    with args.moment_eval.open(newline="") as handle:
        moment_eval_rows = list(csv.DictReader(handle))
    if len(moment_eval_rows) != 2:
        raise ValueError("--moment-eval must contain fixed and learned joint rows")
    uniform_timed = one_csv(args.uniform_timed_eval)
    output_comp_timed = one_csv(args.output_comp_timed_eval)
    uniform_eval_seconds = float(uniform_timed["evaluation_seconds"])
    output_comp_eval_seconds = float(output_comp_timed["evaluation_seconds"])
    exact_seconds, exact_batches = run_timing(args.exact_read_benchmark)
    moment_seconds, moment_batches = run_timing(args.moment_read_benchmark)
    exact_seconds_per_batch = exact_seconds / exact_batches
    moment_seconds_per_batch = moment_seconds / moment_batches

    learned_seconds, learned_batches = run_timing(args.learned_run)
    fixed_seconds, fixed_batches = run_timing(args.moment_fixed_run)
    joint_seconds, joint_batches = run_timing(args.moment_learned_run)
    rows = [
        {
            "protocol": "fixed_p99_uniform_read8",
            "validation_scope": "full",
            "repeats": fixed["runs"],
            "mean_accuracy": fixed["mean_accuracy"],
            "ci95": fixed["ci95_accuracy"],
            "paired_delta_vs_fixed": 0.0,
            "paired_p": "",
            "sensitive_layer_reads": 640,
            "read_reduction": 0.0,
            "training_seconds_per_batch": "",
            "training_speedup_vs_exact": "",
            "evaluation_seconds": uniform_eval_seconds,
            "evaluation_speedup_vs_uniform": 0.0,
            "conclusion": "stable reference",
        },
        {
            "protocol": "constrained_learned_scale_uniform_read8",
            "validation_scope": "full",
            "repeats": learned["runs"],
            "mean_accuracy": learned["mean_accuracy"],
            "ci95": learned["ci95_accuracy"],
            "paired_delta_vs_fixed": learned_comparison["mean_difference"],
            "paired_p": learned_comparison["paired_p"],
            "sensitive_layer_reads": 640,
            "read_reduction": 0.0,
            "training_seconds_per_batch": learned_seconds / learned_batches,
            "training_speedup_vs_exact": "",
            "evaluation_seconds": "",
            "evaluation_speedup_vs_uniform": "",
            "conclusion": "small positive trend; not statistically significant",
        },
        {
            "protocol": "output_compensated_base4_max8",
            "validation_scope": "full",
            "repeats": output_comp["runs"],
            "mean_accuracy": output_comp["mean_accuracy"],
            "ci95": output_comp["ci95_accuracy"],
            "paired_delta_vs_fixed": output_comp_comparison["mean_difference"],
            "paired_p": output_comp_comparison["paired_p"],
            "sensitive_layer_reads": policies["compensated_base4_max8"][
                "total_sensitive_layer_reads"
            ],
            "read_reduction": policies["compensated_base4_max8"][
                "read_reduction_vs_uniform"
            ],
            "training_seconds_per_batch": "",
            "training_speedup_vs_exact": "",
            "evaluation_seconds": output_comp_eval_seconds,
            "evaluation_speedup_vs_uniform": 1
            - output_comp_eval_seconds / uniform_eval_seconds,
            "conclusion": "statistically indistinguishable accuracy at lower read cost",
        },
        {
            "protocol": "moment_matched_read8_fixed_joint",
            "validation_scope": "20 batches",
            "repeats": 1,
            "mean_accuracy": moment_eval_rows[0]["accuracy"],
            "ci95": "",
            "paired_delta_vs_fixed": "",
            "paired_p": "",
            "sensitive_layer_reads": 640,
            "read_reduction": "training uses one physical draw per logical read average",
            "training_seconds_per_batch": fixed_seconds / fixed_batches,
            "training_speedup_vs_exact": exact_seconds_per_batch
            / (fixed_seconds / fixed_batches),
            "evaluation_seconds": "",
            "evaluation_speedup_vs_uniform": "",
            "conclusion": "fast screening path; did not improve final accuracy",
        },
        {
            "protocol": "moment_matched_read8_learned_joint",
            "validation_scope": "20 batches",
            "repeats": 1,
            "mean_accuracy": moment_eval_rows[1]["accuracy"],
            "ci95": "",
            "paired_delta_vs_fixed": "",
            "paired_p": "",
            "sensitive_layer_reads": 640,
            "read_reduction": "training uses one physical draw per logical read average",
            "training_seconds_per_batch": joint_seconds / joint_batches,
            "training_speedup_vs_exact": exact_seconds_per_batch
            / (joint_seconds / joint_batches),
            "evaluation_seconds": "",
            "evaluation_speedup_vs_uniform": "",
            "conclusion": "fast screening path; did not beat fixed reference",
        },
        {
            "protocol": "exact_read8_training_benchmark",
            "validation_scope": "2 training batches",
            "repeats": 1,
            "mean_accuracy": "",
            "ci95": "",
            "paired_delta_vs_fixed": "",
            "paired_p": "",
            "sensitive_layer_reads": 640,
            "read_reduction": 0.0,
            "training_seconds_per_batch": exact_seconds_per_batch,
            "training_speedup_vs_exact": 1.0,
            "evaluation_seconds": "",
            "evaluation_speedup_vs_uniform": "",
            "conclusion": "exact multi-read training reference",
        },
        {
            "protocol": "moment_matched_read8_training_benchmark",
            "validation_scope": "2 training batches",
            "repeats": 1,
            "mean_accuracy": "",
            "ci95": "",
            "paired_delta_vs_fixed": "",
            "paired_p": "",
            "sensitive_layer_reads": 640,
            "read_reduction": "one physical draw per logical read average",
            "training_seconds_per_batch": moment_seconds_per_batch,
            "training_speedup_vs_exact": exact_seconds_per_batch
            / moment_seconds_per_batch,
            "evaluation_seconds": "",
            "evaluation_speedup_vs_uniform": "",
            "conclusion": "48x-class screening acceleration; exact eval remains required",
        },
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"adaptive_study_summary: {args.output}")


if __name__ == "__main__":
    main()
