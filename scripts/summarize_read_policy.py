#!/usr/bin/env python3
import argparse
import ast
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize accuracy and logical read cost for layer-read policies."
    )
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_mapping(value: str) -> dict:
    return ast.literal_eval(value) if value else {}


def main() -> None:
    args = parse_args()
    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have equal length")

    rows = []
    uniform_total = None
    for path, label in zip(args.inputs, args.labels):
        with path.open(newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        if len(source_rows) != 1:
            raise ValueError(f"expected exactly one row in {path}")
        source = source_rows[0]
        if source.get("output_noise_read_compensation", "").lower() == "true":
            counts = parse_mapping(source["output_noise_read_counts"])
            total_reads = int(counts["total_read_repeats"])
            mean_reads = float(counts["mean_read_repeats"])
            histogram = counts["read_histogram"]
        else:
            layer_counts = parse_mapping(source["layerwise_read_repeat_counts"])
            depthwise = int(layer_counts.get("depthwise", 0))
            pointwise = int(layer_counts.get("pointwise", 0))
            total_layers = depthwise + pointwise
            total_reads = (
                depthwise * int(source.get("depthwise_read_repeats") or 1)
                + pointwise * int(source.get("pointwise_read_repeats") or 1)
            )
            mean_reads = total_reads / total_layers if total_layers else 0.0
            histogram = {str(int(mean_reads)): total_layers}
            if uniform_total is None:
                uniform_total = total_reads
        rows.append(
            {
                "policy": label,
                "accuracy": float(source["accuracy"]),
                "examples": int(source["examples"]),
                "total_sensitive_layer_reads": total_reads,
                "mean_sensitive_layer_reads": mean_reads,
                "read_reduction_vs_uniform": "",
                "read_histogram": histogram,
                "noise_scale": source["noise_scale"],
                "eval_seed": source.get("eval_seed", ""),
            }
        )

    if uniform_total is None:
        uniform_total = max(row["total_sensitive_layer_reads"] for row in rows)
    for row in rows:
        row["read_reduction_vs_uniform"] = (
            1 - row["total_sensitive_layer_reads"] / uniform_total
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"read_policy_summary: {args.output}")


if __name__ == "__main__":
    main()
