"""Automatic QA metrics: EM, F1 and Semantic similarity.

EM and F1 are computed on normalized answers (lowercase, no punctuation or
articles) and take the maximum over all accepted gold answers.

Example:
    python evaluation/nlp_metric.py --data nq --model gpt-4o-mini --retrieval_method hypercube --k 5
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.common import add_common_args, resolve_output_path, load_samples, get_gold_answers
from utils.metric import em_check, f1_score, semantic_score


def parse_args():
    parser = argparse.ArgumentParser(description="Compute EM / F1 / Semantic scores of LLM answers.")
    add_common_args(parser)
    parser.add_argument("--metric", type=str, default="all", choices=["em", "f1", "semantic", "all"],
                        help="Metric to compute.")
    return parser.parse_args()


def main():
    args = parse_args()
    path = resolve_output_path(args)
    samples = load_samples(path)
    metrics = ["em", "f1", "semantic"] if args.metric == "all" else [args.metric]
    names = {"em": "average_EM", "f1": "average_F1", "semantic": "average_Semantic"}

    totals = {m: 0.0 for m in metrics}
    for sample in samples:
        predicted = str(sample["predicted_answer"] or "")
        golds = get_gold_answers(sample)
        if "em" in metrics:
            totals["em"] += em_check(predicted, golds)
        if "f1" in metrics:
            totals["f1"] += max(f1_score(g, predicted) for g in golds)
        if "semantic" in metrics:
            totals["semantic"] += max(semantic_score(predicted, g) for g in golds)

    n = len(samples)
    scores = {names[m]: totals[m] / n * 100 for m in metrics}
    print(f"File: {path}")
    print(f"Total samples: {n}, " + ", ".join(f"{k}: {v:.1f}" for k, v in scores.items()))

    out_path = path.replace(".json", "_nlp_scores.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"input": path, "total_samples": n, **scores}, f, indent=2)
    print(f"Scores saved to {out_path}")


if __name__ == "__main__":
    main()
