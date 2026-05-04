"""
eval_script.py – BIS Hackathon Evaluation Script
Computes Hit Rate @3, MRR @5, and Average Latency from results JSON.

Usage:
    python eval_script.py --results team_results.json --ground_truth public_test.json
"""

import argparse
import json


def evaluate(results_path: str, ground_truth_path: str) -> dict:
    with open(results_path) as f:
        results = json.load(f)

    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    # Build GT lookup: id -> set of expected standards
    gt_lookup = {item["id"]: set(item.get("expected_standards", [])) for item in ground_truth}

    hit_count = 0
    mrr_sum = 0.0
    total_latency = 0.0
    n = len(results)

    if n == 0:
        print("❌ No results found.")
        return {}

    for result in results:
        qid = result["id"]
        retrieved = result["retrieved_standards"]
        latency = result["latency_seconds"]
        expected = gt_lookup.get(qid, set())

        total_latency += latency

        # Hit Rate @3
        top3 = set(retrieved[:3])
        if top3 & expected:
            hit_count += 1

        # MRR @5
        for rank, std in enumerate(retrieved[:5], start=1):
            if std in expected:
                mrr_sum += 1 / rank
                break

    metrics = {
        "hit_rate_at_3":   round(hit_count / n * 100, 2),
        "mrr_at_5":        round(mrr_sum / n, 4),
        "avg_latency_sec": round(total_latency / n, 3),
        "n_queries":       n,
    }

    print("\n📊 Evaluation Results")
    print("=" * 40)
    print(f"  Hit Rate @3   : {metrics['hit_rate_at_3']}%   (target > 80%)")
    print(f"  MRR @5        : {metrics['mrr_at_5']}     (target > 0.7)")
    print(f"  Avg Latency   : {metrics['avg_latency_sec']}s    (target < 5s)")
    print(f"  Total Queries : {metrics['n_queries']}")
    print("=" * 40)

    passed = []
    if metrics["hit_rate_at_3"] >= 80:
        passed.append("✅ Hit Rate @3")
    else:
        passed.append("❌ Hit Rate @3")

    if metrics["mrr_at_5"] >= 0.7:
        passed.append("✅ MRR @5")
    else:
        passed.append("❌ MRR @5")

    if metrics["avg_latency_sec"] < 5.0:
        passed.append("✅ Avg Latency")
    else:
        passed.append("❌ Avg Latency")

    print("\nTarget Check:")
    for p in passed:
        print(f"  {p}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIS RAG Evaluation Script")
    parser.add_argument("--results",      required=True, help="Path to team results JSON")
    parser.add_argument("--ground_truth", required=True, help="Path to ground truth JSON")
    args = parser.parse_args()

    evaluate(args.results, args.ground_truth)
