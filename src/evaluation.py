"""
evaluation and metrics
"""

def compute_baseline_metrics(
    baseline_times: list[float],
) -> tuple[list[dict], dict]:
    """
    Compute per-sentence and summary metrics for a baseline run.

    Args:
        baseline_times: List of per-sentence wall-clock times.

    Returns:
        (per_sentence, summary) where:
            per_sentence: list of dicts with {"baseline/sentence_time": float}
            summary: dict with aggregate timing metrics
    """
    total_time = sum(baseline_times)
    avg_time = total_time / len(baseline_times)

    per_sentence = [
        {"baseline/per_sentence_time": t, "sentence": i}
        for i, t in enumerate(baseline_times)
    ]

    summary = {
        "baseline/total_time": total_time,
        "baseline/avg_time_per_sentence": avg_time,
    }

    return per_sentence, summary


def compute_spec_metrics(
    spec_results: list[dict],
    gamma: int,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """
    Compute per-sentence and summary metrics for a speculative decoding run.

    Args:
        spec_results: List of dicts with keys:
            - total_time: Time for this sentence
            - generated_tokens: Tokens generated for this sentence
            - total_draft_tokens: Draft tokens proposed
            - total_matched_tokens: Draft tokens that matched target
            - acceptance_rate: Per-sentence acceptance rate
        gamma: Number of draft tokens per iteration
        verbose: Print metrics to console

    Returns:
        (per_sentence, summary) where:
            per_sentence: list of dicts with per-sentence time, acceptance, token counts
            summary: dict with aggregate metrics
    """
    if not spec_results:
        return [], {}

    # Per-sentence metrics
    per_sentence = []
    for i, r in enumerate(spec_results):
        per_sentence.append({
            "spec/per_sentence_time": r["total_time"],
            "spec/per_sentence_acceptance_rate": r["acceptance_rate"],
            "spec/per_sentence_generated_tokens": r["generated_tokens"],
            "spec/per_sentence_draft_tokens": r["total_draft_tokens"],
            "spec/per_sentence_matched_tokens": r["total_matched_tokens"],
            "sentence": i,
        })

    # Aggregate across all sentences
    total_generated = sum(r['generated_tokens'] for r in spec_results)
    total_draft = sum(r['total_draft_tokens'] for r in spec_results)
    total_matched = sum(r['total_matched_tokens'] for r in spec_results)

    weighted_acceptance_rate = total_matched / total_draft if total_draft > 0 else 0

    num_iterations = total_draft / gamma if gamma > 0 else 0
    mean_accepted_tokens = total_generated / num_iterations if num_iterations > 0 else 0

    block_efficiency = mean_accepted_tokens / gamma if gamma > 0 else 0

    spec_times = [r['total_time'] for r in spec_results]
    total_time = sum(spec_times)
    avg_time = total_time / len(spec_times)

    summary = {
        "weighted_acceptance_rate": weighted_acceptance_rate,
        "mean_accepted_tokens": mean_accepted_tokens,
        "block_efficiency": block_efficiency,
        "spec/total_time": total_time,
        "spec/avg_time_per_sentence": avg_time,
    }

    if verbose:
        print("\n=== Speculative Decoding Metrics ===")
        print(f"Weighted Acceptance Rate: {weighted_acceptance_rate:.2%}")
        print(f"Mean Accepted Tokens (per iteration): {mean_accepted_tokens:.2f}")
        print(f"Block Efficiency: {block_efficiency:.2%}")

    return per_sentence, summary