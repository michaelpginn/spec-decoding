"""
evaluation and metrics
"""

from statistics import median


def _compute_common_metrics(
    times: list[float],
    token_counts: list[int],
) -> tuple[list[dict], dict]:
    """
    Base metrics shared by baseline and spec runs.

    Args:
        times: Per-sentence wall-clock times.
        token_counts: Per-sentence generated token counts.

    Returns:
        (per_sentence, summary) with unified key names usable by both run types.
    """
    if not times:
        return [], {}

    n = len(times)
    total_time = sum(times)
    total_tokens = sum(token_counts)

    time_per_token_list = [
        t / tc if tc > 0 else 0.0 for t, tc in zip(times, token_counts)
    ]

    per_sentence = [
        {
            "sentence/time": t,
            "sentence/time_per_token": tpt,
            "sentence/generated_tokens": tc,
            "sentence_idx": i,
        }
        for i, (t, tpt, tc) in enumerate(
            zip(times, time_per_token_list, token_counts)
        )
    ]

    summary = {
        "total_time": total_time,
        "avg_time_per_sentence": total_time / n,
        "median_time_per_sentence": median(times),
        "avg_time_per_token": sum(time_per_token_list) / n,
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
    }

    return per_sentence, summary


def compute_baseline_metrics(
    baseline_times: list[float],
    token_counts: list[int],
) -> tuple[list[dict], dict]:
    """
    Compute per-sentence and summary metrics for a baseline run.

    Args:
        baseline_times: List of per-sentence wall-clock times.
        token_counts: List of per-sentence generated token counts.

    Returns:
        (per_sentence, summary) with unified metric keys.
    """
    return _compute_common_metrics(baseline_times, token_counts)


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
        (per_sentence, summary) with unified + spec-specific metrics.
    """
    if not spec_results:
        return [], {}

    times = [r["total_time"] for r in spec_results]
    token_counts = [r["generated_tokens"] for r in spec_results]
    per_sentence, summary = _compute_common_metrics(times, token_counts)

    for i, r in enumerate(spec_results):
        per_sentence[i]["sentence/acceptance_rate"] = r["acceptance_rate"]
        per_sentence[i]["sentence/draft_tokens"] = r["total_draft_tokens"]
        per_sentence[i]["sentence/matched_tokens"] = r["total_matched_tokens"]

    total_generated = sum(r["generated_tokens"] for r in spec_results)
    total_draft = sum(r["total_draft_tokens"] for r in spec_results)
    total_matched = sum(r["total_matched_tokens"] for r in spec_results)

    num_iterations = total_draft / gamma if gamma > 0 else 0
    mean_accepted = total_generated / num_iterations if num_iterations > 0 else 0

    summary["acceptance_rate"] = total_matched / total_draft if total_draft > 0 else 0
    summary["mean_acceptance_rate"] = (
        sum(r["acceptance_rate"] for r in spec_results) / len(spec_results)
    )
    summary["mean_accepted_tokens"] = mean_accepted
    summary["block_efficiency"] = mean_accepted / gamma if gamma > 0 else 0

    if verbose:
        print("\n=== Speculative Decoding Metrics ===")
        print(f"Acceptance Rate (weighted): {summary['acceptance_rate']:.2%}")
        print(f"Mean Accepted Tokens (per iteration): {mean_accepted:.2f}")
        print(f"Block Efficiency: {summary['block_efficiency']:.2%}")
        print(f"Tokens/sec: {summary['tokens_per_second']:.2f}")

    return per_sentence, summary
