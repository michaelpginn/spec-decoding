"""
speculative decoding evaluation metrics.
"""

def compute_spec_metrics(
    spec_results: list[dict],
    gamma: int,
    baseline_times: list[float] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Compute speculative decoding metrics from a list of per-sample results.
    
    Args:
        spec_results: List of dicts with keys:
            - total_time: Time for this sample
            - generated_tokens: Tokens generated for this sample
            - total_draft_tokens: Draft tokens proposed
            - total_matched_tokens: Draft tokens that matched target
            - acceptance_rate: Per-sample acceptance rate
        gamma: Number of draft tokens per iteration
        verbose: Print metrics to console
    
    Returns:
        dict with aggregated metrics
    """
    if not spec_results:
        return {}
    
    # Aggregate across all samples
    total_generated = sum(r['generated_tokens'] for r in spec_results)
    total_draft = sum(r['total_draft_tokens'] for r in spec_results)
    total_matched = sum(r['total_matched_tokens'] for r in spec_results)
    
    # Overall acceptance rate (matched / drafted across all samples)
    acceptance_rate = total_matched / total_draft if total_draft > 0 else 0
    
    # Mean accepted tokens per iteration (approximation)
    # This estimates how many tokens are accepted per spec decode iteration
    num_iterations = total_draft / gamma if gamma > 0 else 0
    mean_accepted_tokens = total_generated / num_iterations if num_iterations > 0 else 0
    
    # Block efficiency: what fraction of gamma tokens are we effectively using
    block_efficiency = mean_accepted_tokens / gamma if gamma > 0 else 0
    
    metrics = {
        "acceptance_rate": acceptance_rate,
        "mean_accepted_tokens": mean_accepted_tokens,
        "block_efficiency": block_efficiency,
        "total_generated_tokens": total_generated,
        "total_draft_tokens": total_draft,
        "total_matched_tokens": total_matched,
    }

    if baseline_times:
        total_baseline_time = sum(baseline_times)
        total_spec_time = sum(r['total_time'] for r in spec_results)
        speedup = total_baseline_time / total_spec_time if total_spec_time > 0 else 0.0
        
        metrics["baseline_total_time"] = total_baseline_time
        metrics["spec_total_time"] = total_spec_time
        metrics["speedup"] = speedup

    if verbose:
        print("\n=== Speculative Decoding Metrics ===")
        print(f"Acceptance Rate: {acceptance_rate:.2%}")
        print(f"Mean Accepted Tokens (per iteration): {mean_accepted_tokens:.2f}")
        print(f"Block Efficiency: {block_efficiency:.2%}")
        print(f"Total Generated: {total_generated} tokens")
        print(f"Total Drafted: {total_draft} tokens")
        print(f"Total Matched: {total_matched} tokens")

        if baseline_times:
            print(f"Speedup: {speedup:.2f}x")
    
    return metrics