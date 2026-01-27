"""
translation evaluation metrices, BLEU for now, TODO: add other metrics as well
"""
import sacrebleu

def compute_bleu(references: list[str], hypotheses: list[str], verbose: bool = True):
    """
    computer BLEU score of the references adn hypothesis strings

    returns:
        dict with bleu and chrf2 keys
    """
    refs = [[r] for r in references]
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    chrf = sacrebleu.corpus_chrf(hypotheses, refs)

    out = {
        "bleu": bleu.score,
        "chrf2": chrf.score,
    }
    if verbose:
        print(f"BLEU: {out['bleu']:.2f}  chrF2: {out['chrf2']:.2f}")
    return out


def compute_spec_metrics(
    spec_results: list[dict],
    gamma: int,
    verbose: bool = True
):
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

    if verbose:
        print("\n=== Speculative Decoding Metrics ===")
        print(f"Acceptance Rate: {acceptance_rate:.2%}")
        print(f"Mean Accepted Tokens (per iteration): {mean_accepted_tokens:.2f}")
        print(f"Block Efficiency: {block_efficiency:.2%}")
        print(f"Total Generated: {total_generated} tokens")
        print(f"Total Drafted: {total_draft} tokens")
        print(f"Total Matched: {total_matched} tokens")
    
    return metrics