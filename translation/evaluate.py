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
    baseline_times: list[float],
    spec_results: list[dict],
    draft_k: int,
    verbose: bool = True
):
    """
    compute all metrices
    """
    if not spec_results:
        return {}
    
    avg_baseline_time = sum(baseline_times) / len(baseline_times) if baseline_times else 0
    avg_spec_time = sum(r['total_time'] for r in spec_results) / len(spec_results)
    speedup = avg_baseline_time / avg_spec_time if avg_spec_time > 0 else 0

    total_decode_time = sum(r['decode_time'] for r in spec_results)
    total_tokens = sum(r['generated_tokens'] for r in spec_results)
    decode_tps = total_tokens / total_decode_time if total_decode_time > 0 else 0

    total_accepted = sum(r['total_accepted_tokens'] for r in spec_results)
    total_proposed = sum(r['total_draft_tokens'] for r in spec_results)
    acceptance_rate = total_accepted / total_proposed if total_proposed > 0 else 0

    all_accepted_per_step = []
    for r in spec_results:
        all_accepted_per_step.extend(r['accepted_tokens_per_step'])
    mat = sum(all_accepted_per_step) / len(all_accepted_per_step) if all_accepted_per_step else 0

    block_efficiency = mat / draft_k if draft_k > 0 else 0

    all_kl = []
    for r in spec_results:
        all_kl.extend(r['kl_divergences'])
    avg_kl = sum(all_kl) / len(all_kl) if all_kl else 0

    metrics = {
        "speedup": speedup,
        "decode_tps": decode_tps,
        "acceptance_rate": acceptance_rate,
        "mean_accepted_tokens": mat,
        "block_efficiency": block_efficiency,
        "avg_kl_divergence": avg_kl,
        # Additional useful metrics
        "avg_baseline_time": avg_baseline_time,
        "avg_spec_time": avg_spec_time,
        "total_tokens": total_tokens,
        "total_decode_time": total_decode_time,
    }

    if verbose:
        print("\n=== Speculative Decoding Metrics ===")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Decode-Only TPS: {decode_tps:.2f} tokens/sec")
        print(f"Acceptance Rate (α): {acceptance_rate:.3f} ({acceptance_rate*100:.1f}%)")
        print(f"Mean Accepted Tokens (MAT): {mat:.2f}")
        print(f"Block Efficiency: {block_efficiency:.2%}")
        print(f"Avg KL Divergence: {avg_kl:.4f}")
        print(f"Avg Baseline Time: {avg_baseline_time:.3f}s")
        print(f"Avg Spec Time: {avg_spec_time:.3f}s")
    
    return metrics