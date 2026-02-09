"""
Translation evaluation metrics.
"""
import sacrebleu


def compute_bleu(references: list[str], hypotheses: list[str], verbose: bool = True) -> dict:
    """
    Compute BLEU and chrF2 scores for references and hypothesis strings.

    Returns:
        dict with bleu and chrf2 keys
    """
    refs = [references]
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    chrf = sacrebleu.corpus_chrf(hypotheses, refs)

    out = {
        "bleu": bleu.score,
        "chrf2": chrf.score,
    }
    if verbose:
        print(f"BLEU: {out['bleu']:.2f}  chrF2: {out['chrf2']:.2f}")
    return out


