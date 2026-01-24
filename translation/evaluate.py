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