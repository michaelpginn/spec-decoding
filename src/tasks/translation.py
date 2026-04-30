"""
Load translation data from various sources.
"""

import csv
import logging
from typing import cast

import sacrebleu

from src.config.config import ExperimentConfig
from src.data.dataset import REFERENCE_TABLE, load_bilingual_dataset

logger = logging.getLogger(__name__)


def load_data(config: ExperimentConfig) -> tuple[list[tuple[str, str]], str]:
    """
    Load (source, target) pairs from the bilingual train split.

    Returns:
        - List of (source_text, target_text) tuples
        - Language name
    """
    max_samples = config.max_samples if config.max_samples > 0 else None
    splits = load_bilingual_dataset(config.language_code, max_samples)
    ds = splits["train"]

    lang_name = _get_language_name(config.language_code)
    columns = cast(list[str], ds.column_names)

    if "English" in columns and lang_name in columns:
        src_col, tgt_col = "English", lang_name
    elif len(columns) == 2:
        src_col, tgt_col = columns[0], columns[1]
    else:
        raise ValueError(
            f"Cannot determine source/target columns from {list(columns)}. "
            f"Expected either an 'English' column and a '{lang_name}' column, "
            f"or exactly two columns."
        )

    logger.info(f"Using columns: '{src_col}' (source) -> '{tgt_col}' (target)")

    pairs = []
    for row in ds:
        source = str(row.get(src_col, "")).strip()  # type:ignore
        target = str(row.get(tgt_col, "")).strip()  # type:ignore
        if not source or not target:
            continue
        pairs.append((source, target))
    return pairs, lang_name


def compute_eval_metrics(
    references: list[str], hypotheses: list[str], verbose: bool = False
) -> dict:
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


def _get_language_name(lang_code: str) -> str:
    """
    Get full language name from language code using reference_table_bilingual.csv.
    e.g. 'npi' -> 'Nepali', 'chr' -> 'Cherokee'
    """
    lang_code = lang_code.strip().lower()
    # Use utf-8-sig
    with open(REFERENCE_TABLE, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["Code"].strip().lower() == lang_code:
                return row["Language"].strip()
    # Fallback: return the code itself if not found
    return lang_code


