"""
Load translation data from various sources.
"""

import csv
import logging
from pathlib import Path
from typing import cast

import sacrebleu
from datasets import load_dataset

from src.config.config import ExperimentConfig

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REFERENCE_TABLE = DATA_DIR / "reference_table_bilingual.csv"


def load_data(config: ExperimentConfig) -> tuple[list[tuple[str, str]], str]:
    """
    Load (source, target) pairs from Tatoeba via HuggingFace.

    Args:
        target_lang: Language code, e.g. 'ber', 'chr', 'haw', 'npi'
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        - List of (source_text, target_text) tuples
        - Language name
    """
    hf_id, lang_name = _get_hf_dataset_id(config.language_code)
    logger.info(f"Loading from HuggingFace: {hf_id}")

    start = max(0, int(getattr(config, "data_start", 0) or 0))
    end = int(getattr(config, "data_end", 0) or 0)
    if end > 0 and end <= start:
        raise ValueError(f"Invalid data slice: data_end ({end}) must be > data_start ({start})")

    if end > 0:
        split = f"train[{start}:{end}]"
    elif config.max_samples is not None and config.max_samples > 0:
        split = f"train[{start}:{start + config.max_samples}]"
    elif start > 0:
        split = f"train[{start}:]"
    else:
        split = "train"
    ds = load_dataset(hf_id, split=split)
    columns = ds.column_names
    columns = cast(list[str], columns)
    pairs = []

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

    for row in ds:
        source = str(row.get(src_col, "")).strip()  # type:ignore
        target = str(row.get(tgt_col, "")).strip()  # type:ignore
        if not source or not target:
            continue
        pairs.append((source, target))
    return pairs, _get_language_name(config.language_code)


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


def _get_hf_dataset_id(target_lang: str) -> tuple[str, str]:
    """Resolve HuggingFace dataset ID and language name for a given language code."""
    target_lang = target_lang.strip().lower()
    with open(REFERENCE_TABLE, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            if (
                row["Code"].strip().lower() == target_lang
                and row["source"].strip().lower() == "tatoeba"
            ):
                hf_id = row.get("Hugging face", "").strip()
                lang_name = row["Language"].strip()
                if hf_id:
                    return hf_id, lang_name
    raise FileNotFoundError(
        f"No HuggingFace dataset for language '{target_lang}' in {REFERENCE_TABLE}"
    )
