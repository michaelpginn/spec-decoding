"""
Load translation data from various sources.
"""

import csv
import logging
from typing import cast

from datasets import DatasetDict
import sacrebleu

from src.config.config import DistillConfig, ExperimentConfig
from src.data.dataset import assemble_dataset, get_language_name

logger = logging.getLogger(__name__)


def load_data(config: ExperimentConfig | DistillConfig, tokenizer) -> tuple[DatasetDict, str]:
    """
    Load (source, target) pairs from the bilingual train split.

    Returns:
        - Dataset with 'source' and 'target' column
        - Language name
    """
    max_samples = config.max_samples if config.max_samples > 0 else None
    dataset = assemble_dataset(config.language_code, 'bi', tokenizer, max_samples)
    lang_name = get_language_name(config.language_code)

    columns = cast(list[str], dataset["train"].column_names)
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
    dataset = dataset.rename_columns({
        src_col: "source",
        tgt_col: "target"
    })
    dataset = dataset.filter(lambda row: row['source'].strip() and row['target'].strip())
    return dataset, lang_name


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
