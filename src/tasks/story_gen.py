"""
Story generation task — load monolingual data as prompt seeds.
"""

import logging
from typing import cast

from datasets import DatasetDict

from src.config.config import ExperimentConfig
from src.data.dataset import assemble_dataset, get_language_name

logger = logging.getLogger(__name__)


def load_data(config: ExperimentConfig, tokenizer) -> tuple[DatasetDict, str]:
    """
    Load monolingual text as story seeds.

    Each example gets a 'source' column (the mono text used as a prompt seed)
    and a dummy empty 'target' column (no reference for generation tasks).
    """
    max_samples = config.max_samples if config.max_samples > 0 else None
    dataset = assemble_dataset(config.language_code, 'mono', tokenizer, max_samples)
    lang_name = get_language_name(config.language_code)

    dataset = dataset.rename_column("text", "source")
    dataset = dataset.map(lambda r: {"target": ""})
    dataset = dataset.remove_columns([c for c in cast(list[str], dataset["train"].column_names)
                                      if c not in ("source", "target")])
    logger.info(f"Loaded {len(dataset['test'])} story-gen test examples for {lang_name}")
    return dataset, lang_name


def compute_eval_metrics(
    references: list[str], hypotheses: list[str], verbose: bool = False
) -> dict:
    """No reference-based metrics for story generation."""
    return {}
