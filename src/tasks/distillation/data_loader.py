"""
Load monolingual datasets for knowledge distillation.
"""
import csv
import logging
from pathlib import Path

from datasets import Dataset, load_dataset

from src.config.config import DistillConfig

logger = logging.getLogger(__name__)

_REFERENCE_TABLE = (
    Path(__file__).resolve().parent.parent.parent / "data" / "reference_table_monolingual.csv"
)


def _get_text_column(lang_code: str) -> str:
    """Map a language code to the dataset column name via the reference table."""
    lang_code = lang_code.strip().lower()
    with open(_REFERENCE_TABLE, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["Code"].strip().lower() == lang_code:
                return row["Language"].strip()
    raise KeyError(
        f"Language code '{lang_code}' not found in {_REFERENCE_TABLE}. "
        f"Add it to the reference table or use dataset_text_column override."
    )


def _resolve_dataset_name(config: DistillConfig) -> str:
    if config.dataset_name and config.dataset_name != "None":
        return config.dataset_name
    return f"lecslab/monoling_{config.language_code}"


def _load_raw_dataset(config: DistillConfig, *, streaming: bool):
    """Load the raw HuggingFace dataset."""
    name = _resolve_dataset_name(config)
    logger.info(f"Loading dataset: {name}")
    kwargs = {"split": config.dataset_split, "streaming": streaming}
    if config.dataset_config and config.dataset_config != "None":
        kwargs["name"] = config.dataset_config
    return load_dataset(name, **kwargs)


def _stream_and_collect(config: DistillConfig, text_col: str) -> Dataset:
    """
    Stream the dataset, filtering and deduplicating on the fly,
    stopping as soon as max_samples unique rows are collected.
    """
    stream = _load_raw_dataset(config, streaming=True)

    seen: set[str] = set()
    kept: list[dict] = []
    n_short = 0
    n_dupes = 0

    for row in stream:
        text = row[text_col]
        if len(text) <= config.min_text_length:
            n_short += 1
            continue
        if text in seen:
            n_dupes += 1
            continue
        seen.add(text)
        kept.append(row)
        if len(kept) >= config.max_samples:
            break

    if n_short:
        logger.info(f"Skipped {n_short} short examples (len <= {config.min_text_length})")
    if n_dupes:
        logger.info(f"Skipped {n_dupes} duplicate examples")

    return Dataset.from_list(kept)


def _deduplicate(dataset, text_column: str):
    """Remove duplicate texts, keeping the first occurrence."""
    seen: set[str] = set()
    keep_indices: list[int] = []
    for i, text in enumerate(dataset[text_column]):
        if text not in seen:
            seen.add(text)
            keep_indices.append(i)

    n_dupes = len(dataset) - len(keep_indices)
    if n_dupes > 0:
        logger.info(f"Removed {n_dupes} duplicate examples")
        dataset = dataset.select(keep_indices)
    return dataset


def load_distillation_dataset(config: DistillConfig) -> tuple[Dataset, str]:
    """
    Load and prepare a monolingual dataset for distillation.

    Returns (dataset, text_column).
    """
    if config.dataset_name and config.dataset_name != "None":
        text_col = "text"
    else:
        text_col = _get_text_column(config.language_code)
    logger.info(f"Using text column: '{text_col}'")

    if config.max_samples > 0:
        dataset = _stream_and_collect(config, text_col)
    else:
        dataset = _load_raw_dataset(config, streaming=False)
        dataset = dataset.filter(
            lambda x: len(x[text_col]) > config.min_text_length
        )
        dataset = _deduplicate(dataset, text_col)

    logger.info(f"Dataset ready: {len(dataset)} examples")
    return dataset, text_col
