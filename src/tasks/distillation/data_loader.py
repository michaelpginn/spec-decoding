"""
Load monolingual datasets for knowledge distillation.
"""
import logging

from datasets import load_dataset

from src.config.config import DistillConfig

logger = logging.getLogger(__name__)


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


def load_distillation_dataset(config: DistillConfig):
    """
    Load and prepare a monolingual dataset for distillation.

    Pipeline: load -> filter short texts -> deduplicate -> truncate to max_samples.
    """
    if config.dataset_path and config.dataset_path != "None":
        logger.info(f"Loading dataset from local file: {config.dataset_path}")
        ext = config.dataset_path.rsplit(".", 1)[-1]
        if ext in ("json", "jsonl"):
            fmt = "json"
        elif ext == "csv":
            fmt = "csv"
        else:
            raise ValueError(f"Unsupported file format: {config.dataset_path}")
        dataset = load_dataset(
            fmt,
            data_files=config.dataset_path,
            split=config.dataset_split,
            streaming=False,
        )
    else:
        cfg_info = f" (config: {config.dataset_config})" if config.dataset_config else ""
        logger.info(f"Loading dataset: {config.dataset_name}{cfg_info}")

        kwargs = {"split": config.dataset_split, "streaming": False}
        if config.dataset_config and config.dataset_config != "None":
            kwargs["name"] = config.dataset_config

        dataset = load_dataset(config.dataset_name, **kwargs)

    dataset = dataset.filter(
        lambda x: len(x[config.dataset_text_column]) > config.min_text_length
    )

    dataset = _deduplicate(dataset, config.dataset_text_column)

    if config.max_samples > 0 and len(dataset) > config.max_samples:
        dataset = dataset.select(range(config.max_samples))
        logger.info(f"Truncated dataset to {config.max_samples} examples")

    logger.info(f"Dataset ready: {len(dataset)} examples")
    return dataset
