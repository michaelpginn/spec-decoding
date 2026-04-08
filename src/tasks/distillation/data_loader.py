"""
Data loading for distillation training.

- Task-specific (SeqKD): loads pre-generated teacher translations (bilingual).
- General: loads raw monolingual text for causal LM fine-tuning.
"""
import logging
from typing import Any, cast

from datasets import Dataset, load_dataset, load_from_disk
from transformers import PreTrainedTokenizer

from src.config.config import DistillConfig
from src.data.create_inputs import create_prompt
from src.data.hugging_face_data import get_data
from src.tasks.translation import _get_language_name

logger = logging.getLogger(__name__)


def load_seqkd_dataset(config: DistillConfig) -> Dataset:
    """
    Load the pre-generated teacher translation dataset.

    Returns a Dataset with columns: source, teacher_translation.
    """
    path = config.seqkd_data_path
    if not path or path == "None":
        raise ValueError(
            "seqkd_data_path must be set. "
            "Point it to a local path or HF dataset ID with teacher translations."
        )

    ds: Dataset
    if "/" in path and not path.startswith(".") and not path.startswith("/"):
        logger.info(f"Loading SeqKD dataset from HF: {path}")
        ds = cast(Dataset, load_dataset(path, split="train"))
    else:
        logger.info(f"Loading SeqKD dataset from disk: {path}")
        ds = cast(Dataset, load_from_disk(path))

    required = {"source", "teacher_translation"}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(
            f"SeqKD dataset missing columns: {missing}. "
            f"Available: {ds.column_names}"
        )

    if config.max_samples > 0 and len(ds) > config.max_samples:
        ds = ds.select(range(config.max_samples))
        logger.info(f"Truncated to {config.max_samples} examples")

    logger.info(f"SeqKD dataset loaded: {len(ds)} examples")
    return ds


def tokenize_seqkd(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    config: DistillConfig,
) -> Dataset:
    """
    Tokenize the SeqKD dataset with prompt masking.

    Builds the full chat string (prompt + translation) for each example,
    tokenizes it, and creates labels where prompt tokens are -100.
    """
    lang_name = _get_language_name(config.language_code)

    def _tokenize(examples):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for source, translation in zip(examples["source"], examples["teacher_translation"]):
            prompt_text = create_prompt("translation", lang_name, source)
            messages = [{"role": "user", "content": prompt_text}]

            prompt_str: str = cast(str, tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))
            full_str = prompt_str + translation + tokenizer.eos_token

            prompt_ids: Any = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_ids)

            tokenized: Any = tokenizer(
                full_str,
                truncation=True,
                max_length=config.max_length,
                padding="max_length",
                add_special_tokens=False,
            )

            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            labels = list(input_ids)
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100
            for i in range(len(labels)):
                if attention_mask[i] == 0:
                    labels[i] = -100

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }

    logger.info("Tokenizing SeqKD dataset with prompt masking...")
    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing SeqKD",
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def load_general_dataset(config: DistillConfig) -> Dataset:
    """
    Load monolingual text for the target language.

    Uses the monolingual reference table via hugging_face_data.get_data().
    Returns a Dataset with a 'text' column.
    """
    lang_name = _get_language_name(config.language_code)
    logger.info(f"Loading monolingual data for {lang_name} ({config.language_code})")

    ds = cast(Dataset, get_data(lang_name, "mono"))

    cols = ds.column_names
    text_col = None
    if "text" in cols:
        text_col = "text"
    elif lang_name in cols:
        text_col = lang_name
    elif len(cols) == 1:
        text_col = cols[0]
    else:
        for c in cols:
            if c.lower() not in {"id", "idx", "index"}:
                text_col = c
                break
    if text_col is None:
        raise ValueError(f"Cannot determine text column from {cols}")

    if text_col != "text":
        ds = ds.rename_column(text_col, "text")

    keep = {"text"} & set(ds.column_names)
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])

    ds = ds.filter(lambda row: bool(row["text"] and str(row["text"]).strip()))

    if config.max_samples > 0 and len(ds) > config.max_samples:
        ds = ds.select(range(config.max_samples))
        logger.info(f"Truncated to {config.max_samples} examples")

    logger.info(f"General dataset loaded: {len(ds)} examples")
    return ds


def tokenize_general(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    config: DistillConfig,
) -> Dataset:
    """
    Tokenize monolingual text for standard causal LM training.

    Every token is a label (no prompt masking). Padding tokens are set to -100.
    """

    def _tokenize(examples):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for text in examples["text"]:
            text = str(text).strip()
            if not text:
                continue

            eos: str = cast(str, tokenizer.eos_token)
            tokenized: Any = tokenizer(
                text + eos,
                truncation=True,
                max_length=config.max_length,
                padding="max_length",
                add_special_tokens=False,
            )

            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            labels = list(input_ids)
            for i in range(len(labels)):
                if attention_mask[i] == 0:
                    labels[i] = -100

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }

    logger.info("Tokenizing general dataset for causal LM training...")
    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing general",
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized
