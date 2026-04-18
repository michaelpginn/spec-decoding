import csv
import logging
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

DATA_DIR = Path(__file__).resolve().parent
REFERENCE_TABLE = DATA_DIR / "reference_table_bilingual.csv"
logger = logging.getLogger(__name__)


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


def load_bilingual_dataset(
    language_code: str,
    max_samples: int | None = None,
) -> DatasetDict:
    """
    Load the bilingual dataset for a language and split into train/test.

    Returns a DatasetDict with 'train' and 'test' splits (80/20, seed=42).
    If max_samples is set, the full dataset is truncated before splitting.
    """
    hf_id, lang_name = _get_hf_dataset_id(language_code)
    logger.info(f"Loading bilingual data from HuggingFace: {hf_id}")

    ds = cast(Dataset, load_dataset(hf_id, split="train"))

    if max_samples and max_samples > 0 and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    splits = ds.train_test_split(test_size=0.2, seed=42)
    logger.info(
        f"Bilingual split: {len(splits['train'])} train, {len(splits['test'])} test"
    )
    return splits


def assemble_dataset(language: str, type: Literal["mono", "bi"], include_aya:bool):
    file = "reference_table_monolingual.csv" if type=="mono" else "reference_table_bilingual.csv"
    file_path = DATA_DIR / file
    df = pd.read_csv(file_path)
    aya = load_dataset("CohereLabs/aya_dataset", split="train")
    if include_aya:
        if language not in df["Language"].values:
            dataset = cast(Dataset, aya.filter(lambda lang: lang["language"] == language))
        else:
            df = df[df["Language"] == language]
            df = df[df["hugging face "].notna()]
            paths = df["hugging face "].tolist()

            lang_aya = cast(Dataset, aya.filter(lambda lang: lang["language"] == language))
            datasets_list = [load_dataset(path, split="train") for path in paths]
            other_datasets = concatenate_datasets(datasets_list)
            dataset = concatenate_datasets([lang_aya, other_datasets])
    else:
        df = df[df["Language"] == language]
        df = df[df["hugging face "].notna()]
        paths = df["hugging face "].tolist()
        datasets_list = [cast(Dataset, load_dataset(path, split="train")) for path in paths]
        dataset = concatenate_datasets(datasets_list)
    if language in dataset.column_names:
            dataset = dataset.rename_column(language, "text")
    dataset = dataset.filter(lambda row: row['text'])
    return dataset.train_test_split(test_size=0.2, seed=42)
