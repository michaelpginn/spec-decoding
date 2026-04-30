import csv
import logging
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from datasets import Dataset, Features, Value, concatenate_datasets, load_dataset

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


def get_raw_url(url: str) -> str:
    """Converts a GitHub blob URL to a raw content URL."""
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url

def get_raw_url(url: str) -> str:
    """Converts a GitHub blob URL to a raw content URL."""
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url

def assemble_dataset(language: str, type: Literal["mono", "bi"], include_aya:bool):
    file = "reference_table_monolingual.csv" if type=="mono" else "reference_table_bilingual.csv"
    file_path = DATA_DIR / file
    df = pd.read_csv(file_path)
    aya = load_dataset("CohereLabs/aya_dataset", split="train")

    paths = df[(df["Language"] == language) & (df["hugging face "].notna())]

    """
    Just like how load data was but checking for the new cherokee data
    """
    dataset_list = []

    for _, row in paths.iterrows():
        path = row["hugging face "]
        lang_code = str(row["Code"])

        if str(path).startswith("http"):
            raw_url = get_raw_url(str(path))
            sep = '\t' if raw_url.endswith('.tsv') or 'tatoeba' in raw_url.lower() else ','
            temp_df = pd.read_csv(raw_url, sep=sep, storage_options={'ssl': False})
            ds = Dataset.from_pandas(temp_df)

        else:
            repo = path
            config = None
            split_to_load = "train"

            if ':' in path:
                parts = path.split(":")
                repo = parts[0]
                config = parts[1]
                if len(parts) > 2:
                    split_to_load = parts[2]

            try:
                ds = cast(Dataset, load_dataset(repo, config, split=split_to_load))
            except ValueError as e:
                # Check the error message for available splits
                available_splits = str(e)

                # Sequence of fallbacks
                if split_to_load == "train":
                    if "full" in available_splits:
                        ds = cast(Dataset, load_dataset(repo, config, split="full"))
                    elif lang_code in available_splits:
                        ds = cast(Dataset, load_dataset(repo, config, split=lang_code))
                    else:
                        raise e
                else:
                    raise e

        # Column standardization logic
        current_cols = ds.column_names
        if "text" not in current_cols:
            # Expanded search list to include 'Mayan', 'Source', and 'Target'
            search_cols = [
                language, lang_code, language.lower(),
                "Mayan", "Mayan language",  # Specific to yua datasets
                "sentence", "text_sentence", "content",
                "Source", "Target","inputs"          # Common in parallel-formatted mono data
            ]
            for col in search_cols:
                if col in current_cols:
                    ds = ds.rename_column(col, "text")
                    break

        if "text" in ds.column_names:
            ds = ds.select_columns(["text"])
            dataset_list.append(ds)
        else:
            print(f"Warning: Could not find text column in {repo}. Available: {current_cols}")

    """
    Handling the aya data
    """

    standard_features = Features({"text": Value("string")})
    dataset_list = [ds.cast(standard_features) for ds in dataset_list]

    lang_aya = cast(Dataset, aya.filter(lambda x: x["language"].lower() == language.lower()))
    if include_aya:
        current_cols = lang_aya.column_names
        search_cols = ["inputs", "targets", "text", "sentence"]
        for col in search_cols:
            if col in current_cols:
                lang_aya = lang_aya.rename_column(col, "text")
                break

        lang_aya = lang_aya.select_columns(["text"]).cast(standard_features)

        if not dataset_list:
            dataset = lang_aya
        else:
            dataset = concatenate_datasets([lang_aya] + dataset_list)
    else:
        if not dataset_list:
            raise ValueError(f"No datasets found for {language} and include_aya is False.")
        dataset = concatenate_datasets(dataset_list)

    dataset = dataset.filter(lambda row: row['text'])
    return dataset.train_test_split(test_size=0.2, seed=42)
