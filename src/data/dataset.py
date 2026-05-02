from collections import Counter
import csv
import json
import logging
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from datasets import Dataset, Features, Value, concatenate_datasets, load_dataset

DATA_DIR = Path(__file__).resolve().parent
REFERENCE_TABLE = DATA_DIR / "reference_table_bilingual.csv"
logger = logging.getLogger(__name__)


def get_raw_url(url: str) -> str:
    """Converts a GitHub blob URL to a raw content URL."""
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url


def get_language_name(lang_code: str) -> str:
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


def load_with_max(repo, config, split, max_samples: int | None):
    if max_samples is None:
        return cast(Dataset, load_dataset(repo, config, split=split))
    stream = load_dataset(repo, config, split=split, streaming=True).take(max_samples) # type:ignore
    return Dataset.from_list(list(stream))


mono_features = Features({"text": Value("string"), "source": Value("string")})

def standardize_columns_mono(ds: Dataset, language: str, lang_code: str):
    # Column standardization logic
    current_cols = ds.column_names
    if "text" not in current_cols:
        # Expanded search list to include 'Mayan', 'Source', and 'Target'
        search_cols = [
            language, lang_code, language.lower(),
            "Mayan", "Mayan language",  # Specific to yua datasets
            "sentence", "text_sentence", "content", "Article"
            "Source", "Target","inputs"         # Common in parallel-formatted mono data
        ]
        for col in search_cols:
            if col in current_cols:
                ds = ds.rename_column(col, "text")
                break
        else:
            raise AssertionError("Could not find matching column")
    ds = ds.select_columns(["text", "source"])
    ds = ds.filter(lambda row: row['text'])
    return ds.cast(mono_features)


def standardize_columns_bi(ds: Dataset, language: str, lang_code: str):
    if 'english' in ds.column_names:
        ds = ds.rename_column('english', "English")
    if language.lower() in ds.column_names:
        ds = ds.rename_column(language.lower(), language)
    ds = ds.select_columns(['English', language, "source"])
    bi_features = Features({"English": Value("string"), language: Value("string"), "source": Value("string")})
    return ds.cast(bi_features)


def assemble_dataset(lang_code: str, type: Literal["mono", "bi"], max_samples: int | None = None, include_aya=True):
    file = "reference_table_monolingual.csv" if type=="mono" else "reference_table_bilingual.csv"
    file_path = DATA_DIR / file
    language = get_language_name(lang_code)
    df = pd.read_csv(file_path)
    paths = df[(df["Language"] == language) & (df["hugging face"].notna())]
    dataset_list = []
    for _, row in paths.iterrows():
        path = row["hugging face"]
        assert isinstance(path, str)

        if str(path).startswith("http"):
            raw_url = get_raw_url(str(path))
            sep = '\t' if raw_url.endswith('.tsv') or 'tatoeba' in raw_url.lower() else ','
            temp_df = pd.read_csv(raw_url, sep=sep, storage_options={'ssl': False})
            ds = Dataset.from_pandas(temp_df)
        else:
            # HF dataset
            repo = path
            config = None
            split_to_load = "train"
            if ':' in path:
                parts = path.split(":")
                repo = parts[0]
                config = parts[1]
                if len(parts) > 2:
                    split_to_load = parts[2]

            if path == 'Helsinki-NLP/opus-100':
                ds = load_with_max(repo, config, "train", max_samples)
                def map(r):
                    d = json.loads(r['translation'])
                    return {"English": d['en'], language: d[lang_code]}
                ds = ds.map(map)

            for split in [split_to_load, 'full', lang_code]:
                try:
                    ds = load_with_max(repo, config, split, max_samples)
                    break
                except:
                    continue
            else:
                raise ValueError(f"No split matching {[split_to_load, 'full', lang_code]} in {repo}")

        # Add sources
        if "Source" in ds.column_names:
            ds = ds.rename_column('Source', 'source')
        elif "source" not in ds.column_names:
            ds = ds.map(lambda r: {"source": path})

        if type == 'mono':
            ds = standardize_columns_mono(ds, language, lang_code)
        else:
            ds = standardize_columns_bi(ds, language, lang_code)
        dataset_list.append(ds)

    if type == 'mono' and include_aya:
        aya_dataset = load_dataset("CohereLabs/aya_dataset", split="train")
        aya_dataset = cast(Dataset, aya_dataset.filter(lambda x: x["language"].lower() == language.lower()))
        if len(aya_dataset) != 0:
            current_cols = aya_dataset.column_names
            search_cols = ["inputs", "targets", "text", "sentence"]
            for col in search_cols:
                if col in current_cols:
                    aya_dataset = aya_dataset.rename_column(col, "text")
                    break
            aya_dataset = aya_dataset.map(lambda r: {"source": "CohereLabs/aya_dataset"})
            aya_dataset = aya_dataset.select_columns(["text", "source"]).cast(mono_features)
            dataset_list.append(aya_dataset)

    if not dataset_list:
        raise ValueError(f"No datasets found for {language}")
    dataset: Dataset = concatenate_datasets(dataset_list)
    if max_samples and max_samples > 0 and len(dataset) > max_samples:
        old_size = len(dataset)
        dataset = dataset.shuffle(42).select(range(max_samples))
        logger.info(f"Filtered full dataset from {old_size} to {max_samples} examples")
    logger.info(f"Data source breakdown: {Counter(dataset['source'])}")
    splits = dataset.train_test_split(test_size=0.2, seed=42)
    logger.info(f"Data splits: {splits}")
    return splits
