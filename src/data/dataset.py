from pathlib import Path
from typing import Literal, cast

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

DATA_DIR = Path(__file__).resolve().parent

def assemble_dataset(language: str, type: Literal["mono", "bi"], dedup: bool = True):
    file = "reference_table_monolingual.csv" if type=="mono" else "reference_table_bilingual.csv"
    file_path = DATA_DIR / file
    df = pd.read_csv(file_path)
    df = df[df["Language"] == language]
    df = df[df["hugging face "].notna()]
    paths = df["hugging face "].tolist()
    datasets_list = [cast(Dataset, load_dataset(path, split="train")) for path in paths]
    dataset = concatenate_datasets(datasets_list)
    if language in dataset.column_names:
        dataset = dataset.rename_column(language, "text")
    dataset = dataset.filter(lambda row: row['text'])
    if dedup:
        before = len(dataset)
        seen: set[str] = set()
        def _is_unique(row):
            text = row["text"].strip()
            if text in seen:
                return False
            seen.add(text)
            return True
        dataset = dataset.filter(_is_unique)
        removed = before - len(dataset)
        if removed:
            import logging
            logging.getLogger(__name__).info(f"Dedup: removed {removed} duplicates ({before} -> {len(dataset)})")
    return dataset.train_test_split(test_size=0.2, seed=42)
