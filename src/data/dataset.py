from pathlib import Path
from typing import Literal, cast

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

DATA_DIR = Path(__file__).resolve().parent

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
