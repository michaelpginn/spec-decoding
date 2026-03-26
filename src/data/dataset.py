from pathlib import Path
from typing import Literal, cast

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

DATA_DIR = Path(__file__).resolve().parent

def assemble_dataset(language: str, type: Literal["mono", "bi"]):
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
    return dataset.train_test_split(test_size=0.2, seed=42)
