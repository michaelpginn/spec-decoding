from pathlib import Path
from typing import cast

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

DATA_DIR = Path(__file__).resolve().parent
def load_data(df: list, lang: str, type: str):
    land_df = [d for d in df if lang in d["Language"]]

    if lang == "Chinese" and type == "mono":
        hugging_paths = [item["hugging face "] for item in land_df if pd.notna(item["hugging face "])]
        if not hugging_paths:
            raise ValueError(f"No Hugging Face path found for {lang}")
        return cast(Dataset, load_dataset(hugging_paths[0]))

    hugging_paths = [
        item["hugging face "]
        for item in land_df
        if pd.notna(item["hugging face "]) and item["hugging face "] != ""
    ]

    if not hugging_paths:
        raise ValueError(f"No valid Hugging Face datasets found for language: {lang}")

    datasets_list = [cast(Dataset, load_dataset(path, split="train")) for path in hugging_paths]

    if len(datasets_list) > 1:
        return concatenate_datasets(datasets_list)
    return datasets_list[0]


def get_data(lang: str, mono_or_bi: str):
    file = "reference_table_monolingual.csv" if mono_or_bi=="mono" else "reference_table_bilingual.csv"
    file_path = DATA_DIR / file
    try:
        df = pd.read_csv(file_path).to_dict("records")
        return load_data(df, lang, mono_or_bi)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file at: {file_path}")
