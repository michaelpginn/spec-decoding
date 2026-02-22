from pathlib import Path
from typing import cast

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

DATA_DIR = Path(__file__).resolve().parent
def load_data(df: list, lang: str, type: str):
    land_df = [d for d in df if d["Language"] == lang]
    if lang == "Chinese" and type == "monolingual":
        hugging = [item["hugging face "] for item in land_df]
        loading_data = cast(Dataset, load_dataset(hugging[0], "en-zh"))
        return loading_data
    elif len(land_df) == 2:
        hugging = [
            item["hugging face "]
            for item in land_df
            if item["hugging face "] is not None
        ]
        loading_data_1 = cast(Dataset, load_dataset(hugging[0], split="train"))
        loading_data_2 = cast(Dataset, load_dataset(hugging[1], split="train"))

        loading_data = concatenate_datasets([loading_data_1, loading_data_2])
        return loading_data
    else:
        hugging = [
            item["hugging face "]
            for item in land_df if item["hugging face "] is not None
        ]
        loading_data = cast(Dataset, load_dataset(hugging[0], split="train"))
        return loading_data


def get_data(lang: str, mono_or_bi: str):
    file = "reference_table_monolingual.csv" if mono_or_bi=="mono" else "reference_table_bilingual.csv"
    file_path = DATA_DIR / file
    try:
        df = pd.read_csv(file_path).to_dict("records")
        return load_data(df, lang, mono_or_bi)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file at: {file_path}")
