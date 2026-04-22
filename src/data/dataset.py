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

    df = df[df["Language"] == language]
    df = df[df["hugging face "].notna()]
    paths = df["hugging face "].tolist()

    """
    Just like how load data was but checking for the new cherokee data
    """
    dataset_list = []

    for path in paths:
        if ':' in path:
            repo, config = path.split(":", 1)
            ds = cast(Dataset, load_dataset(repo, config, split="train"))
        else:
            ds = cast(Dataset, load_dataset(path, split="train"))

        current_cols = ds.column_names
        if "text" not in current_cols:
            for col in [language, language.lower(), "sentence", "content"]:
                if col in current_cols:
                    ds = ds.rename_column(col, "text")
                    break
        dataset_list.append(ds)

    """
    Handling the aya data
    """
    lang_aya = cast(Dataset, aya.filter(lambda x: x["language"] == language.lower()))
    if include_aya:
        if not dataset_list:
            dataset = lang_aya
        else:
            other_datasets = concatenate_datasets(dataset_list)
            dataset = concatenate_datasets([lang_aya, other_datasets])
    else:
        if not dataset_list:
            raise ValueError(f"No datasets found for {language} and include_aya is False.")
        dataset = concatenate_datasets(dataset_list)

    dataset = dataset.filter(lambda row: row['text'])
    return dataset.train_test_split(test_size=0.2, seed=42)
