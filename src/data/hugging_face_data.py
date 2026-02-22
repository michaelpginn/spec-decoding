import pandas as pd
from datasets import concatenate_datasets, load_dataset


def load_data(df: list, lang: str, type: str):
    land_df = [d for d in df if d["Language"] == lang]
    if len(land_df) == 0:
        return "Wrong input"
    if lang == "Chinese" and type == "monolingual":
        hugging = [item["hugging face "] for item in land_df]
        loading_data = load_dataset(hugging[0], "en-zh")
        return loading_data
    elif len(land_df) == 2:
        hugging = [
            item["hugging face "]
            for item in land_df
            if item["hugging face "] is not None
        ]
        loading_data_1 = load_dataset(hugging[0], split="train")
        loading_data_2 = load_dataset(hugging[1], split="train")

        loading_data = concatenate_datasets([loading_data_1, loading_data_2])
        return loading_data
    else:
        hugging = [
            item["hugging face "]
            for item in land_df if item["hugging face "] is not None
        ]
        loading_data = load_dataset(hugging[0])
        return loading_data


def get_data(lang: str, mono_or_bi: str):
    if mono_or_bi == "mono":
        df = pd.read_csv("reference_table_monolingual.csv").to_dict("records")
        return load_data(df, lang, mono_or_bi)

    elif mono_or_bi == "bi":
        df = pd.read_csv("reference_table_bilingual.csv").to_dict("records")
        return load_data(df, lang, mono_or_bi)
    else:
        return "Wrong input"
