import pandas as pd
from datasets import load_dataset, concatenate_datasets


def get_data(lang: str, mono_or_bi: str):
    if mono_or_bi == "mono":
        df = pd.read_csv("reference_table_monolingual.csv")
        land_df = [data for data in df if data["Language"] == lang]
        if len(land_df) == 0:
            return "Wrong input"
        elif len(land_df) > 0:
            hugging = [
                item["hugging face "]
                for item in land_df
                if not pd.isna(item["hugging face "])
            ]
            loading_data_1, loading_data_2 = (
                load_dataset(hugging[0]),
                load_dataset(hugging[1]),
            )
            loading_data = concatenate_datasets([loading_data_1, loading_data_2])
            return loading_data
        else:
            hugging = [item["hugging face "] for item in land_df]
            return hugging

    elif mono_or_bi == "bi":
        df = pd.read_csv("reference_table_bilingual.csv")
    else:
        return "Wrong input"
