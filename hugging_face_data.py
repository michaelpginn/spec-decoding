import pandas as pd


def get_data(lang: str, mono_or_bi: str):
    if mono_or_bi == "mono":
        pd.read_csv("reference_table_monolingual.csv")
    else:
        pd.read_csv("reference_table_bilingual.csv")
