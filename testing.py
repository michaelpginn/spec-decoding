import pandas as pd

data = list(
    pd.read_csv("data/monolingual/haw/2grams_haw.txt", sep="\t")
    .to_dict(orient="dict")
    .values()
)
