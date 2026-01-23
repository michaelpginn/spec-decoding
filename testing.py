import pandas as pd

data = pd.read_csv(
    "data/monolingual/haw/2grams_haw.txt", sep="\t", usecols=[1]
).to_dict(orient="records")

print(data)
