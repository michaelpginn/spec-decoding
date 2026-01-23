import pandas as pd

data = pd.read_csv(
    "data/bilingual/ber/tatoeba.tsv", sep="\t", header=None, usecols=[1, 3]
).to_dict(orient="records")

print(data)
