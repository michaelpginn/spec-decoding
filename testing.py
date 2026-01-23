import pandas as pd

data = pd.read_csv("data/bilingual/ber/tatoeba.tsv", sep="\t", header=None)

print(data[3].tolist())
