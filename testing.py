import pandas as pd

data = pd.read_csv("data/bilingual/chr/dictionary.csv").to_dict(orient="records")

print(data)
