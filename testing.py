import pandas as pd

bilingual_reference = pd.read_csv("reference_table_bilingual.csv").to_dict("records")

data = [ref["path"] for ref in bilingual_reference if ref["Language"] == "Berber"]
print(data)
