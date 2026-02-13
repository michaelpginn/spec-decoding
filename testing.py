import pandas as pd

df = pd.read_csv("reference_table_monolingual.csv").to_dict(orient="records")
land_df = [data for data in df if data["Language"] == "Hawaiian"]
hugging = [
    item["hugging face "] for item in land_df if not pd.isna(item["hugging face "])
]
print(hugging)
