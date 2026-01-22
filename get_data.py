import pandas as pd


class get_data:
    def __init__(self, lang: str, dataset_type: str):
        self.lang = lang
        self.dataset_type = dataset_type
        self.bilingual_reference = pd.read_csv("reference_table_bilingual.csv").to_dict(
            "records"
        )
        self.monolingual_reference = pd.read_csv(
            "reference_table_monolingual.csv"
        ).to_dict("records")

    def get_data(self):
        if self.dataset_type == "monolingual":
            data = [
                ref["path"]
                for ref in self.monolingual_reference
                if ref["Language"] == self.lang
            ]
        elif self.dataset_type == "bilingual":
            data = [
                ref["path"]
                for ref in self.bilingual_reference
                if ref["Language"] == self.lang
            ]
        else:
            return "invalid dataset type"
