import importlib
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

    def type_of_dataset(self) -> str | list:
        if self.dataset_type == "monolingual":
            data = [
                ref["path"]
                for ref in self.monolingual_reference
                if ref["Language"] == self.lang
            ]
            return data
        elif self.dataset_type == "bilingual":
            data = [
                ref["path"]
                for ref in self.bilingual_reference
                if ref["Language"] == self.lang
            ]
            return data
        else:
            return "invalid dataset type"

    def get_data(self):
        self.data = []
        if self.type_of_dataset() != "invalid dataset type":
            for item in self.type_of_dataset():
                if ".tsv" in item:
                    self.data.extend(self.deal_tsv(item))
                elif ".py" in item:
                    item_spec = importlib.util.spec_from_file_location(
                        "get_huggingface", item
                    )
                    if item_spec is None:
                        return f"cannot run {item}"
                    else:
                        module = importlib.util.module_from_spec(item_spec)
                        try:
                            self.data.extend(item_spec.loader.exec_module(module))
                        except Exception:
                            return f"cannot run {item}"
        return self.data

    def deal_tsv(self, path: str) -> list:
        self.df = pd.read_csv(path, sep="\t", header=None)[3].tolist()
        return self.df
