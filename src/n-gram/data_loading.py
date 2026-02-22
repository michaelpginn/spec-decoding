import pandas as pd
from hugging_face_data import get_data


class data_prep:
    def __init__(self):
        self.languages: list = [
            "Berber",
            "Cherokee",
            "Hawaiian",
            "Igbo",
            "Lakota",
            "Muskogee(Creek)",
            "Nepali",
            "Occitan",
            "Occitan",
            "Ojibwe",
            "Quechua",
            "Maya",
            "Tamazight",
            "Chinese",
        ]

        def retrieve_data(self):
            self.df = pd.read_csv(
                "../../src/data/reference_table_monolingual.csv"
            ).to_dict("records")
            for language in self.languages:
                self.data = get_data(language, "mono")
