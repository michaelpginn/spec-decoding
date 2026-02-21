import pandas as pd
from hugging_face_data import get_data

languages: list = [
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


def main():
    df = pd.read_csv("../../src/data/reference_table_monolingual.csv").to_dict(
        "records"
    )
    for language in languages:
        data = get_data(language, "mono")
    return 0


if __name__ == "__main__":
    main()
