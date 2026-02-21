import pandas as pd

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
    df = pd.read_csv("./src/data/reference_table_monolingual.csv").to_dict("records")
    print(df)
    # for language in languages:

    return 0


if __name__ == "__main__":
    main()
