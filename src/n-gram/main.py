from data_loading import data_prep

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
    for language in languages:
        data = data_prep(language=language, text_type="mono")
        train,test = data.prepare_data()
        train = [item for item in train]
        test = [item for item in test]
    return 0


if __name__ == "__main__":
    main()
