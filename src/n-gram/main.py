import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

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
        dataset = data_prep(language=language, text_type="mono")
        train,test = dataset.prepare_data()
        train = [item[language] for item in train]
        test = [item[language] for item in test]
    return 0


if __name__ == "__main__":
    main()
