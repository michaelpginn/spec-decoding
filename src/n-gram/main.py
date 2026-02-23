import os
import sys

import model

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

    dataset = data_prep(language=languages[0], text_type="mono")
    train,test = dataset.prepare_data()
    train = [item[languages[0]] for item in train]
    test = [item[languages[0]] for item in test]

    model_bigram = model.bigram(train, test).ngram_model()
    # print(model_bigram)
    return 0


if __name__ == "__main__":
    main()
