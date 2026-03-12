import argparse
import os
import sys

import n_gram
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from data_loading import data_prep


def main():
    dataset = data_prep(language=language, text_type="mono")
    train,test = dataset.prepare_data()

    print(train)

    train = train[dataset.get_col_name(language)]
    test = test[dataset.get_col_name(language)]

    model_bigram = n_gram.ngram(2, "Qwen/Qwen-7B",)
    model_trigram = n_gram.ngram(3, "Qwen/Qwen-7B",)

    model_bigram.train(train)
    model_trigram.train(train)
    return 0


if __name__ == "__main__":
    main()
