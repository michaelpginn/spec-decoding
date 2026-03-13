import argparse
import os
import sys

import n_gram
import torch
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from data_loading import data_prep


def get_arg():
    parser = argparse.ArgumentParser(description="N-gram experiment")

    parser.add_argument('--language', type=str, required=True, help="Language from hugging face dataset")

    return parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()  else "cpu")
    args = get_arg()
    language = args.language

    dataset = data_prep(language=language, text_type="mono")
    train, test = dataset.prepare_data()

    print(train)

    train = train[dataset.get_col_name(language)]
    test = test[dataset.get_col_name(language)]

    model_bigram = n_gram.ngram(
        2,
        "Qwen/Qwen-7B",
        device
    )
    model_trigram = n_gram.ngram(
        3,
        "Qwen/Qwen-7B",
        device
    )

    model_bigram.train(train,device)
    model_trigram.train(train, device)
    return 0


if __name__ == "__main__":
    main()
