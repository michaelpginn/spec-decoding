import os
import sys

import model
import pandas as pd

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
    "Muskogee (Creek)",
    "Nepali",
    "Occitan",
    "Ojibwe",
    "Quechua",
    "Maya",
    "Tamazight"
]

def main():
    csv_file = []
    for language in languages:
        dataset = data_prep(language=language, text_type="mono")
        train,test = dataset.prepare_data()

        print(train)
        available_cols = train.column_names
        if "text" in available_cols:
            target_col = "text"
        elif language in available_cols:
            target_col = language
        else:
            target_col = available_cols[0]

        train = train[target_col]
        test = test[target_col]

        model_bigram = model.bigram(train, test).perplexity()
        model_trigram = model.trigram(train,test).perplexity()

        csv_file.append({"language":language, "bigram perplexity":model_bigram,"trigram perplexity":model_trigram})

    df = pd.DataFrame(csv_file)
    df.to_csv("n-gram_perplexity.csv",index=False)
    return 0


if __name__ == "__main__":
    main()
