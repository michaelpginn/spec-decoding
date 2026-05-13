import itertools
import os

import nltk
import pandas as pd
from nltk.corpus import wordnet as wn

try:
    wn.synsets('dog')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def create_prompt(language_code:str, adj_n:bool=False, num_prompts:int=10):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    language = language_code
    if len(language_code) == 3:
        try:
            biling_path = os.path.join(data_dir, "reference_table_bilingual.csv")
            monoling_path = os.path.join(data_dir, "reference_table_monolingual.csv")

            biling = pd.read_csv(biling_path)
            monoling = pd.read_csv(monoling_path)

            concat = pd.concat([biling, monoling])[['Language', "Code"]].drop_duplicates()
            final_set = set(map(tuple, concat.values))

            language = next((item[0] for item in final_set if item[1] == language_code), language_code)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            language = language_code
    prompts: dict = {}
    if not adj_n:
        nouns = list(wn.all_synsets('n'))
        sample_nouns = [synset.name().split('.')[0] for synset in nouns[:num_prompts]]

        for i, noun in enumerate(sample_nouns):
                   prompts[i+1] = f"Write a story in {language} about a(n) {noun}"
    else:
        adjs = [s.name().split('.')[0] for s in itertools.islice(wn.all_synsets('a'), num_prompts)]
        nouns = [s.name().split('.')[0] for s in itertools.islice(wn.all_synsets('n'), num_prompts)]

        for i in range(min(len(adjs), len(nouns), num_prompts)):
            prompts[i+1] = f"Write a story in {language} about a(n) {adjs[i]} {nouns[i]}"

    return prompts
