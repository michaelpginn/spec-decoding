import itertools

import nltk
from nltk.corpus import wordnet as wn

try:
    wn.synsets('dog')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def create_prompt(language:str, adj_n:bool=False, num_prompts:int=10):
    prompts: dict = {}
    if not adj_n:
        nouns = list(wn.all_synsets('n'))
        sample_nouns = [synset.name().split('.')[0] for synset in nouns[:num_prompts]]

        for i in range(num_prompts):
            prompts[i+1] = f"Write a story in {language} about a(n) {sample_nouns[i]}"
    else:
        adjs = [s.name().split('.')[0] for s in itertools.islice(wn.all_synsets('a'), num_prompts)]
        nouns = [s.name().split('.')[0] for s in itertools.islice(wn.all_synsets('n'), num_prompts)]
        wombos = [f"{a} {n}" for a, n in zip(adjs, nouns)]

        for i in range(num_prompts):
            prompts[i+1] = f"Write a stroy in {language} about a(n) {wombos[i]}"

    return prompts
