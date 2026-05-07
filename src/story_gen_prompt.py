import itertools

import nltk
from nltk.corpus import wordnet as wn

try:
    wn.synsets('dog')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class create_prompt:
    def __init__(self, num_prompts: int = 10) -> None:
        self.num_prompts = num_prompts

    def get_combinations(self):
        adjs = [s.name().split('.')[0] for s in itertools.islice(wn.all_synsets('a'), self.num_prompts)]
        nouns = [s.name().split('.')[0] for s in itertools.islice(wn.all_synsets('n'), self.num_prompts)]

        return [f"{a} {n}" for a, n in zip(adjs, nouns)]

    def make_prompts(self, language, adj_n:bool=False):
        self.language = language
        self.prompts: dict = {}
        if not adj_n:
            nouns = list(wn.all_synsets('n'))
            sample_nouns = [synset.name().split('.')[0] for synset in nouns[:self.num_prompts]]

            for i in range(self.num_prompts):
                self.prompts[i+1] = f"Write a story in {self.language} about a(n) {sample_nouns[i]}"
        else:
            wombos = self.get_combinations() #wombo = word combos
            for i in range(self.num_prompts):
                self.prompts[i+1] = f"Write a stroy in {self.language} about a(n) {wombos[i]}"
        return self.prompts
