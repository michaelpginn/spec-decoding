from collections import defaultdict

import nltk
from nltk import bigrams, trigrams


class bigram_model:
    def __init__(self, train, test) -> None:
        self.model = defaultdict(lambda: defaultdict(lambda: 0.0))

        for


    def training(self):
        for w1, w2 in self.bi_train:
            self.model[w1][w2] += 1.0

        for w in self.model:
            total_count = float(sum(self.model[w].values()))
            for w2 in self.model[w]:
                self.model[w][w2] /= total_count

    def predict(self, word1):
        self.next_word_pred = self.model[word1]
        if self.next_word_pred:
            return max(self.next_word_pred, key=self.next_word_pred.get)
        else:
            return "No prediction available"

class trigram_model:
    def __init__(self, train,test) -> None:
        self.train,self.test = [],[]
        for tr,te in zip(train,test):
            self.train.append(nltk.word_tokenize(' '.join(tr)))
            self.test.append(nltk.word_tokenize(' '.join(te)))
