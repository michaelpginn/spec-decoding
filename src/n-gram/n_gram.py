# import math
from collections import defaultdict


# bigram model
class bigram:
    def __init__(self, train_list) -> None:
        self.train_text = train_list
        self.model = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.vocabulary = set()

    def ngram_model(self):
        for sentence in self.train_text:
            if sentence is not None:
                sentence = f'<s> {sentence} </s>'
                train_token = [token for token in sentence.split() if token != ""]

                gram = list(zip(train_token, train_token[1:]))

                for w1, w2 in gram:
                    self.model[w1][w2] += 1.0
                    self.vocabulary.add(w1)
                    self.vocabulary.add(w2)

        for word in self.model:
            total_count = float(sum(self.model[word].values()))
            for w2 in self.model[word]:
                self.model[word][w2] = (self.model[word][w2])/(total_count)
        return self.model

    def predict(self, word):
        self.next_word_pred = self.model.get(word)

        if self.next_word_pred and len(self.next_word_pred) > 0:
            return max(self.model[word], key=lambda k: self.model[word][k])
        else:
            return "<UNK>"

# trigram model
class trigram():
    def __init__(self, train_list) -> None:
        self.train_text = train_list
        self.model = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.vocabulary = set()

    def ngram_model(self):
        for sentence in self.train_text:
            if sentence is not None:
                sentence = f'<s> {sentence} </s>'
                self.train_token = [token for token in sentence.split() if token != ""]

                self.gram = list(zip(self.train_token, self.train_token[1:], self.train_token[2:]))

                for w1, w2, w3 in self.gram:
                    self.model[(w1, w2)][w3] += 1.0
                    self.vocabulary.add(w1)
                    self.vocabulary.add(w2)
                    self.vocabulary.add(w3)

        for w1_w2 in self.model:
            total_count = float(sum(self.model[w1_w2].values()))
            for w3 in self.model[w1_w2]:
                self.model[w1_w2][w3] /= total_count

    def predict(self, w1, w2):
        next_word_probs = self.model[w1, w2]

        if next_word_probs:
            return max(self.model[(w1,w2)], key=lambda k: self.model[(w1,w2)][k])
        else:
            return "<UNK>"
