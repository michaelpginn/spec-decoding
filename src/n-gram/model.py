import math
from collections import defaultdict


# bigram model
class bigram:
    def __init__(self, train_list, test_list) -> None:
        self.train_text = train_list
        self.test_text = test_list
        self.model = defaultdict(lambda: defaultdict(lambda: 0.0))

    def ngram_model(self):
        for sentence in self.train_text:
            self.train_token = [token for token in sentence.split() if token != ""]

            self.gram = list(zip(self.train_token, self.train_token[1:]))

            for w1, w2 in self.gram:
                self.model[(w1)][w2] += 1.0

        for word in self.model:
            total_count = float(sum(self.model[word].values()))
            for w2 in self.model[word]:
                self.model[word][w2] /= total_count
        return self.model

    def predict(self, word):
        self.next_word_pred = self.model.get(word)

        if self.next_word_pred and len(self.next_word_pred) > 0:
            return max(self.model[word], key=lambda k: self.model[word][k])
        else:
            return "No prediction available"

    def perplexity(self):
        self.log_prob = 0
        self.word_count = 0
        self.epsilon = 1e-10

        for sentence in self.train_text:
            self.train_token = [token for token in sentence.split() if token != ""]

            self.gram = list(zip(self.train_token, self.train_token[1:]))

            for w1, w2 in self.gram:
                self.model[(w1)][w2] += 1.0

# trigram model
class trigram():
    def __init__(self, train_list, test_list) -> None:
        self.train_text = train_list
        self.test_text = test_list
        self.model = defaultdict(lambda: defaultdict(lambda: 0.0))

    def ngram_model(self):
        for sentence in self.train_text:
            self.train_token = [token for token in sentence.split() if token != ""]

            self.gram = list(zip(self.train_token, self.train_token[1:], self.train_token[2:]))

            for w1, w2, w3 in self.gram:
                self.model[(w1, w2)][w3] += 1.0

        for w1_w2 in self.model:
            total_count = float(sum(self.model[w1_w2].values()))
            for w3 in self.model[w1_w2]:
                self.model[w1_w2][w3] /= total_count

    def predict(self, w1, w2):
        next_word_probs = self.model[w1, w2]
        if next_word_probs:
            return max(self.model[(w1,w2)], key=lambda k: self.model[(w1,w2)][k])
        else:
            return "No prediction available"

    def perplexity(self):
        self.log_prob = 0
        self.word_count = 0
        self.epsilon = 1e-10

        for sentence in self.train_text:
            self.train_token = [token for token in sentence.split() if token != ""]

            self.gram = list(zip(self.train_token, self.train_token[1:], self.train_token[2:]))

            for w1, w2, w3 in self.gram:
                self.word_count += 1
                prob = self.model.get((w1, w2), {}).get(w3, self.epsilon)

                self.log_prob += math.log2(prob)

        if self.word_count==0:
            return float('inf')

        return 2**(-(self.log_prob/self.word_count))
