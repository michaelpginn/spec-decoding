from collections import defaultdict


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
