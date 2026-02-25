from collections import defaultdict

import torch
from transformers import AutoTokenizer

"""
Would need to install tiktoken and transformers
"""
# bigram model
class bigram:
    def __init__(self, train_list, tokenizer) -> None:
        self.train_text = train_list
        self.model = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.vocabulary = set()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def train(self):
        for sentence in self.train_text:
            if sentence is not None:
                train_token = self.tokenizer.tokenize(sentence)
                train_token = [self.tokenizer.bos_token] + train_token + [self.tokenizer.eos_token]

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
            return self.tokenizer.unk_token

    def perplexity(self, test_data):
        log_prob_sum = 0
        word_count = 0

        epsilon = 1e-10

        for sentence in test_data:
            token = self.tokenizer.tokenize(sentence)
            tokens = [self.tokenizer.bos_token] + token + [self.tokenizer.eos_token]

            grams = list(zip(tokens, tokens[1:]))

            for w1, w2 in grams:
                word_count += 1
                prob = torch.tensor(self.model.get((w1), {}).get(w2, epsilon))
                log_prob_sum += torch.log2(prob)

        if word_count == 0:
            return float('inf')

        return 2 ** (-(log_prob_sum / word_count))

# trigram model
class trigram():
    def __init__(self, train_list, tokenizer) -> None:
        self.train_text = train_list
        self.model = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.vocabulary = set()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def train(self):
        for sentence in self.train_text:
            if sentence is not None:
                train_token = self.tokenizer.tokenize(sentence)
                train_token = [self.tokenizer.bos_token] + train_token + [self.tokenizer.eos_token]

                self.gram = list(zip(train_token, train_token[1:], train_token[2:]))

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
            return self.tokenizer.unk_token

    def perplexity(self, test_data):
        log_prob_sum = 0
        word_count = 0

        epsilon = 1e-10

        for sentence in test_data:
            token = self.tokenizer.tokenize(sentence)
            tokens = [self.tokenizer.bos_token] + token + [self.tokenizer.eos_token]

            grams = list(zip(tokens, tokens[1:], tokens[2:]))

            for w1, w2, w3 in grams:
                word_count += 1
                prob = torch.tensor(self.model.get((w1, w2), {}).get(w3, epsilon))
                log_prob_sum += torch.log2(prob)

        if word_count == 0:
            return float('inf')

        return 2 ** (-(log_prob_sum / word_count))
