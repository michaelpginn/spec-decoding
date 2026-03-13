from collections import defaultdict

import torch
from transformers import AutoTokenizer

"""
Would need to install tiktoken and transformers
"""
class ngram:
    def __init__(self, n, hug_tokenizer, device):
        '''
            takes tokenizer and checks if it is a huggingface and takes the tokenizer from it
            n is the number of gram.
        '''
        self.n = n
        self.gram_freq = defaultdict(lambda: defaultdict(lambda: 0.0))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(hug_tokenizer, trust_remote_code=True)
        except OSError:
            return "Not a hugging face tokenizer"
        self.vocabulary = defaultdict(lambda: defaultdict(lambda: 0.0))

    def train(self, train_list, device):
        '''
        making the n-gram from the training set
        '''
        train_list = train_list.to(device)
        for sentence in train_list:
            if sentence is not None:
                train_token = self.tokenizer.tokenize(sentence)
                train_token = [self.tokenizer.bos_token] + train_token + [self.tokenizer.eos_token]

                for idx in range(len(train_token) - self.n + 1):
                    context = tuple(train_token[idx: idx + self.n - 1])
                    try:
                        target = train_token[idx + self.n]
                    except IndexError:
                        target = train_token[idx]
                    self.gram_freq[context][target] += 1.0

        for key, value in self.gram_freq.items():
            key = key.to(device)
            value = value
            total_instances = float(sum(self.gram_freq[key].values()))
            if total_instances > 0:
                for inner, count in value.items():
                    self.vocabulary[key][inner] = count / total_instances
        return self.gram_freq, self.vocabulary

    def predict(self, input, device):
        '''
            tokenize input text
        '''
        input = input.to(device)
        if isinstance(input, str):
            token = self.tokenizer.tokenizer(input).to(device)
        else:
            token = input

        size = self.n-1 #gets the size for the lookup

        if len(token)<size:
            return self.tokenizer.unk_token


        probabilities = torch.zeros(len(self.vocabulary)).to(device)
        context = token[-size:].to(device)
        context_key = tuple(context)

        for token, prob in self.gram_freq[context_key]:
            token_id = self.tokenizer.id_for_token(token)
            probabilities[token_id] = prob

        return probabilities
