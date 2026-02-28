from collections import defaultdict

from transformers import AutoTokenizer

"""
Would need to install tiktoken and transformers
"""
class ngram:
    def __init__(self, n, hug_tokenizer):
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

    def train(self, train_list):
        '''
        making the n-gram from the training set
        '''
        for sentence in train_list:
            if sentence is not None:
                train_token = self.tokenizer.tokenize(sentence)
                train_token = [self.tokenizer.bos_token] + train_token + [self.tokenizer.eos_token]

                for idx in range(len(train_token) - self.n + 1):
                  context = tuple(train_token[idx: idx + self.n - 1])
                  target = train_token[idx + self.n]
                  self.gram_freq[context][target] += 1.0

        for key, value in self.gram_freq.items():
            total_instances = float(sum(self.gram_freq[key].values()))
            if total_instances > 0:
                for inner, count in value.items():
                    self.vocabulary[key][inner] = count / total_instances
        return self.gram_freq, self.vocabulary

    def predict(self, input):
        '''
            tokenize input text
        '''
        if isinstance(input, str):
            token = self.tokenizer.tokenizer(input)
        else:
            token = input

        size = self.n-1 #gets the size for the lookup

        if len(token)<size:
            return self.tokenizer.unk_token

        context = token[-size:]
        context_key = tuple(context)

        pred = self.gram_freq.get(context_key)

        if pred:
            return max(pred, key=lambda k: pred[k])
        else:
            return self.tokenizer.unk_token
