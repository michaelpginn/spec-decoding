import nltk
from nltk import bigrams, trigrams


class bigram_model:
    def __init__(self, train, test) -> None:
        self.train,self.test = [],[]
        for tr,te in zip(train,test):
            self.train.append(nltk.word_tokenize(' '.join(tr)))
            self.test.append(nltk.word_tokenize(' '.join(te)))

class trigram_model:
    def __init__(self, train,test) -> None:
        self.train,self.test = [],[]
        for tr,te in zip(train,test):
            self.train.append(nltk.word_tokenize(' '.join(tr)))
            self.test.append(nltk.word_tokenize(' '.join(te)))
