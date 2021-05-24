import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    
    def item(self):
        return self.word2idx

    def __len__(self):
        return len(self.idx2word)


'''
    tokenize the sentences
'''
class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        # train dataset
        self.train = self.tokenize(os.path.join(path, 'sentences_whole_train_K6.txt'))
        # valid dataset
        self.valid = self.tokenize(os.path.join(path, 'sentences_whole_val_K6.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
    def get_dict(self):
        return self.dictionary

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = ['<cls>'] + line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = ['<cls>'] + line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        return ids
