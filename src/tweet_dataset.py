import math

import pandas
import torch
from torch import nn


class TweetDataset(nn.Module):
    # The word_to_idx is of the word2vec
    def __init__(self, tweets, labels, wordtoidx):
        super(TweetDataset, self).__init__()
        self.max_len = 90
        label_idx = {label: i for i, label in enumerate(sorted(set(labels)))}
        self.tweets = torch.empty((len(tweets), self.max_len)).long().fill_(-1)
        self.labels = torch.zeros(len(tweets))
        if len(label_idx.keys()) > 2:
            self.labels = self.labels.long()

        for idx, (tweet, label) in enumerate(zip(tweets, labels)):
            for word_idx, word in enumerate(tweet.split()):
                if word_idx > (self.max_len - 1):
                    print("Caution: Max word length limit breached.")
                    break
                word = word if word in wordtoidx else "<unk>"
                self.tweets[idx, word_idx] = wordtoidx[word]
            self.labels[idx] = label_idx[label]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        return self.tweets[item], self.labels[item]
