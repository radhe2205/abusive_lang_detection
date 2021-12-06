import sys
import torch
from torch import nn
import numpy as np

# Base class, should contain common methods
# These methods are used from the program, any embedding fastext/glove provides implementation for these.
class WordEmbedding(nn.Module):
    def __init__(self):
        super(WordEmbedding, self).__init__()

    def load_embeddings(self, file_path, wordtoidx, *kwargs):
        return None

    def get_embeddings_w_words(self, words):
        return None

    def get_embeddings(self, idxes, w_lens):
        return None

class GloveEmbedding(WordEmbedding):
    def __init__(self, embedding_dim, wordtoidx, embedding_path, reload = True):
        super(GloveEmbedding, self).__init__()

        self.embedding_dim = embedding_dim
        self.wordtoidx = wordtoidx if wordtoidx is not None else {}

        self.fixed_embedding = nn.Embedding(400002, embedding_dim, padding_idx=-1).requires_grad_(False)

        # One index to be used for padding
        if len(wordtoidx) > 400001:
            self.trainable_embedding = nn.Embedding(len(wordtoidx.keys()) + 1 - 400001, embedding_dim, padding_idx=-1).requires_grad_(True)

        # Whether the embedding be loaded from model.
        if reload:
            if embedding_path is not None:
                self.load_embeddings(embedding_path, wordtoidx, embedding_dim)

    def get_embeddings_w_words(self, words):
        return self.embeddings[[self.wordtoidx[word] for word in words]]

    def get_embeddings(self, idxes):
        fixed_idxes = idxes.clone()
        train_idxes = idxes.clone()

        # fixed_idxes[idxes > 400001] = self.fixed_embedding.padding_idx
        fixed_idxes[idxes > 400001] = self.wordtoidx["<unk>"]
        fixed_idxes[idxes < 0] = self.fixed_embedding.padding_idx

        train_idxes = train_idxes - 400001
        train_idxes[train_idxes < 0] = self.trainable_embedding.padding_idx

        return self.fixed_embedding(fixed_idxes) #+ self.trainable_embedding(train_idxes)

    # form_word_idx_dict: used for when we do not want to train extra word embeddings and just want to utilize glove embeddings.
    def load_embeddings(self, embedding_path, wordtoidx, embedding_dim = 50):
        with torch.no_grad():
            wordtoidx_mod = {}
            print("Loading embeddings started.")
            word_count = 0
            with open(embedding_path, "r", encoding="utf-8") as f:
                for line in f:
                    vals = line.split()
                    word = vals[0]
                    vector = torch.from_numpy(np.asarray(vals[1:], "float32"))
                    self.fixed_embedding.weight[word_count] = vector
                    wordtoidx_mod[word] = word_count
                    word_count += 1

            for word in wordtoidx:
                if word in wordtoidx_mod:
                    continue
                wordtoidx_mod[word] = word_count
                word_count += 1

            # Last index is padding
            self.trainable_embedding = nn.Embedding(word_count - 400001 + 1, embedding_dim, padding_idx=-1).requires_grad_(True)
            for word in wordtoidx_mod:
                wordtoidx[word] = wordtoidx_mod[word]

# glove_embeddings = GloveEmbedding(400001, 50, {})
# glove_embeddings.load_embeddings(file_path="data/glove822/glove.6B.50d.txt")
# torch.save(glove_embeddings.state_dict(), "saved_models/all_embedding.model")
# print("Done")
