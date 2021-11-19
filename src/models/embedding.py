import sys
import torch
from torch import nn
import numpy as np

# Base class, should contain common methods
# These methods are used from the program, any embedding fastext/glove provides implementation for these.
class WordEmbedding(nn.Module):
    def __init__(self):
        super(WordEmbedding, self).__init__()

    def load_embeddings(self, file_path):
        return None

    def get_embeddings_w_words(self, words):
        return None

class GloveEmbedding(WordEmbedding):
    def __init__(self, word_count, embedding_dim, wordtoidx, trainable = False):
        super(GloveEmbedding, self).__init__()

        self.word_count = word_count
        self.embedding_dim = embedding_dim
        self.wordtoidx = wordtoidx if wordtoidx is not None else {}

        self.embeddings = nn.Embedding(self.word_count, self.embedding_dim).requires_grad_(trainable)

    def get_embeddings_w_words(self, words):
        return self.embeddings[[self.wordtoidx[word] for word in words]]

    # form_word_idx_dict: used for when we do not want to train extra word embeddings and just want to utilize glove embeddings.
    def load_embeddings(self, file_path, form_word_idx_dict = True):
        print("Loading embeddings started.")
        word_count = 0
        with torch.no_grad():
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    vals = line.split()
                    word = vals[0]

                    vector = torch.from_numpy(np.asarray(vals[1:], "float32"))
                    if form_word_idx_dict:
                        word_idx = word_count
                        self.wordtoidx[word] = word_idx
                    else:
                        word_idx = self.wordtoidx[word] if word in self.wordtoidx else self.wordtoidx["<unk>"]
                    self.embeddings.weight.data[word_idx] = vector

                    word_count += 1
        print("Loading embeddings complete.")

# glove_embeddings = GloveEmbedding(400001, 50, {})
# glove_embeddings.load_embeddings(file_path="data/glove822/glove.6B.50d.txt")
# torch.save(glove_embeddings.state_dict(), "saved_models/all_embedding.model")
# print("Done")
