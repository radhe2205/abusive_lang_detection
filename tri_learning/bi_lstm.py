import json
import os
import sys
import argparse

sys.path.append(os.getcwd())

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from preprocessing import Preprocessor
from tri_learning.models.bi_rnn import RNNModel
from src.models.embedding import GloveEmbedding
from tri_learning.datasets.tweet_dataset import TweetDataset
from tri_learning.models.model import Model 

class BiLSTM(Model):
    def __init__(self, params):
        self.params = params

    def get_dataloader(self,tweets, labels, wordtoidx, batch_size):
        dataset = TweetDataset(tweets, labels, wordtoidx).cuda()
        return DataLoader(dataset = dataset, shuffle=False, batch_size = batch_size)

    def get_all_words_from_train(self,tweets):
        pp = Preprocessor()
        vocab = pp.gen_vocab(sentences=tweets)
        word_to_idx = pp.gen_word_to_idx(vocab=vocab)
        return word_to_idx

    def load_embeddings_n_words(self,tweets, embedding_path, embedding_type = "glove", embedding_dim = 50):
        wordtoidx = self.get_all_words_from_train(tweets)
        embedding = GloveEmbedding(embedding_dim, wordtoidx, embedding_path)
        wordtoidx = embedding.wordtoidx
        return embedding, wordtoidx
    
    def train_model(self,experiment,train_x,train_y,val_x,val_y):
        embedding, wordtoidx = self.load_embeddings_n_words(tweets=train_x, 
                                                            embedding_path=self.params['embedding_path'], 
                                                            embedding_dim=self.params['embedding_dim'])
        self.save_vocab(wordtoidx, self.params['vocab_path'][experiment])

        model = RNNModel(embeddings=embedding, 
                         in_dim=self.params['embedding_dim'],
                         num_layers=self.params['num_layers'],
                         hidden_size=self.params['hidden_size'], 
                         out_dim=1).cuda()

        train_loader = self.get_dataloader(tweets=train_x,
                                           labels=train_y, 
                                           wordtoidx=wordtoidx, 
                                           batch_size=self.params['batch_size'])

        val_loader = self.get_dataloader(tweets=val_x,
                                         labels=val_y, 
                                         wordtoidx=wordtoidx, 
                                         batch_size=self.params['batch_size'])

        optimizer = Adam(model.parameters(), lr=self.params['lr'])
        loss_fn = nn.BCELoss(reduction='sum')
        sched = ExponentialLR(optimizer, gamma=0.95)
        
        f1_scores = []
        best_f1 = 0
        for t in range(self.params['epochs']):
            print(f'epoch {t}')
            print('-'*40)
            self.train(train_loader, model, loss_fn, optimizer)
            stop, f1_score = self.validation(val_loader, model, loss_fn, f1_scores)
            f1_scores.append(f1_score)

            if f1_score > best_f1:
                self.save_model(model, self.params['model_path'][experiment])
            
            if t > 10 and stop: break

            sched.step()    
            print()

    def test_model(self,experiment,test_x,test_y):
        wordtoidx = self.load_saved_vocab(self.params['vocab_path'][experiment])
        embedding = GloveEmbedding(self.params['embedding_dim'], 
                                   wordtoidx, 
                                   self.params['embedding_path'], 
                                   False)

        model = RNNModel(embeddings=embedding, 
                         in_dim=self.params['embedding_dim'],
                         num_layers=self.params['num_layers'],
                         hidden_size=self.params['hidden_size'], 
                         out_dim=1).cuda()

        self.load_model(model, self.params['model_path'][experiment])

        test_loader = self.get_dataloader(tweets=test_x,
                                          labels=test_y, 
                                          wordtoidx=wordtoidx, 
                                          batch_size=self.params['batch_size'])

        results, preds = self.test(test_loader, model)
        return results, preds    