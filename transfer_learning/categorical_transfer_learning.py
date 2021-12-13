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
from src.models.embedding import GloveEmbedding
from src.tweet_dataset import TweetDataset
from model import Model 
from sklearn.model_selection import train_test_split
from collections import Counter

class CatTransferLSTM(nn.Module):
    def __init__(self, embeddings, in_dim, num_layers = 1, hidden_size = 100, out_dim = 1):
        super(CatTransferLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.out_dim = out_dim

        self.embeddings = embeddings

        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.lstm = nn.LSTM(input_size=hidden_size*2, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        self.linear_layers_new = nn.Sequential(
            nn.BatchNorm1d(32 * 2),
            nn.Linear(32 * 2, self.out_dim),
            nn.Sigmoid()
        )

        # Weight Initialization
        for layer in self.linear_layers_new:
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer.weight.data)

    # number of words in tweet are limited, will use padded fixed length sequence.
    def forward(self, samples):
        word_embs = self.embeddings.get_embeddings(samples)
        o, (h,c) = self.rnn(word_embs)
        o, (h,c) = self.lstm(o)
        o = torch.cat((o[:,-1,:32], o[:, 0, 32:]), dim=-1)
        return self.linear_layers_new(o)

class CatLSTMTransferModel(Model):
    def __init__(self, params):
        super().__init__(params)
        self.params = params

    def get_dataloader(self,tweets, labels, wordtoidx, batch_size):
        dataset = TweetDataset(tweets, labels, wordtoidx).cuda()
        return DataLoader(dataset = dataset, shuffle=False, batch_size = batch_size)

    def get_all_words_from_train(self,tweets):
        wordtoidx = self.load_saved_vocab(self.params['vocab_path'])
        return wordtoidx

    def load_embeddings_n_words(self,tweets, embedding_path, embedding_type = "glove", embedding_dim = 50):
        wordtoidx = self.get_all_words_from_train(tweets)
        embedding = GloveEmbedding(embedding_dim, wordtoidx, embedding_path)
        wordtoidx = embedding.wordtoidx
        return embedding, wordtoidx
    
    def train_model(self,train_x,train_y,val_x,val_y):
        embedding, wordtoidx = self.load_embeddings_n_words(tweets=train_x, 
                                                            embedding_path=self.params['embedding_path'], 
                                                            embedding_dim=self.params['embedding_dim'])

        model = CatTransferLSTM(embeddings=embedding, 
                                in_dim=self.params['embedding_dim'],
                                num_layers=self.params['num_layers'],
                                hidden_size=self.params['hidden_size'], 
                                out_dim=self.params['out_dim']).cuda()

        self.load_model(model, self.params['model_pretrain_path'], flag=False)

        train_loader = self.get_dataloader(tweets=train_x,
                                           labels=train_y, 
                                           wordtoidx=wordtoidx, 
                                           batch_size=self.params['batch_size'])

        val_loader = self.get_dataloader(tweets=val_x,
                                         labels=val_y, 
                                         wordtoidx=wordtoidx, 
                                         batch_size=self.params['batch_size'])

        optimizer = Adam(model.parameters(), lr=self.params['lr'])
        optimizer = Adam([
            {'params':model.rnn.parameters(), 'lr':0.00001},
            {'params':model.embeddings.parameters(), 'lr':0.00001},
            {'params':model.linear_layers_new.parameters(), 'lr':0.01},
            {'params':model.lstm.parameters(), 'lr':0.01}
        ],
        lr=5e-8)
        
        loss_fn = nn.CrossEntropyLoss() if self.params['task'] == 'c' else nn.BCELoss()
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
                self.save_model(model, self.params['model_path'])
            
            if t > 10 and stop: break

            sched.step()    
            print()

    def test_model(self,test_x,test_y):
        embedding, wordtoidx = self.load_embeddings_n_words(tweets=test_x, 
                                                            embedding_path=self.params['embedding_path'], 
                                                            embedding_dim=self.params['embedding_dim'])

        model = CatLSTM(embeddings=embedding, 
                         in_dim=self.params['embedding_dim'],
                         num_layers=self.params['num_layers'],
                         hidden_size=self.params['hidden_size'], 
                         out_dim=self.params['out_dim']).cuda()

        self.load_model(model, self.params['model_path'])

        test_loader = self.get_dataloader(tweets=test_x,
                                          labels=test_y, 
                                          wordtoidx=wordtoidx, 
                                          batch_size=self.params['batch_size'])

        results, preds = self.test(test_loader, model)
        return results, preds    

if __name__ == "__main__":
    train_options = {
        "train_data_path": "data/OLIDv1.0/olid-training-v1.0_clean.tsv",
        "test_tweet_path": "data/OLIDv1.0/testset-levela_clean.tsv",
        "test_label_path": "data/OLIDv1.0/labels-levela.csv",
        "sample_size":1,
        "seed":1
    }   
    params = {
        'model_pretrain_path':'model_cat.pth',
        'model_path':'model_cat_a.pth',
        'vocab_path':'model_vocab_cat.json',
        'embedding_path':'data/glove822/glove.6B.300d.txt',
        'embedding_dim':300,
        'num_layers':2,
        'hidden_size':256,
        'batch_size':32,
        'lr':0.01,
        'epochs':100,
        'task':'a',
        'out_dim':1
    }
    
    model = CatLSTMTransferModel(params=params)

    pp = Preprocessor()
    OLID_train_tweets, OLID_train_labels = pp.get_train_data(train_options["train_data_path"], 
                                                             task='subtask_a',
                                                                sample=train_options['sample_size'],
                                                                seed=train_options['seed'])
    OLID_train_tweets, OLID_val_tweets, OLID_train_labels, OLID_val_labels = train_test_split(OLID_train_tweets,
                                                                                                OLID_train_labels,
                                                                                                test_size=0.1,
                                                                                                stratify=OLID_train_labels,
                                                                                                random_state=1)
    OLID_test_tweets, OLID_test_labels = pp.get_test_data(train_options['test_tweet_path'],
                                                            train_options['test_label_path'])
    
    model.train_model(OLID_train_tweets, OLID_train_labels, OLID_val_tweets, OLID_val_labels)
    results, _ = model.test_model(OLID_test_tweets, OLID_test_labels)
    print(results)
