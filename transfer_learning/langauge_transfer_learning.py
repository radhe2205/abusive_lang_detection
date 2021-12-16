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
from tri_learning.datasets.tweet_dataset import TweetDataset
from tri_learning.models.model import Model 
from preprocessing import Preprocessor
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TransferLSTM(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, hidden_size_2, num_layers, d=300):
        super(TransferLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden_size_2 = hidden_size_2
        self.out_dim = out_dim
        
        self.embedding = nn.Embedding(in_dim, d, padding_idx=-1)

        self.lstm = nn.LSTM(input_size=d, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm_top = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size_2, num_layers=1, bidirectional=True, batch_first=True)
        self.linear_layer = nn.Sequential(
            nn.Linear(2*hidden_size_2, out_dim),
            [nn.Sigmoid, nn.Identity()][out_dim >= 2]
        )

    def forward(self, input_seq):
        input_seq[input_seq==-1] = self.embedding.padding_idx
        embedding = self.embedding(input_seq)
        o, hidden_state = self.lstm(embedding)
        o, hidden_state = self.lstm_top(o)
        output = torch.cat((o[:,-1,:self.hidden_size_2], o[:, 0, self.hidden_size_2:]), dim=-1)
        output = self.linear_layer(output)
        return output

class TransferLearningModel(Model):
    def __init__(self, params):
        super().__init__(params)
        self.params = params

    def get_dataloader(self,tweets, labels, wordtoidx, batch_size):
        dataset = TweetDataset(tweets, labels, wordtoidx).to(device)
        return DataLoader(dataset = dataset, shuffle=False, batch_size = batch_size)

    def get_all_words_from_train(self,tweets):
        wordtoidx = self.load_saved_vocab(self.params['vocab_path'])
        return wordtoidx
    
    def train_model(self,train_x,train_y,val_x,val_y):
        wordtoidx = self.get_all_words_from_train(np.concatenate((train_x,val_x)))

        self.save_vocab(wordtoidx, self.params['vocab_path'])

        model = TransferLSTM(in_dim=len(wordtoidx),
                            out_dim=3,
                            num_layers=2,
                            hidden_size=256,
                            hidden_size_2=32).to(device)

        self.load_model(model, self.params['model_pretrain_path'], flag=False)

        model.lstm.requires_grad = False
        model.embedding.requires_grad = False

        train_loader = self.get_dataloader(tweets=train_x,
                                           labels=train_y, 
                                           wordtoidx=wordtoidx, 
                                           batch_size=32)

        val_loader = self.get_dataloader(tweets=val_x,
                                         labels=val_y, 
                                         wordtoidx=wordtoidx, 
                                         batch_size=32)

        optimizer = Adam([
            {'params':model.lstm.parameters(), 'lr':0.00001},
            {'params':model.embedding.parameters(), 'lr':0.00001},
            {'params':model.lstm_top.parameters(), 'lr':0.01},
            {'params':model.linear_layer.parameters(), 'lr':0.01}
        ],
        lr=5e-8)
        loss_fn = nn.CrossEntropyLoss() if self.params['task'] == 'c' else nn.BCELoss()
        sched = ExponentialLR(optimizer, gamma=0.95)
        
        f1_scores = []
        best_f1 = 0
        for t in range(100):
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
        wordtoidx = self.load_saved_vocab(self.params['vocab_path'])

        model = TransferLSTM(in_dim=len(wordtoidx),
                            out_dim=3,
                            num_layers=2,
                            hidden_size=256,
                            hidden_size_2=32).to(device)

        self.load_model(model, self.params['model_path'])

        test_loader = self.get_dataloader(tweets=test_x,
                                          labels=test_y, 
                                          wordtoidx=wordtoidx, 
                                          batch_size=32)

        results, preds = self.test(test_loader, model)
        return results, preds    

if __name__ == "__main__":
    train_options = {
        "train_data_path": "data/OLIDv1.0/olid-training-v1.0_clean.tsv",
        "test_tweet_path": "data/OLIDv1.0/testset-levelc_clean.tsv",
        "test_label_path": "data/OLIDv1.0/labels-levelc.csv",
        "sample_size":1,
        "seed":1
    }   
    params = {
        'model_pretrain_path':'model.pth',
        'model_path':'model_trained.pth',
        'vocab_path':'model_vocab.json',
        'task':'c'
    }

    model = TransferLearningModel(params=params)
    
    pp = Preprocessor()
    OLID_train_tweets, OLID_train_labels = pp.get_train_data(train_options["train_data_path"], 
                                                             task='subtask_c',
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