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
from tri_learning.models.char_lstm import CharLSTM
from tri_learning.datasets.char_dataset import CharDataset
from tri_learning.models.model import Model 

class CharacterLSTM(Model):
    def __init__(self, params):
        self.params = params

    def gen_data(self,tweets, stoi, max_len):
        x = []
        for tweet in tweets:
            data_point = []
            for ch in list(tweet):
                data_point.append(stoi[ch])
            if len(data_point) < max_len:
                data_point.extend([stoi['<PAD>']]*(max_len-len(data_point)))
            else:
                data_point = data_point[:max_len]
            x.append(data_point)   
        return torch.tensor(x)

    def gen_labels(self,labels):
        label_map= {label: i for i, label in enumerate(sorted(set(labels)))}
        return torch.tensor([[label_map[label]] for label in labels])

    def save_max_len(self,max_len, path):
        with open(path, 'w') as f:
            f.write(str(max_len))
    
    def load_max_len(self,path):
        with open(path, 'r') as f:
            max_len = int(f.read())
        return max_len

    def train_model(self,experiment,train_x,train_y,val_x,val_y):
        all_text = ''
        for tweet in train_x:
            all_text += ' ' + tweet
        chars = sorted(list(set(all_text)))
        stoi = {ch:i for i,ch in enumerate(chars)}
        itos = {i:ch for i,ch in enumerate(chars)}
        stoi['<PAD>'] = len(stoi)
        vocab_size = len(stoi)

        max_len = list(map(lambda t: len(t), train_x))
        max_len = round(np.mean(max_len) + 1.5*np.std(max_len))
 
        self.save_vocab(stoi, self.params['vocab_path'][experiment])
        self.save_max_len(max_len, self.params['max_len_path'][experiment])

        train_x = self.gen_data(train_x, stoi, max_len)
        train_y = self.gen_labels(train_y)
        val_x = self.gen_data(val_x, stoi, max_len)
        val_y = self.gen_labels(val_y)

        train_dataset = CharDataset(train_x, train_y)
        val_dataset = CharDataset(val_x, val_y)

        train_loader = DataLoader(train_dataset, batch_size=self.params['batch_size'])
        val_loader = DataLoader(val_dataset, batch_size=self.params['batch_size'])

        model = CharLSTM(input_size=vocab_size, 
                         output_size=1, 
                         hidden_size=self.params['hidden_size'], 
                         num_layers=self.params['num_layers']).cuda()

        loss_fn = nn.BCELoss()
        optimizer = Adam(model.parameters(), lr=self.params['lr'])
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
        stoi = self.load_saved_vocab(self.params['vocab_path'][experiment])
        vocab_size = len(stoi)

        max_len = self.load_max_len(self.params['max_len_path'][experiment])

        model = CharLSTM(input_size=vocab_size, 
                         output_size=1, 
                         hidden_size=self.params['hidden_size'], 
                         num_layers=self.params['num_layers']).cuda()

        self.load_model(model, self.params['model_path'][experiment])

        test_x = self.gen_data(test_x, stoi, max_len)
        test_y = self.gen_labels(test_y)

        test_dataset = CharDataset(test_x, test_y)
        test_loader = DataLoader(test_dataset, batch_size=self.params['batch_size'])

        results, preds = self.test(test_loader, model)
        return results, preds   
