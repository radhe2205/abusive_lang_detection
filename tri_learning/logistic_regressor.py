import json
import os
import sys
import argparse

sys.path.append(os.getcwd())

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from preprocessing import Preprocessor
from tri_learning.models.logistic_regression import LogisticRegressionModel
from tri_learning.datasets.text_dataset import TextDataset
from tri_learning.models.model import Model 
import pickle

class LogisticRegressor(Model):
    def __init__(self, params):
        self.params = params

    def gen_labels(self,labels):
        label_map= {label: i for i, label in enumerate(sorted(set(labels)))}
        return torch.tensor([[label_map[label]] for label in labels])
    
    def load_vectorizer(self, path):
        with open(path, 'rb') as f:
            vectorizer = pickle.load(f)
        return vectorizer

    def save_vectorizer(self, path, vectorizer):
        with open(path, 'wb') as f:
            pickle.dump(vectorizer, f)

    def train_model(self,experiment,train_x,train_y,val_x,val_y):
        vectorizer = CountVectorizer(input='content',
                                     encoding='utf-8',
                                     analyzer='word',
                                     ngram_range=(1,3))   
    
        train_x = vectorizer.fit_transform(train_x)
        val_x = vectorizer.transform(val_x)
        train_y = self.gen_labels(train_y)
        val_y = self.gen_labels(val_y)

        self.save_vectorizer(self.params['vectorizer_path'][experiment], vectorizer)

        train_dataset = TextDataset(train_x, train_y)
        val_dataset = TextDataset(val_x, val_y)

        train_loader = DataLoader(train_dataset, batch_size=self.params['batch_size'])
        val_loader = DataLoader(val_dataset, batch_size=self.params['batch_size'])
        
        model = LogisticRegressionModel(input_dim=train_x.shape[1]).cuda()
        loss_fn = nn.BCELoss()
        optimizer = SGD(model.parameters(), lr=self.params['lr'], momentum=0.9)
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
        vectorizer = self.load_vectorizer(self.params['vectorizer_path'][experiment])

        test_x = vectorizer.transform(test_x)   
        test_y = self.gen_labels(test_y)   

        test_dataset = TextDataset(test_x, test_y)
        test_loader = DataLoader(test_dataset, batch_size=self.params['batch_size'])

        model = LogisticRegressionModel(input_dim=test_x.shape[1]).cuda()
        self.load_model(model, self.params['model_path'][experiment])

        results, preds = self.test(test_loader, model)
        return results, preds