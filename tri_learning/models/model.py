import os
import sys
import json

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
import torch.nn
from sklearn.metrics import classification_report

class Model(ABC):
    def __init__(self, params):
        self.params = params
        self.train_loss = []
        self.val_loss = []

        self.train_acc = []
        self.val_acc = []
    
    def reset_losses(self):
        self.train_loss = []
        self.val_loss = []

        self.train_acc = []
        self.val_acc = []   

    def save_model(self,model, model_path):
        directory_path = "/".join(model_path.split("/")[:-1])
        if len(directory_path) > 0:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

        torch.save(model.state_dict(), model_path)

    def load_model(self,model, model_path, flag=True):
        try:
            if not os.path.exists(model_path):
                return model

            model.load_state_dict(torch.load(model_path), strict=flag)
            return model
        except Exception as e:
            traceback.print_exc(e)
            print("Error occured while loading, ignoring...")

    def save_vocab(self,vocab, vocab_path):
        with open(vocab_path, "w") as f:
            f.write(json.dumps(vocab))

    def load_saved_vocab(self,vocab_path):
        with open(vocab_path, "r") as f:
            vocab = f.read()
            vocab = json.loads(vocab)
        return vocab

    def early_stop(self,f1_scores, latency):
        if len(f1_scores) < latency:
            return False

        window = f1_scores[-latency:]
        curr = window[-1]
        prevs = window[:-1]
        sma = np.mean(prevs)
        if curr < sma:
            return True
        return False
    
    def train(self,dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.train()
        train_loss, train_correct = 0,0
        for batch, (x,y) in enumerate(dataloader):
            if self.params['task'] == 'c':
                x, y = x.cuda(), y.cuda()
            else:
                x, y = x.cuda(), y.float().cuda()
            pred = model(x).cuda()
            loss = loss_fn(pred,y)
            
            train_loss += loss.item()

            if len(pred.shape) == 1:
                train_correct += (torch.round(pred)==y).float().sum().item()
            else:
                m = nn.Softmax(dim=0)
                train_correct += (y==m(pred).argmax(dim = -1)).float().sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= num_batches
        train_correct /= size
        self.train_loss.append(train_loss)
        self.train_acc.append(train_correct)

    def validation(self,dataloader, model, loss_fn, f1_scores):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0,0
        preds = torch.zeros(0).cuda()
        targets = torch.zeros(0).cuda()
        with torch.no_grad():
            for x,y in dataloader:
                if self.params['task'] == 'c':
                    x, y = x.cuda(), y.cuda()
                else:
                    x, y = x.cuda(), y.float().cuda()
                pred = model(x).cuda()
                test_loss += loss_fn(pred,y).item()

                if len(pred.shape) == 1:
                    correct += (torch.round(pred)==y).type(torch.float).sum().item()
                    preds = torch.cat((preds,torch.round(pred)), dim=0)
                    targets = torch.cat((targets,y), dim=0)
                else:
                    m = nn.Softmax(dim=0)
                    correct += (y==m(pred).argmax(dim = -1)).float().sum().item()
                    preds = torch.cat((preds,m(pred).argmax(dim = -1)), dim=0)
                    targets = torch.cat((targets,y), dim=0)                    

        test_loss /= num_batches
        correct /= size
        self.val_loss.append(test_loss)
        self.val_acc.append(correct)

        results = classification_report(y_true=targets.cpu(), y_pred=preds.cpu(), output_dict=True, digits=4)
        print('-'*40)
        print('testing')
        print(f'precision: \t{results["macro avg"]["precision"]}')
        print(f'recall: \t{results["macro avg"]["recall"]}')
        print(f'F1-score: \t{results["macro avg"]["f1-score"]}')
        print(f'validation error: \naccuracy {correct:>5f}, avg loss: {test_loss:>7f}')
        print('-'*40)
        return self.early_stop(f1_scores, latency=8), results["macro avg"]["f1-score"]

    def test(self,dataloader, model):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0,0
        preds = torch.zeros(0).cuda()
        targets = torch.zeros(0).cuda()
        with torch.no_grad():
            for x,y in dataloader:
                if self.params['task'] == 'c':
                    x, y = x.cuda(), y.cuda()
                else:
                    x, y = x.cuda(), y.float().cuda()
                pred = model(x).cuda()

                if len(pred.shape) == 1:
                    correct += (torch.round(pred)==y).type(torch.float).sum().item()
                    preds = torch.cat((preds,torch.round(pred)), dim=0)
                    targets = torch.cat((targets,y), dim=0)
                else:
                    m = nn.Softmax(dim=0)
                    correct += (y==m(pred).argmax(dim = -1)).float().sum().item()
                    preds = torch.cat((preds,m(pred).argmax(dim = -1)), dim=0)
                    targets = torch.cat((targets,y), dim=0)   

        correct /= size

        results = classification_report(y_true=targets.cpu(), y_pred=preds.cpu(), output_dict=True, digits=4)
        return {'f1-score:':results["macro avg"]["f1-score"],
                'accuracy':correct}, preds
    
    @abstractmethod
    def train_model(self,experiment,train_x,train_y,val_x,val_y):
        pass
    
    @abstractmethod
    def test_model(self,experiment,test_x,test_y):
        pass
