import json
import os
import sys
import argparse

sys.path.append(os.getcwd())

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ExponentialLR
import time
import string
from torch.distributions import Categorical
from preprocessing import Preprocessor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LanguageLSTM(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, num_layers, d=300):
        super(LanguageLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        
        self.embedding = nn.Embedding(in_dim, d)

        self.lstm = nn.LSTM(input_size=d, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.hidden = None

    def forward(self, input_seq):
        embedding = self.embedding(input_seq)
        if self.hidden:
            output, self.hidden = self.lstm(embedding, self.hidden)
        else:
            output, self.hidden = self.lstm(embedding)
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

        return self.linear(output)

def train(data, model, loss_fn, optimizer):
    i = 0
    iteration = 0
    losses = []
    	 
    while True:
        seq_len = np.random.randint(5,10)
        if i+seq_len+1 >= len(data):
            break

        sample, target = data[i:i+seq_len], data[i+1:i+1+seq_len]
        sample = sample.unsqueeze(0)
        output = model(sample)
        loss = loss_fn(output.squeeze(), target.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        i += seq_len
        
        if iteration % 10000 == 0:
            loss = loss.item()
            losses.append(loss)
            print(f'loss: {loss:>0.7f}')
        
        iteration += 1

    print(f'average loss: {np.mean(losses)}')

def gen_output(model, out_len, start_sample, temperature, itos):
    s = itos[start_sample.item()]
    index = start_sample

    for _ in range(out_len):
        index = index.unsqueeze(0)
        output = model(index)
        pred_vec = torch.nn.functional.softmax(output.squeeze()/temperature, dim=0)
        dist = Categorical(pred_vec)
        index = dist.sample()
        word = itos[index.item()]
        s += ' ' + word
        index = torch.tensor([index.item()]).to(device)
    
    print(s)

def train_model(training_set, temp, save_path):
    hidden_size = 256
    num_layers = 2
    lr = 0.001

    model = LanguageLSTM(in_dim=num_words, out_dim=num_words, hidden_size=hidden_size, num_layers=num_layers).to(device)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
       
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, lr=lr)
    sched = ExponentialLR(optimizer, gamma=0.95)


    epochs = 100
    for t in range(epochs):
        print(f'epoch {t}\n-----------------------------')
        train(training_set, model, loss_fn, optimizer)
        
        sched.step()
            
        # save model
        torch.save(model.state_dict(), save_path)
        
        # input as random from training data
        idx = np.random.randint(len(training_set)-1)
        c = training_set[idx:idx+1]
        print()
        print('-'*40)
        gen_output(model, out_len=100, start_sample=c, temperature=temp, itos=itos)
        print('-'*40)
        print()

def test_model(training_set, temp, itos, load_path):
    hidden_size = 256
    num_layers = 2

    model = LanguageLSTM(in_dim=num_words, out_dim=num_words, hidden_size=hidden_size, num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(load_path))

    idx = np.random.randint(len(training_set)-1)
    c = training_set[idx:idx+1]
    gen_output(model, out_len=1000, start_sample=c, temperature=temp, itos=itos)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action='store', dest='mode', default='train', type=str)
    parser.add_argument('--model-path', action='store', dest='model_path', default='saved_models/model.pth', type=str)
    parser.add_argument('--temperature', action='store', dest='temp', default=1.0, type=float)
    res = parser.parse_args()
    
    pp = Preprocessor()
    OLID_train_tweets, OLID_train_labels = pp.get_train_data('data/OLIDv1.0/olid-training-v1.0_clean.tsv', 
                                                             sample=1,
                                                             seed=1)
    text = []
    for tweet in OLID_train_tweets:
        text.extend(tweet.split(' '))
    
    itos = {i:s for i,s in enumerate(sorted(set(text)))}
    itos['<unk>'] = len(itos)
    itos['<pad>'] = len(itos)
    stoi = {s:i for i,s in enumerate(sorted(set(text)))}

    training_set = torch.Tensor([stoi[x] for x in text]).type(torch.LongTensor).to(device)
    num_words = len(itos)

    if res.mode == 'train':
        train_model(training_set=training_set, temp=res.temp, save_path=res.model_path)
    else:
        test_model(training_set=training_set, temp=res.temp, itos=itos, load_path=res.model_path)
