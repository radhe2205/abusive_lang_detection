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
from tri_learning.datasets.tweet_dataset import TweetDataset
from tri_learning.models.model import Model 
from sklearn.model_selection import train_test_split
from collections import Counter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CatLSTM(nn.Module):
    def __init__(self, embeddings, in_dim, num_layers = 1, hidden_size = 100, out_dim = 1):
        super(CatLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.out_dim = out_dim

        self.embeddings = embeddings

        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.5)

        self.linear_layers = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 2),
            nn.Linear(self.hidden_size * 2, self.out_dim),
        )

        # Weight Initialization
        for layer in self.linear_layers:
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer.weight.data)

    # number of words in tweet are limited, will use padded fixed length sequence.
    def forward(self, samples):
        word_embs = self.embeddings.get_embeddings(samples)
        o, (h,c) = self.rnn(word_embs)
        o = torch.cat((o[:,-1,:self.hidden_size], o[:, 0, self.hidden_size:]), dim=-1)
        return self.linear_layers(o)

class CatLSTMModel(Model):
    def __init__(self, params):
        super().__init__(params)
        self.params = params

    def get_dataloader(self,tweets, labels, wordtoidx, batch_size):
        dataset = TweetDataset(tweets, labels, wordtoidx).to(device)
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
    
    def train_model(self,train_x,train_y,val_x,val_y):
        embedding, wordtoidx = self.load_embeddings_n_words(tweets=train_x, 
                                                            embedding_path=self.params['embedding_path'], 
                                                            embedding_dim=self.params['embedding_dim'])
        self.save_vocab(wordtoidx, self.params['vocab_path'])

        model = CatLSTM(embeddings=embedding, 
                         in_dim=self.params['embedding_dim'],
                         num_layers=self.params['num_layers'],
                         hidden_size=self.params['hidden_size'], 
                         out_dim=self.params['out_dim']).to(device)

        train_loader = self.get_dataloader(tweets=train_x,
                                           labels=train_y, 
                                           wordtoidx=wordtoidx, 
                                           batch_size=self.params['batch_size'])

        val_loader = self.get_dataloader(tweets=val_x,
                                         labels=val_y, 
                                         wordtoidx=wordtoidx, 
                                         batch_size=self.params['batch_size'])

        optimizer = Adam(model.parameters(), lr=self.params['lr'])
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
        wordtoidx = self.load_saved_vocab(self.params['vocab_path'])
        embedding = GloveEmbedding(self.params['embedding_dim'], 
                                   wordtoidx, 
                                   self.params['embedding_path'], 
                                   False)

        model = CatLSTM(embeddings=embedding, 
                         in_dim=self.params['embedding_dim'],
                         num_layers=self.params['num_layers'],
                         hidden_size=self.params['hidden_size'], 
                         out_dim=self.params['out_dim']).to(device)

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
        "test_tweet_path": "data/OLIDv1.0/testset-levelc_clean.tsv",
        "test_label_path": "data/OLIDv1.0/labels-levelc.csv",
        "sample_size":1,
        "seed":1
    }   
    params = {
        'model_path':'model_cat.pth',
        'vocab_path':'model_vocab_cat.json',
        'embedding_path':'data/glove822/glove.6B.300d.txt',
        'embedding_dim':300,
        'num_layers':2,
        'hidden_size':256,
        'batch_size':32,
        'lr':0.0001,
        'epochs':100,
        'task':'c'
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', action='store', dest='mode', default='test', type=str)
    res = parser.parse_args()
    
    data = pd.read_csv(train_options['train_data_path'], sep='\t')
    data = data[data['tweet'].notna()]
    tweets = data['tweet'].values
    labels = data[['subtask_a', 'subtask_b', 'subtask_c']]
    labels.loc[labels['subtask_b'].isna(), 'subtask_b'] = 'NONE'
    labels.loc[labels['subtask_c'].isna(), 'subtask_c'] = 'NONE'
    labels = labels.values
    labels = [tuple(l) for l in labels]
    label_keys = {l:i for i,l in enumerate(sorted(set(labels)))}
    labels = np.array([label_keys[l] for l in labels])
    params['out_dim'] = len(label_keys)
    train_x, val_x, train_y, val_y = train_test_split(tweets,
                                                      labels,
                                                      test_size=0.1,
                                                      stratify=labels,
                                                      random_state=0)

    train_x, test_x, train_y, test_y = train_test_split(train_x,
                                                        train_y,
                                                        test_size=0.2,
                                                        stratify=train_y,
                                                        random_state=0)
    
    if res.mode == 'train':
        model = CatLSTMModel(params=params)
        model.train_model(train_x,train_y,val_x,val_y)
        results, preds = model.test_model(test_x,test_y)
        print(results)
    if res.mode == 'test':
        model = CatLSTMModel(params=params)
        results, preds = model.test_model(test_x,test_y)
        print(results)       
