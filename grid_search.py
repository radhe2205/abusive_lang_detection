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
from sklearn.model_selection import train_test_split

from preprocessing import Preprocessor
from src.models.attention_rnn import AttentionModel
from src.models.bi_rnn import RNNModel
from src.models.embedding import GloveEmbedding
from src.tweet_dataset import TweetDataset
from src.utils import change_path_to_absolute, save_model, load_model
from collections import Counter

def get_dataloader(tweets, labels, wordtoidx, batch_size):
    dataset = TweetDataset(tweets, labels, wordtoidx).cuda()
    return DataLoader(dataset = dataset, shuffle=False, batch_size = batch_size)

def get_all_words_from_train(tweets):
    pp = Preprocessor()
    vocab = pp.gen_vocab(sentences=tweets)
    word_to_idx = pp.gen_word_to_idx(vocab=vocab)
    return word_to_idx

def load_embeddings_n_words(tweets, embedding_path, embedding_type = "glove", embedding_dim = 50):
    wordtoidx = get_all_words_from_train(tweets)
    embedding = GloveEmbedding(embedding_dim, wordtoidx, embedding_path)
    wordtoidx = embedding.wordtoidx
    return embedding, wordtoidx

def save_vocab(vocab, vocab_path):
    with open(vocab_path, "w") as f:
        f.write(json.dumps(vocab))

def load_saved_vocab(vocab_path):
    with open(vocab_path, "r") as f:
        vocab = f.read()
        vocab = json.loads(vocab)
    return vocab

def save_model_results(model_params):
    with open(model_params['results_path'], "w") as f:
        f.write(json.dumps(model_params['results']))

def early_stop(f1_scores, latency):
    if len(f1_scores) < latency:
        return False
    window = f1_scores[-latency:]
    curr = window[-1]
    prevs = window[:-1]
    sma = np.mean(prevs)
    if curr < sma:
        return True
    return False

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x,y,w_len) in enumerate(dataloader):
        pred = model(x.cuda(), w_len.cuda()).cuda()
        pred = pred.squeeze()
        loss = loss_fn(pred,y.cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validation(dataloader, model, loss_fn, f1_scores):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    preds = torch.zeros(0).cuda()
    targets = torch.zeros(0).cuda()
    with torch.no_grad():
        for x,y,w_len in dataloader:
            pred = model(x.cuda(), w_len.cuda()).cuda()
            pred = pred.squeeze()
            test_loss += loss_fn(pred,y.cuda()).item()
            correct += (torch.round(pred)==y.cuda()).type(torch.float).sum().item()
            preds = torch.cat((preds,torch.round(pred)), dim=0)
            targets = torch.cat((targets,y.cuda()), dim=0)

    test_loss /= num_batches
    correct /= size

    results = classification_report(y_true=targets.cpu(), y_pred=preds.cpu(), output_dict=True, digits=4)
    return early_stop(f1_scores, latency=8), results["macro avg"]["f1-score"]

def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    preds = torch.zeros(0).cuda()
    targets = torch.zeros(0).cuda()
    with torch.no_grad():
        for x,y,w_len in dataloader:
            pred = model(x.cuda(), w_len.cuda()).cuda()
            pred = pred.squeeze()
            correct += (torch.round(pred)==y.cuda()).type(torch.float).sum().item()
            preds = torch.cat((preds,torch.round(pred)), dim=0)
            targets = torch.cat((targets,y.cuda()), dim=0)

    correct /= size

    results = classification_report(y_true=targets.cpu(), y_pred=preds.cpu(), output_dict=True, digits=4)
    return {'f1-score:':results["macro avg"]["f1-score"],
            'accuracy':correct}, preds

def train_model(params,train_x,train_y,val_x,val_y):
    embedding, wordtoidx = load_embeddings_n_words(tweets=train_x, 
                                                   embedding_path=params["embedding_path"], 
                                                   embedding_dim=params["embedding_dim"])
    save_vocab(wordtoidx, params['vocab_path'])

    model = params['model_class'](embeddings=embedding, 
                                  in_dim=params["embedding_dim"],
                                  num_layers=params['num_layers'],
                                  hidden_size=params['hidden_size'], 
                                  out_dim=1).cuda()

    train_loader = get_dataloader(tweets=train_x,
                                  labels=train_y, 
                                  wordtoidx=wordtoidx, 
                                  batch_size=params["batch_size"])

    val_loader = get_dataloader(tweets=val_x,
                                labels=val_y, 
                                wordtoidx=wordtoidx, 
                                batch_size=params["batch_size"])

    optimizer = Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.BCELoss(reduction='sum')
    sched = ExponentialLR(optimizer, gamma=0.95)
    
    f1_scores = []
    best_f1 = 0
    for t in range(train_options['epochs']):
        train(train_loader, model, loss_fn, optimizer)
        stop, f1_score = validation(val_loader, model, loss_fn, f1_scores)
        f1_scores.append(f1_score)

        if f1_score > best_f1:
            save_model(model, params['model_path'])
        
        if t > 15 and stop:
            break

        sched.step()    

def test_model(params,test_x,test_y):
    wordtoidx = load_saved_vocab(params['vocab_path'])
    embedding = GloveEmbedding(params["embedding_dim"], wordtoidx, params['embedding_path'], False)

    model = params['model_class'](embeddings=embedding, 
                                  in_dim=params["embedding_dim"],
                                  num_layers=params['num_layers'],
                                  hidden_size=params['hidden_size'], 
                                  out_dim=1).cuda()
    load_model(model, params['model_path'])

    test_loader = get_dataloader(tweets=test_x,
                                 labels=test_y, 
                                 wordtoidx=wordtoidx, 
                                 batch_size=params["batch_size"])

    results, preds = test(test_loader, model)
    return results, preds


def grid_search(param_map):
    best_params = {
        'f1-score':0,
        'params':{
            'num_layers':None,
            'hidden_size':None,
            'lr':None,
            'batch_size':None,
            'embedding_dim':None,
        }
    }

    pp = Preprocessor()
    OLID_train_tweets, OLID_train_labels = pp.get_train_data(train_options["train_data_path"],
                                                             sample=train_options['sample_size'],
                                                             seed=1)
    OLID_train_tweets, OLID_val_tweets, OLID_train_labels, OLID_val_labels = train_test_split(OLID_train_tweets,
                                                                                              OLID_train_labels,
                                                                                              test_size=0.1,
                                                                                              stratify=OLID_train_labels,
                                                                                              random_state=1)
    OLID_test_tweets, OLID_test_labels = pp.get_test_data(train_options['test_tweet_path'],
                                                          train_options['test_label_path'])
    for i,param in enumerate(param_map):
        print(f'{i}\t|\t{len(param_map)}')
        model_params = {
            'model_class':RNNModel,
            'num_layers':param['num_layers'],
            'hidden_size':param['hidden_size'],
            'lr':param['lr'],
            'batch_size':param['batch_size'],
            'embedding_dim':param['embedding_dim'],
            'embedding_path': f'data/glove822/glove.6B.{param["embedding_dim"]}d.txt',
            'vocab_path':f'{res.folder}/vocab.json',
            'model_path':f'{res.folder}/model.model',
            'results':None
        }
        train_model(params=model_params,
                    train_x=OLID_train_tweets,
                    train_y=OLID_train_labels,
                    val_x=OLID_val_tweets,
                    val_y=OLID_val_labels)    

        results, _ = test_model(params=model_params,
                                test_x=OLID_test_tweets,
                                test_y=OLID_test_labels)         

        if results['f1-score:'] > best_params['f1-score']:
            best_params['f1-score'] = results['f1-score:']
            best_params['params'] = {
                'num_layers':param['num_layers'],
                'hidden_size':param['hidden_size'],
                'lr':param['lr'],
                'batch_size':param['batch_size'],
                'embedding_dim':param['embedding_dim'],
            }
        
        # remove vocab and model save
        os.system(f'rm {res.folder}/*')

    
    return best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', action='store', dest='folder', default='saved_models', type=str)
    res = parser.parse_args()

    train_options = {
        "train_data_path": "data/OLIDv1.0/olid-training-v1.0_clean.tsv",
        "test_tweet_path": "data/OLIDv1.0/testset-levela_clean.tsv",
        "test_label_path": "data/OLIDv1.0/labels-levela.csv",
        "sub_task": "subtask_a",
        "sample_size":1,
        "epochs": 100,
        "model_type": "rnn"
    }   
   
    num_layers_grid = [1,2,3]
    hidden_size_grid = [32,64,128,256]
    lr_grid = [1e-2, 1e-3, 1e-4, 1e-5]
    batch_size_grid = [16, 32, 64, 128]
    embedding_dim_grid = [50, 100, 200, 300]

    param_map = []
    for nl in num_layers_grid:
        for hs in hidden_size_grid:
            for lr in lr_grid:
                for bs in batch_size_grid:
                    for em in embedding_dim_grid:
                        param_map.append({
                            'num_layers':nl,
                            'hidden_size':hs,
                            'lr':lr,
                            'batch_size':bs,
                            'embedding_dim':em,
                        })

    print(f'total parameter combinations: {len(param_map)}')
    best_params = grid_search(param_map=param_map)
    print(best_params)
    with open(f'{res.folder}/best_params.json', "w") as f:
        f.write(json.dumps(best_params))