import json
import os
import sys
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

def load_solid_train(datapath):
    data = pd.read_csv(datapath, sep='\t')
    data = data.sample(frac=1).reset_index(drop=True)
    data = data[data['tweet'].notna()]
    return data['tweet'].values, data['label'].values

def load_solid_test(datapath, labelpath):
    data = pd.read_csv(datapath, sep='\t')
    labels = pd.read_csv(labelpath, names=['id','label'])
    is_nan = data.isnull()
    row_has_nan = is_nan.any(axis=1)
    labels = labels[~row_has_nan]
    data = data.dropna()
    return data['tweet'].values, labels['label'].values

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

def save_model_results(models):
    for i,model_params in enumerate(models):
        with open(model_params['results_path'], "w") as f:
            f.write(json.dumps(model_params['results']))

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

def validation(dataloader, model, loss_fn):
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
    print('-'*40)
    print('testing')
    print(f'precision: \t{results["macro avg"]["precision"]}')
    print(f'recall: \t{results["macro avg"]["recall"]}')
    print(f'F1-score: \t{results["macro avg"]["f1-score"]}')
    print(f'validation error: \naccuracy {correct:>5f}, avg loss: {test_loss:>7f}')
    print('-'*40)
    return {'f1-score:':results["macro avg"]["f1-score"],
            'accuracy':correct}

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

def train_model(params,experiment,train_x,train_y,test_x,test_y):
    embedding, wordtoidx = load_embeddings_n_words(tweets=train_x, 
                                                   embedding_path=train_options["embedding_path"], 
                                                   embedding_dim=train_options["embedding_dim"])
    save_vocab(wordtoidx, params['vocab_path'][experiment])

    model = params['model_class'](embeddings=embedding, 
                                  in_dim=train_options["embedding_dim"],
                                  num_layers=params['num_layers'],
                                  hidden_size=params['hidden_size'], 
                                  out_dim=1).cuda()

    train_loader = get_dataloader(tweets=train_x,
                                  labels=train_y, 
                                  wordtoidx=wordtoidx, 
                                  batch_size=train_options["batch_size"])

    test_loader = get_dataloader(tweets=test_x,
                                 labels=test_y, 
                                 wordtoidx=wordtoidx, 
                                 batch_size=train_options["batch_size"])

    optimizer = Adam(model.parameters(), lr=train_options["lr"])
    loss_fn = nn.BCELoss(reduction='sum')
    sched = ExponentialLR(optimizer, gamma=0.99)
    
    for t in range(train_options['epochs']):
        print(f'epoch {t}')
        print('-'*40)
        train(train_loader, model, loss_fn, optimizer)
        results = validation(test_loader, model, loss_fn)
        sched.step()    
        print()
    
    save_model(model, params['model_path'][experiment])

def test_model(params,experiment,test_x,test_y):
    wordtoidx = load_saved_vocab(params['vocab_path'][experiment])
    embedding = GloveEmbedding(train_options["embedding_dim"], wordtoidx, train_options["embedding_dim"], False)

    model = params['model_class'](embeddings=embedding, 
                                  in_dim=train_options["embedding_dim"],
                                  num_layers=params['num_layers'],
                                  hidden_size=params['hidden_size'], 
                                  out_dim=1).cuda()
    load_model(model, params['model_path'][experiment])

    test_loader = get_dataloader(tweets=test_x,
                                 labels=test_y, 
                                 wordtoidx=wordtoidx, 
                                 batch_size=train_options["batch_size"])

    results, preds = test(test_loader, model)
    return results, preds
    
if __name__ == "__main__":
    SOLID_train_tweets, SOLID_train_labels = load_solid_train(datapath='data/SOLID/task_a_distant_tweets_clean.tsv')
    SOLID_test_tweets, SOLID_test_labels = load_solid_test(datapath='data/SOLID/test_a_tweets_clean.tsv',
                                                           labelpath='data/SOLID/test_a_labels.csv')

    train_options = {
        "embedding_dim": 300,
        "embedding_path": "data/glove822/glove.6B.300d.txt",
        "train_data_path": "data/OLIDv1.0/olid-training-v1.0_clean.tsv",
        "test_tweet_path": "data/OLIDv1.0/testset-levela_clean.tsv",
        "test_label_path": "data/OLIDv1.0/labels-levela.csv",
        "sub_task": "subtask_a",
        "batch_size": 32,
        "lr": 0.001,
        "epochs": 100,
        "model_type": "rnn" # attention | rnn
    }   

    models = [
        {   
            'model_class':RNNModel,
            'num_layers':1,
            'hidden_size':128,
            'model_path':{
                'olid-train':'saved_models/model_1_olid.model',
                'olid-solid-pred-train':'saved_models/model_1_solid_pred.model',
                'olid-solid-acc-train':'saved_models/model_1_solid_acc.model'
            },
            'vocab_path':{
                'olid-train':'saved_models/vocab_1_olid.json',
                'olid-solid-pred-train':'saved_models/vocab_1_solid_pred.json',
                'olid-solid-acc-train':'saved_models/vocab_1_solid_acc.json'
            },
            'seed':1,
            'results':{
                'olid-train-olid-test':None,
                'olid-train-solid-test':None,
                'olid-solid-pred-train-olid-test':None,
                'olid-solid-pred-train-solid-test':None,
                'olid-solid-acc-train-olid-test':None,
                'olid-solid-acc-train-solid-test':None
            },
            'results_path':'saved_models/model_1_results.json'
        },
        {
            'model_class':RNNModel,
            'num_layers':1,
            'hidden_size':128,
            'model_path':{
                'olid-train':'saved_models/model_2_olid.model',
                'olid-solid-pred-train':'saved_models/model_2_solid_pred.model',
                'olid-solid-acc-train':'saved_models/model_2_solid_acc.model'
            },
            'vocab_path':{
                'olid-train':'saved_models/vocab_2_olid.json',
                'olid-solid-pred-train':'saved_models/vocab_2_solid_pred.json',
                'olid-solid-acc-train':'saved_models/vocab_2_solid_acc.json'
            },
            'seed':2,
            'results':{
                'olid-train-olid-test':None,
                'olid-train-solid-test':None,
                'olid-solid-pred-train-olid-test':None,
                'olid-solid-pred-train-solid-test':None,
                'olid-solid-acc-train-olid-test':None,
                'olid-solid-acc-train-solid-test':None
            },
            'results_path':'saved_models/model_2_results.json'
        },
        {
            'model_class':RNNModel,
            'num_layers':1,
            'hidden_size':128,
            'model_path':{
                'olid-train':'saved_models/model_3_olid.model',
                'olid-solid-pred-train':'saved_models/model_3_solid_pred.model',
                'olid-solid-acc-train':'saved_models/model_3_solid_acc.model'
            },
            'vocab_path':{
                'olid-train':'saved_models/vocab_3_olid.json',
                'olid-solid-pred-train':'saved_models/vocab_3_solid_pred.json',
                'olid-solid-acc-train':'saved_models/vocab_3_solid_acc.json'
            },
            'seed':3,
            'results':{
                'olid-train-olid-test':None,
                'olid-train-solid-test':None,
                'olid-solid-pred-train-olid-test':None,
                'olid-solid-pred-train-solid-test':None,
                'olid-solid-acc-train-olid-test':None,
                'olid-solid-acc-train-solid-test':None
            },
            'results_path':'saved_models/model_3_results.json'
        }
    ]

    # train and test all the models on OLID and SOLID test sets
    print('*'*40)
    print('experiment 1: train on OLID and test on OLID and SOLID test sets')
    for i,model_params in enumerate(models):
        print('='*40)
        print(f'model {i+1}')
        pp = Preprocessor()
        OLID_train_tweets, OLID_train_labels = pp.get_train_data(train_options["train_data_path"], 
                                                                 sample=0.77,
                                                                 seed=model_params['seed'])
        OLID_test_tweets, OLID_test_labels = pp.get_test_data(train_options['test_tweet_path'],
                                                              train_options['test_label_path'])
        train_model(params=model_params,
                    experiment='olid-train',
                    train_x=OLID_train_tweets,
                    train_y=OLID_train_labels,
                    test_x=OLID_test_tweets,
                    test_y=OLID_test_labels)

        results, _ = test_model(params=model_params,
                                experiment='olid-train',
                                test_x=OLID_test_tweets,
                                test_y=OLID_test_labels)
        model_params['results']['olid-train-olid-test'] = results
        print('='*40)
        print()
    
    save_model_results(models)
    
    # test each model on SOLID
    for i,model_params in enumerate(models):
        results, _ = test_model(params=model_params,
                                experiment='olid-train',
                                test_x=SOLID_test_tweets,
                                test_y=SOLID_test_labels)
        model_params['results']['olid-train-solid-test'] = results
    
    save_model_results(models)
    
    print('*'*40)
    print()

    print('*'*40)
    print('experiment 3: test on train data add majority vote to minority vote models training set and retrain')
    print('then test on both OLID and SOLID test sets')

    # read in N SOLID samples from the data set (stratified)
    N = 15000
    SOLID_train_tweets, SOLID_train_labels = load_solid_train(datapath='data/SOLID/task_a_distant_tweets_clean.tsv')
    solid_train_x, _, solid_train_y, _ = train_test_split(SOLID_train_tweets,SOLID_train_labels,
                                                          train_size=N,stratify=SOLID_train_labels,
                                                          random_state=0)
    
    # using this new data, get prediction for each point and assign majority vote to each training set
    vote_map = {
        0:'NOT',
        1:'NOT',
        2:'OFF',
        3:'OFF'
    }
    preds = []
    for i,model_params in enumerate(models):
        _, pred = test_model(params=model_params,
                             experiment='olid-train',
                             test_x=solid_train_x,
                             test_y=solid_train_y)
        preds.append(pred)
    preds = torch.vstack(preds)
    votes = torch.sum(preds,axis=0)
    labels = []
    for vote in votes:
        label = vote_map[vote.item()]
        labels.append(label)

    # train each classifier using new data and test on OLID
    for i,model_params in enumerate(models):
        print('='*40)
        print(f'model {i+1}')
        pp = Preprocessor()
        OLID_train_tweets, OLID_train_labels = pp.get_train_data(train_options["train_data_path"], 
                                                                 sample=0.77,
                                                                 seed=model_params['seed'])
        OLID_test_tweets, OLID_test_labels = pp.get_test_data(train_options['test_tweet_path'],
                                                              train_options['test_label_path'])
        
        full_train_tweets = np.concatenate([OLID_train_tweets,solid_train_x])
        full_train_labels = np.concatenate([OLID_train_labels,labels])

        train_model(params=model_params,
                    experiment='olid-solid-pred-train',
                    train_x=full_train_tweets,
                    train_y=full_train_labels,
                    test_x=OLID_test_tweets,
                    test_y=OLID_test_labels)

        results, _ = test_model(params=model_params,
                                experiment='olid-solid-pred-train',
                                test_x=OLID_test_tweets,
                                test_y=OLID_test_labels)
        model_params['results']['olid-solid-pred-train-olid-test'] = results
        print('='*40)
        print()        
    
    save_model_results(models)
    
    # test SOLID test sets
    for i,model_params in enumerate(models):
        results, _ = test_model(params=model_params,
                                experiment='olid-solid-pred-train',
                                test_x=SOLID_test_tweets,
                                test_y=SOLID_test_labels)
        model_params['results']['olid-solid-pred-train-solid-test'] = results
    
    save_model_results(models)
    
    print('*'*40)
    print()

    print('*'*40)
    print('experiment 4: add same number of new training points but use actual labels and retrain')
    print('then test on both OLID and SOLID test sets')
    
    # train each classifier using new data and test on OLID
    for i,model_params in enumerate(models):
        print('='*40)
        print(f'model {i+1}')
        pp = Preprocessor()
        OLID_train_tweets, OLID_train_labels = pp.get_train_data(train_options["train_data_path"], 
                                                                 sample=0.77,
                                                                 seed=model_params['seed'])
        OLID_test_tweets, OLID_test_labels = pp.get_test_data(train_options['test_tweet_path'],
                                                              train_options['test_label_path'])
        
        full_train_tweets = np.concatenate([OLID_train_tweets,solid_train_x])
        full_train_labels = np.concatenate([OLID_train_labels,solid_train_y])

        train_model(params=model_params,
                    experiment='olid-solid-acc-train',
                    train_x=full_train_tweets,
                    train_y=full_train_labels,
                    test_x=OLID_test_tweets,
                    test_y=OLID_test_labels)

        results, _ = test_model(params=model_params,
                                experiment='olid-solid-acc-train',
                                test_x=OLID_test_tweets,
                                test_y=OLID_test_labels)
        model_params['results']['olid-solid-acc-train-olid-test'] = results
        print('='*40)
        print()        
    
    save_model_results(models)

    # test SOLID test sets
    for i,model_params in enumerate(models):
        results, _ = test_model(params=model_params,
                                experiment='olid-solid-acc-train',
                                test_x=SOLID_test_tweets,
                                test_y=SOLID_test_labels)
        model_params['results']['olid-solid-acc-train-solid-test'] = results
    
    print('*'*40)
    print()   

    print('program finished')
    print('saving results')
    save_model_results(models)
