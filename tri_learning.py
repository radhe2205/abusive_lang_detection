import json
import os
import sys
sys.path.append(os.getcwd())

import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from preprocessing import Preprocessor
from src.models.attention_rnn import AttentionModel
from src.models.bi_rnn import RNNModel
from src.models.embedding import GloveEmbedding
from src.tweet_dataset import TweetDataset
from src.utils import change_path_to_absolute, save_model, load_model
from collections import Counter


def get_train_dataloader(tweets, labels, wordtoidx, batch_size, task = "subtask_a"):
    dataset = TweetDataset(tweets, labels, wordtoidx).cuda()
    return DataLoader(dataset = dataset, shuffle=True, batch_size = batch_size)

def get_test_dataloader(tweet_path, label_path, wordtoidx, batch_size):
    pp = Preprocessor()
    tweets, labels = pp.get_test_data(tweet_path, label_path)
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

def train_test_model(params,experiment):
    pp = Preprocessor()
    tweets, labels = pp.get_train_data(train_options["train_data_path"], 
                                       sample=0.77,
                                       seed=params['seed'])
    embedding, wordtoidx = load_embeddings_n_words(tweets, 
                                                   train_options["embedding_path"], 
                                                   embedding_dim=train_options["embedding_dim"])
    
    model = params['model_class'](embeddings=embedding, 
                                  in_dim=train_options["embedding_dim"],
                                  num_layers=params['num_layers'],
                                  hidden_size=params['hidden_size'], 
                                  out_dim=1).cuda()

    train_loader = get_train_dataloader(tweets,labels, 
                                        wordtoidx, 
                                        train_options["batch_size"], 
                                        train_options["sub_task"])
    test_loader = get_test_dataloader(train_options["test_tweet_path"],
                                      train_options["test_label_path"], 
                                      wordtoidx, 
                                      train_options["batch_size"])
    
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
    
    save_model(model, params['path'][experiment])
    return results
    
if __name__ == "__main__":
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

    
    # train and test all the models
    models = [
        {   
            'model_class':RNNModel,
            'num_layers':1,
            'hidden_size':128,
            'path':{
                'olid':'saved_models/model_1_olid.model',
                'solid-pred':'saved_models/model_1_solid_pred.model',
                'solid-acc':'saved_models/model_1_solid_acc.model'
            },
            'seed':1,
            'results':{
                'olid':None,
                'solid-pred':None,
                'solid-acc':None
            }
        },
        {
            'model_class':RNNModel,
            'num_layers':1,
            'hidden_size':128,
            'path':{
                'olid':'saved_models/model_2_olid.model',
                'solid-pred':'saved_models/model_2_solid_pred.model',
                'solid-acc':'saved_models/model_2_solid_acc.model'
            },
            'seed':2,
            'results':{
                'olid':None,
                'solid-pred':None,
                'solid-acc':None
            }
        },
        {
            'model_class':RNNModel,
            'num_layers':1,
            'hidden_size':128,
            'path':{
                'olid':'saved_models/model_3_olid.model',
                'solid-pred':'saved_models/model_3_solid_pred.model',
                'solid-acc':'saved_models/model_3_solid_acc.model'
            },
            'seed':3,
            'results':{
                'olid':None,
                'solid-pred':None,
                'solid-acc':None
            }
        }
    ]

    for i,model_params in enumerate(models):
        print('='*40)
        print(f'model {i+1}')
        model_params['results']['olid'] = train_test_model(params=model_params,experiment='olid')
        print('='*40)
        print()

    print('*'*40)
    print()
    print(models)

    # test each model on SOLID

    # read in random sample of n SOLID tweets and drop labels

    # experiment A
    # add flagged missclasifications to data each cloned data set
    
    # experiment B
    # add same number of samples but use the training label

    # re train all models using newly added points (both experiments)

    # test on OLID data set again (individually and as ensemble) (both experiments)

    # test on SOLID data set again (individually and as ensemble) (both experiments)
