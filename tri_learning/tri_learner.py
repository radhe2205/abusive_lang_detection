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
from sklearn.model_selection import train_test_split
from preprocessing import Preprocessor

from character_lstm import CharacterLSTM
from bi_lstm import BiLSTM
from logistic_regressor import LogisticRegressor

from src.utils import format_tri_learning_results

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_model_graphs(models):
    for i,model in enumerate(models):
        with open(model.params['graph_path'], "w") as f:
            f.write(json.dumps(model.params['graphs']))

def save_model_results(models):
    for i,model in enumerate(models):
        with open(model.params['results_path'], "w") as f:
            f.write(json.dumps(model.params['results']))

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

def test_ensemble(models,experiment):
    pp = Preprocessor()
    SOLID_test_tweets, SOLID_test_labels = load_solid_test(datapath='data/SOLID/test_a_tweets_clean.tsv',
                                                           labelpath='data/SOLID/test_a_labels.csv')
    OLID_test_tweets, OLID_test_labels = pp.get_test_data(train_options['test_tweet_path'],
                                                          train_options['test_label_path'])
    vote_map = {
        0:'NOT',
        1:'NOT',
        2:'OFF',
        3:'OFF'
    }
    solid_preds = []
    olid_preds = []
    for i,model in enumerate(models):
        _, solid_pred = model.test_model(experiment=experiment,
                                         test_x=SOLID_test_tweets,
                                         test_y=SOLID_test_labels)
        _, olid_pred = model.test_model(experiment=experiment,
                                        test_x=OLID_test_tweets,
                                        test_y=OLID_test_labels)
        solid_preds.append(solid_pred.squeeze())
        olid_preds.append(olid_pred.squeeze())

    solid_preds = torch.vstack(solid_preds)
    olid_preds = torch.vstack(olid_preds)
    solid_votes = torch.sum(solid_preds,axis=0)
    olid_votes = torch.sum(olid_preds,axis=0)
    
    solid_labels = []
    for s_vote in solid_votes:
        s_label = vote_map[s_vote.item()]
        solid_labels.append(s_label)
    
    olid_labels = []
    for o_vote in olid_votes:
        o_label = vote_map[o_vote.item()]
        olid_labels.append(o_label)
    
    solid_results = classification_report(y_true=SOLID_test_labels, y_pred=solid_labels, output_dict=True, digits=4)
    solid_correct = np.mean([y_true==y_pred for y_true,y_pred in zip(SOLID_test_labels,solid_labels)])
    olid_results = classification_report(y_true=OLID_test_labels, y_pred=olid_labels, output_dict=True, digits=4)
    olid_correct = np.mean([y_true==y_pred for y_true,y_pred in zip(OLID_test_labels,olid_labels)])
    
    return {'f1-score:':solid_results["macro avg"]["f1-score"],
            'accuracy':solid_correct},{'f1-score:':olid_results["macro avg"]["f1-score"],
            'accuracy':olid_correct}

if __name__ == "__main__":
    train_options = {
        "train_data_path": "data/OLIDv1.0/olid-training-v1.0_clean.tsv",
        "test_tweet_path": "data/OLIDv1.0/testset-levelb_clean.tsv",
        "test_label_path": "data/OLIDv1.0/labels-levelb.csv",
        "sample_size":1,
        "seed":1
    }   
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', action='store', dest='folder', default='saved_models', type=str)
    parser.add_argument('--mode', action='store', dest='mode', default='test', type=str)
    parser.add_argument('--models', action='store', dest='models', default='grid', type=str)
    res = parser.parse_args()

    SOLID_train_tweets, SOLID_train_labels = load_solid_train(datapath='data/SOLID/task_a_distant_tweets_clean.tsv')
    SOLID_test_tweets, SOLID_test_labels = load_solid_test(datapath='data/SOLID/test_a_tweets_clean.tsv',
                                                           labelpath='data/SOLID/test_a_labels.csv')

    bi_lstm_params = {
        'embedding_path':'data/glove822/glove.6B.300d.txt',
        'embedding_dim':300,
        'num_layers':2,
        'hidden_size':32,
        'batch_size':32,
        'lr':0.0001,
        'epochs':100,
        'task':'a',
        'model_path':{
            'olid-train':f'{res.folder}/model_1_olid.model',
            'olid-solid-pred-train':f'{res.folder}/model_1_solid_pred.model',
            'olid-solid-acc-train':f'{res.folder}/model_1_solid_acc.model'
        },
        'vocab_path':{
            'olid-train':f'{res.folder}/vocab_1_olid.json',
            'olid-solid-pred-train':f'{res.folder}/vocab_1_solid_pred.json',
            'olid-solid-acc-train':f'{res.folder}/vocab_1_solid_acc.json'
        },
        'results':{
            'olid-train-olid-test':None,
            'olid-train-solid-test':None,
            'olid-solid-pred-train-olid-test':None,
            'olid-solid-pred-train-solid-test':None,
            'olid-solid-acc-train-olid-test':None,
            'olid-solid-acc-train-solid-test':None
        },
        'graphs':{
            'olid-train':{},
            'olid-solid-pred-train':{},
            'olid-solid-acc-train':{}
        },
        'graph_path':f'{res.folder}/model_1_graphs.json',
        'results_path':f'{res.folder}/model_1_results.json'
    }
    bi_lstm = BiLSTM(params=bi_lstm_params)
    
    bi_lstm_params_2 = {
        'embedding_path':'data/glove822/glove.6B.300d.txt',
        'embedding_dim':300,
        'num_layers':2,
        'hidden_size':32,
        'batch_size':32,
        'lr':0.0001,
        'epochs':100,
        'task':'a',
        'model_path':{
            'olid-train':f'{res.folder}/model_2_olid.model',
            'olid-solid-pred-train':f'{res.folder}/model_2_solid_pred.model',
            'olid-solid-acc-train':f'{res.folder}/model_2_solid_acc.model'
        },
        'vocab_path':{
            'olid-train':f'{res.folder}/vocab_2_olid.json',
            'olid-solid-pred-train':f'{res.folder}/vocab_2_solid_pred.json',
            'olid-solid-acc-train':f'{res.folder}/vocab_2_solid_acc.json'
        },
        'results':{
            'olid-train-olid-test':None,
            'olid-train-solid-test':None,
            'olid-solid-pred-train-olid-test':None,
            'olid-solid-pred-train-solid-test':None,
            'olid-solid-acc-train-olid-test':None,
            'olid-solid-acc-train-solid-test':None
        },
        'graphs':{
            'olid-train':{},
            'olid-solid-pred-train':{},
            'olid-solid-acc-train':{}
        },
        'graph_path':f'{res.folder}/model_2_graphs.json',
        'results_path':f'{res.folder}/model_2_results.json'
    }
    bi_lstm_2 = BiLSTM(params=bi_lstm_params_2)

    bi_lstm_params_3 = {
        'embedding_path':'data/glove822/glove.6B.300d.txt',
        'embedding_dim':300,
        'num_layers':2,
        'hidden_size':32,
        'batch_size':32,
        'lr':0.0001,
        'epochs':100,
        'task':'a',
        'model_path':{
            'olid-train':f'{res.folder}/model_3_olid.model',
            'olid-solid-pred-train':f'{res.folder}/model_3_solid_pred.model',
            'olid-solid-acc-train':f'{res.folder}/model_3_solid_acc.model'
        },
        'vocab_path':{
            'olid-train':f'{res.folder}/vocab_3_olid.json',
            'olid-solid-pred-train':f'{res.folder}/vocab_3_solid_pred.json',
            'olid-solid-acc-train':f'{res.folder}/vocab_3_solid_acc.json'
        },
        'results':{
            'olid-train-olid-test':None,
            'olid-train-solid-test':None,
            'olid-solid-pred-train-olid-test':None,
            'olid-solid-pred-train-solid-test':None,
            'olid-solid-acc-train-olid-test':None,
            'olid-solid-acc-train-solid-test':None
        },
        'graphs':{
            'olid-train':{},
            'olid-solid-pred-train':{},
            'olid-solid-acc-train':{}
        },
        'graph_path':f'{res.folder}/model_3_graphs.json',
        'results_path':f'{res.folder}/model_3_results.json'
    }
    bi_lstm_3 = BiLSTM(params=bi_lstm_params_3)

    char_lstm_params = {
        'num_layers':2,
        'hidden_size':128,
        'batch_size':32,
        'lr':0.001,
        'epochs':100,
        'task':'a',
        'model_path':{
            'olid-train':f'{res.folder}/model_2_olid.model',
            'olid-solid-pred-train':f'{res.folder}/model_2_solid_pred.model',
            'olid-solid-acc-train':f'{res.folder}/model_2_solid_acc.model'
        },
        'vocab_path':{
            'olid-train':f'{res.folder}/vocab_2_olid.json',
            'olid-solid-pred-train':f'{res.folder}/vocab_2_solid_pred.json',
            'olid-solid-acc-train':f'{res.folder}/vocab_2_solid_acc.json'
        },
        'max_len_path':{
            'olid-train':f'{res.folder}/max_2_olid.json',
            'olid-solid-pred-train':f'{res.folder}/max_2_solid_pred.json',
            'olid-solid-acc-train':f'{res.folder}/max_2_solid_acc.json'
        },
        'results':{
            'olid-train-olid-test':None,
            'olid-train-solid-test':None,
            'olid-solid-pred-train-olid-test':None,
            'olid-solid-pred-train-solid-test':None,
            'olid-solid-acc-train-olid-test':None,
            'olid-solid-acc-train-solid-test':None
        },
        'graphs':{
            'olid-train':{},
            'olid-solid-pred-train':{},
            'olid-solid-acc-train':{}
        },
        'graph_path':f'{res.folder}/model_2_graphs.json',
        'results_path':f'{res.folder}/model_2_results.json'
    }
    char_lstm = CharacterLSTM(params=char_lstm_params)

    logit_reg_params = {
        'batch_size':32,
        'lr':0.01,
        'epochs':100,
        'task':'a',
        'model_path':{
            'olid-train':f'{res.folder}/model_3_olid.model',
            'olid-solid-pred-train':f'{res.folder}/model_3_solid_pred.model',
            'olid-solid-acc-train':f'{res.folder}/model_3_solid_acc.model'
        },
        'vectorizer_path':{
            'olid-train':f'{res.folder}/vec_3_olid.json',
            'olid-solid-pred-train':f'{res.folder}/vec_3_solid_pred.json',
            'olid-solid-acc-train':f'{res.folder}/vec_3_solid_acc.json'
        },
        'results':{
            'olid-train-olid-test':None,
            'olid-train-solid-test':None,
            'olid-solid-pred-train-olid-test':None,
            'olid-solid-pred-train-solid-test':None,
            'olid-solid-acc-train-olid-test':None,
            'olid-solid-acc-train-solid-test':None
        },
        'graphs':{
            'olid-train':{},
            'olid-solid-pred-train':{},
            'olid-solid-acc-train':{}
        },
        'graph_path':f'{res.folder}/model_3_graphs.json',
        'results_path':f'{res.folder}/model_3_results.json'        
    }
    logit_reg = LogisticRegressor(params=logit_reg_params)

    ensemble_models = {
        'olid-train':{
            'results':{
                'olid-ensemble-test':None,
                'solid-ensemble-test':None,
            },
            'results_path':f'{res.folder}/olid_train_ensemble_results.json'
        },
        'olid-solid-pred-train':{
            'results':{
                'olid-ensemble-test':None,
                'solid-ensemble-test':None,
            },
            'results_path':f'{res.folder}/olid_solid_pred_train_ensemble_results.json'
        },
        'olid-solid-acc-train':{
            'results':{
                'olid-ensemble-test':None,
                'solid-ensemble-test':None,
            },
            'results_path':f'{res.folder}/olid_solid_acc_train_ensemble_results.json'
        },
    }

    if res.models == 'grid':
        models = [bi_lstm, bi_lstm_2, bi_lstm_3]
    if res.models == 'diverse':
        models = [bi_lstm, char_lstm, logit_reg]


    if res.mode == 'train':
        # train and test all the models on OLID and SOLID test sets
        print('*'*40)
        print('experiment 1: train on OLID and test on OLID and SOLID test sets')
        for i,model in enumerate(models):
            print('='*40)
            print(f'model {i+1}')
            pp = Preprocessor()
            OLID_train_tweets, OLID_train_labels = pp.get_train_data(train_options["train_data_path"], 
                                                                    sample=train_options['sample_size'],
                                                                    seed=train_options['seed'])
            OLID_train_tweets, OLID_val_tweets, OLID_train_labels, OLID_val_labels = train_test_split(OLID_train_tweets,
                                                                                                    OLID_train_labels,
                                                                                                    test_size=0.1,
                                                                                                    stratify=OLID_train_labels,
                                                                                                    random_state=1)

            OLID_test_tweets, OLID_test_labels = pp.get_test_data(train_options['test_tweet_path'],
                                                                train_options['test_label_path'])
            model.train_model(experiment='olid-train',
                            train_x=OLID_train_tweets,
                            train_y=OLID_train_labels,
                            val_x=OLID_val_tweets,
                            val_y=OLID_val_labels)

            results, _ = model.test_model(experiment='olid-train',
                                        test_x=OLID_test_tweets,
                                        test_y=OLID_test_labels)

            model.params['results']['olid-train-olid-test'] = results
            model.params['graphs']['olid-train']['train_loss'] = model.train_loss
            model.params['graphs']['olid-train']['val_loss'] = model.val_loss
            model.params['graphs']['olid-train']['train_acc'] = model.train_acc
            model.params['graphs']['olid-train']['val_acc'] = model.val_acc
            model.reset_losses()

            print('='*40)
            print()
        
        save_model_graphs(models)
        save_model_results(models)
        
        # test each model on SOLID
        for i,model in enumerate(models):
            results, _ = model.test_model(experiment='olid-train',
                                        test_x=SOLID_test_tweets,
                                        test_y=SOLID_test_labels)
            model.params['results']['olid-train-solid-test'] = results
        
        save_model_results(models)
        
        print('*'*40)
        print()

        print('*'*40)
        print('experiment 2: test on train data add majority vote to minority vote models training set and retrain')
        print('then test on both OLID and SOLID test sets')

        # read in N SOLID samples from the data set (stratified)
        N = 30000
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
        for i,model in enumerate(models):
            _, pred = model.test_model(experiment='olid-train',
                                    test_x=solid_train_x,
                                    test_y=solid_train_y)
            preds.append(pred.squeeze())

        preds = torch.vstack(preds)
        votes = torch.sum(preds,axis=0)
        labels = []
        for vote in votes:
            label = vote_map[vote.item()]
            labels.append(label)

        # train each classifier using new data and test on OLID
        for i,model in enumerate(models):
            print('='*40)
            print(f'model {i+1}')
            pp = Preprocessor()
            OLID_train_tweets, OLID_train_labels = pp.get_train_data(train_options["train_data_path"], 
                                                                    sample=train_options['sample_size'],
                                                                    seed=train_options['seed'])
            OLID_train_tweets, OLID_val_tweets, OLID_train_labels, OLID_val_labels = train_test_split(OLID_train_tweets,
                                                                                                    OLID_train_labels,
                                                                                                    test_size=0.1,
                                                                                                    stratify=OLID_train_labels,
                                                                                                    random_state=1)      
            
            OLID_test_tweets, OLID_test_labels = pp.get_test_data(train_options['test_tweet_path'],
                                                                train_options['test_label_path'])
            
            full_train_tweets = np.concatenate([OLID_train_tweets,solid_train_x])
            full_train_labels = np.concatenate([OLID_train_labels,labels])

            model.train_model(experiment='olid-solid-pred-train',
                            train_x=full_train_tweets,
                            train_y=full_train_labels,
                            val_x=OLID_val_tweets,
                            val_y=OLID_val_labels)

            results, _ = model.test_model(experiment='olid-solid-pred-train',
                                        test_x=OLID_test_tweets,
                                        test_y=OLID_test_labels)
            model.params['results']['olid-solid-pred-train-olid-test'] = results
            model.params['graphs']['olid-solid-pred-train']['train_loss'] = model.train_loss
            model.params['graphs']['olid-solid-pred-train']['val_loss'] = model.val_loss
            model.params['graphs']['olid-solid-pred-train']['train_acc'] = model.train_acc
            model.params['graphs']['olid-solid-pred-train']['val_acc'] = model.val_acc
            model.reset_losses()
            print('='*40)
            print()        
        
        save_model_graphs(models)
        save_model_results(models)
        
        # test SOLID test sets
        for i,model in enumerate(models):
            results, _ = model.test_model(experiment='olid-solid-pred-train',
                                        test_x=SOLID_test_tweets,
                                        test_y=SOLID_test_labels)
            model.params['results']['olid-solid-pred-train-solid-test'] = results
        
        save_model_results(models)
        
        print('*'*40)
        print()

        print('*'*40)
        print('experiment 3: add same number of new training points but use actual labels and retrain')
        print('then test on both OLID and SOLID test sets')
        
        # train each classifier using new data and test on OLID
        for i,model in enumerate(models):
            print('='*40)
            print(f'model {i+1}')
            pp = Preprocessor()
            OLID_train_tweets, OLID_train_labels = pp.get_train_data(train_options["train_data_path"], 
                                                                    sample=train_options['sample_size'],
                                                                    seed=train_options['seed'])
            OLID_train_tweets, OLID_val_tweets, OLID_train_labels, OLID_val_labels = train_test_split(OLID_train_tweets,
                                                                                                    OLID_train_labels,
                                                                                                    test_size=0.1,
                                                                                                    stratify=OLID_train_labels,
                                                                                                    random_state=1)     
            
            OLID_test_tweets, OLID_test_labels = pp.get_test_data(train_options['test_tweet_path'],
                                                                train_options['test_label_path'])
            
            full_train_tweets = np.concatenate([OLID_train_tweets,solid_train_x])
            full_train_labels = np.concatenate([OLID_train_labels,solid_train_y])
            
            model.train_model(experiment='olid-solid-acc-train',
                            train_x=full_train_tweets,
                            train_y=full_train_labels,
                            val_x=OLID_val_tweets,
                            val_y=OLID_val_labels)

            results, _ = model.test_model(experiment='olid-solid-acc-train',
                                        test_x=OLID_test_tweets,
                                        test_y=OLID_test_labels)
            model.params['results']['olid-solid-acc-train-olid-test'] = results
            model.params['graphs']['olid-solid-acc-train']['train_loss'] = model.train_loss
            model.params['graphs']['olid-solid-acc-train']['val_loss'] = model.val_loss
            model.params['graphs']['olid-solid-acc-train']['train_acc'] = model.train_acc
            model.params['graphs']['olid-solid-acc-train']['val_acc'] = model.val_acc
            model.reset_losses()
            print('='*40)
            print()        
            
        save_model_graphs(models)
        save_model_results(models)

        # test SOLID test sets
        for i,model in enumerate(models):
            results, _ = model.test_model(experiment='olid-solid-acc-train',
                                        test_x=SOLID_test_tweets,
                                        test_y=SOLID_test_labels)
            model.params['results']['olid-solid-acc-train-solid-test'] = results
        
        save_model_results(models)

        print('*'*40)
        print()   

        print('*'*40)
        print('experiment 4: use each of the three trained models to test as an ensemble on both OLID and SOLID test sets')

        # test on OLID and SOLID test sets
        experiments = ['olid-train','olid-solid-pred-train','olid-solid-acc-train']
        for experiment in experiments:
            solid_results,olid_results = test_ensemble(models=models, experiment=experiment)
            ensemble_models[experiment]['results']['olid-ensemble-test'] = olid_results
            ensemble_models[experiment]['results']['solid-ensemble-test'] = solid_results

            with open(ensemble_models[experiment]['results_path'], "w") as f:
                f.write(json.dumps(ensemble_models[experiment]['results']))

        print('*'*40)
        print()

        print('program finished')
        print('saving results')
    elif res.model == 'test':
        format_tri_learning_results(res.folder)