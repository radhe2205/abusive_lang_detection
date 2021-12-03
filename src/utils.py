import copy
import os
import traceback
import json 

import torch

def change_path_to_absolute(train_options):
    train_options["embedding_path"] = train_options["embedding_path"].format(dims=train_options["embedding_dim"])
    if train_options["base_path"] in train_options["model_path"]:
        return train_options
    train_options = copy.deepcopy(train_options)
    train_options["model_path"] = train_options["base_path"] + train_options["model_path"]
    train_options["embedding_path"] = train_options["base_path"] + train_options["embedding_path"]
    train_options["vocab_path"] = train_options["base_path"] + train_options["vocab_path"]
    train_options["train_data_path"] = train_options["base_path"] + train_options["train_data_path"]
    train_options["test_tweet_path"] = train_options["base_path"] + train_options["test_tweet_path"]
    train_options["test_label_path"] = train_options["base_path"] + train_options["test_label_path"]
    return train_options

def save_model(model, model_path):
    directory_path = "/".join(model_path.split("/")[:-1])
    if len(directory_path) > 0:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    torch.save(model.state_dict(), model_path)

def get_scores(pred, target):
    tp = (target[pred == 1] == 1).sum()
    fp = (pred[target == 0] == 1).sum()
    fn = (pred[target == 1] == 0).sum()
    precision = tp / (fp + tp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1_score

def load_model(model, model_path):
    try:
        if not os.path.exists(model_path):
            return model

        model.load_state_dict(torch.load(model_path))
        return model
    except Exception as e:
        traceback.print_exc(e)
        print("Error occured while loading, ignoring...")

def format_tri_learning_results(folder):
    path = f'saved_models/{folder}'

    print('olid train')
    for i in range(3):
        results_path = f'{path}/model_{i+1}_results.json'
        with open(results_path, 'r') as f:
            results = f.read()
            results = json.loads(results)
        olid_f1_score = results['olid-train-olid-test']['f1-score:']
        olid_acc = results['olid-train-olid-test']['accuracy']
        solid_f1_score = results['olid-train-solid-test']['f1-score:']
        solid_acc = results['olid-train-solid-test']['accuracy']
        print(f'model {i+1}')
        print(f'olid f1: {round(olid_f1_score,4)} acc: {round(olid_acc,4)}')
        print(f'solid f1: {round(solid_f1_score,4)} acc: {round(solid_acc,4)}')
    
    with open(f'{path}/olid_train_ensemble_results.json', 'r') as f:
        ens_results = f.read()
        ens_results = json.loads(ens_results)
    olid_res = ens_results["olid-ensemble-test"]
    solid_res = ens_results["solid-ensemble-test"]
    print(f'ensemble')
    print(f'olid f1: {olid_res["f1-score:"]} acc: {olid_res["accuracy"]}')
    print(f'solid f1: {solid_res["f1-score:"]} acc: {solid_res["accuracy"]}')

    # ---------------------------------------------------------------------------#
    print('olid solid pred train')
    for i in range(3):
        results_path = f'{path}/model_{i+1}_results.json'
        with open(results_path, 'r') as f:
            results = f.read()
            results = json.loads(results)
        olid_f1_score = results['olid-solid-pred-train-olid-test']['f1-score:']
        olid_acc = results['olid-solid-pred-train-olid-test']['accuracy']
        solid_f1_score = results['olid-solid-pred-train-solid-test']['f1-score:']
        solid_acc = results['olid-solid-pred-train-solid-test']['accuracy']
        print(f'model {i+1}')
        print(f'olid f1: {round(olid_f1_score,4)} acc: {round(olid_acc,4)}')
        print(f'solid f1: {round(solid_f1_score,4)} acc: {round(solid_acc,4)}')
    
    with open(f'{path}/olid_solid_pred_train_ensemble_results.json', 'r') as f:
        ens_results = f.read()
        ens_results = json.loads(ens_results)
    olid_res = ens_results["olid-ensemble-test"]
    solid_res = ens_results["solid-ensemble-test"]
    print(f'ensemble')
    print(f'olid f1: {olid_res["f1-score:"]} acc: {olid_res["accuracy"]}')
    print(f'solid f1: {solid_res["f1-score:"]} acc: {solid_res["accuracy"]}')
     # ---------------------------------------------------------------------------#

    print('olid solid acc train')
    for i in range(3):
        results_path = f'{path}/model_{i+1}_results.json'
        with open(results_path, 'r') as f:
            results = f.read()
            results = json.loads(results)
        olid_f1_score = results['olid-solid-acc-train-olid-test']['f1-score:']
        olid_acc = results['olid-solid-acc-train-olid-test']['accuracy']
        solid_f1_score = results['olid-solid-acc-train-solid-test']['f1-score:']
        solid_acc = results['olid-solid-acc-train-solid-test']['accuracy']
        print(f'model {i+1}')
        print(f'olid f1: {round(olid_f1_score,4)} acc: {round(olid_acc,4)}')
        print(f'solid f1: {round(solid_f1_score,4)} acc: {round(solid_acc,4)}')
    
    with open(f'{path}/olid_solid_acc_train_ensemble_results.json', 'r') as f:
        ens_results = f.read()
        ens_results = json.loads(ens_results)
    olid_res = ens_results["olid-ensemble-test"]
    solid_res = ens_results["solid-ensemble-test"]
    print(f'ensemble')
    print(f'olid f1: {olid_res["f1-score:"]} acc: {olid_res["accuracy"]}')
    print(f'solid f1: {solid_res["f1-score:"]} acc: {solid_res["accuracy"]}')

if __name__ == "__main__":
    format_tri_learning_results(folder='simple_rnn_2')