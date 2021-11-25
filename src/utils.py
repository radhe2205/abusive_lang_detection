import copy
import os
import traceback

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
