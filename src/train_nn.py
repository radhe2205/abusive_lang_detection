import copy
import json

import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import Preprocessor
from src.models.attention_exp import AttentionExp
from src.models.attention_rnn import AttentionModel
from src.models.bi_rnn import RNNModel
from src.models.embedding import GloveEmbedding
from src.tweet_dataset import TweetDataset
from src.utils import change_path_to_absolute, save_model, load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_train_val_dataloader(dataset_path, wordtoidx, batch_size, task ="subtask_a"):
    pp = Preprocessor()
    tweets, labels = pp.get_train_data(dataset_path, task)
    from sklearn.model_selection import train_test_split
    tweets_train, tweets_val, label_train, label_val = train_test_split(tweets, labels, test_size=0.2, random_state=41)
    dataset_train = TweetDataset(tweets_train, label_train, wordtoidx)
    dataset_val = TweetDataset(tweets_val, label_val, wordtoidx)
    return DataLoader(dataset = dataset_train, shuffle=True, batch_size = batch_size), DataLoader(dataset = dataset_val, shuffle=False, batch_size = batch_size)

def get_test_dataloader(tweet_path, label_path, wordtoidx, batch_size):
    pp = Preprocessor()
    tweets, labels = pp.get_test_data(tweet_path, label_path)
    dataset = TweetDataset(tweets, labels, wordtoidx)
    return DataLoader(dataset = dataset, shuffle=False, batch_size = batch_size)

def get_all_words_from_train(dataset_path):
    pp = Preprocessor()
    tweets, labels = pp.get_train_data(dataset_path)
    word_count_dict = {}
    for tweet in tweets:
        for word in tweet.split():
            if word not in word_count_dict:
                word_count_dict[word] = 0
            word_count_dict[word] += 1
    wordtoidx = {}
    total_words = 0
    for word in word_count_dict:
        if word_count_dict[word] == 1:
            continue
        wordtoidx[word] = total_words
        total_words += 1
    return wordtoidx

# Combine all words and load embeddings
def load_embeddings_n_words(dataset_path, embedding_path, embedding_type = "glove", embedding_dim = 50):
    if embedding_type == "glove":
        wordtoidx = get_all_words_from_train(dataset_path)
        embedding = GloveEmbedding(embedding_dim, wordtoidx, embedding_path)
        wordtoidx = embedding.wordtoidx
        return embedding, wordtoidx
    raise NotImplementedError(embedding_type + " not implemented")

def save_vocab(vocab, vocab_path):
    with open(vocab_path, "w") as f:
        f.write(json.dumps(vocab))

def load_saved_vocab(vocab_path):
    with open(vocab_path, "r") as f:
        vocab = f.read()
        vocab = json.loads(vocab)
    return vocab

def get_validation_metrics(model, loader, loss_fn, epoch_data = None, subtask = "a"):
    total_loss = 0
    model.eval()
    all_pred = torch.zeros(0)
    all_target = torch.zeros(0)

    with torch.no_grad():
        for X,Y in loader:
            Y_pred = model(X.to(device)).cpu()
            Y_pred = Y_pred.squeeze()
            total_loss += calculate_weighted_loss(Y_pred, Y, loss_fn, subtask).item()
            if len(Y_pred.shape) == 1:
                pred = (Y_pred > 0.5)
                all_pred = torch.cat((all_pred, pred), dim=0)
            else:
                pred = Y_pred.argmax(dim = -1)
                all_pred = torch.cat((all_pred, pred), dim=0)
            all_target = torch.cat((all_target, Y), dim=0)

    results = classification_report(y_true=all_target, y_pred=all_pred, output_dict=True, digits=4, zero_division = 1)

    if epoch_data is not None:
        add_epoch_data(all_pred, all_target, epoch_data, "val")
        epoch_data["val"]["loss"].append(total_loss / len(loader))

    print(f'precision: {results["macro avg"]["precision"]}, recall: {results["macro avg"]["recall"]}, f1 score: {results["macro avg"]["f1-score"]}')
    return results["accuracy"], total_loss / len(loader), results["macro avg"]["precision"], results["macro avg"]["recall"], results["macro avg"]["f1-score"]

def calculate_weighted_loss(pred, target, loss_fn, subtask = "subtask_a"):
    # class_multiplier = {"subtask_a":{0: 1, 1: 2}, "subtask_b":{0:1, 1:7}, "subtask_c": {0:2, 1:1,2:5}}
    class_multiplier = {"subtask_a":{0: 1, 1: 1}, "subtask_b":{0:1, 1:1}, "subtask_c": {0:1, 1:1,2:1}}
    if type(loss_fn) == nn.BCELoss:
        loss_1 = loss_fn(pred[target==1], target[target == 1]) if (target == 1).sum() > 0 else 0
        loss_0 = loss_fn(pred[target==0], target[target == 0]) if (target == 0).sum() > 0 else 0
        return (loss_1 * class_multiplier[subtask][1] + loss_0 * class_multiplier[subtask][0]) / ((loss_0!=0) + (loss_1!=0))
    if type(loss_fn) == nn.CrossEntropyLoss:
        loss_0 = loss_fn(pred[target == 0], target[target==0]) if (target == 0).sum() > 0 else 0
        loss_1 = loss_fn(pred[target == 1], target[target==1]) if (target == 1).sum() > 0 else 0
        loss_2 = loss_fn(pred[target == 2], target[target==2]) if (target == 2).sum() > 0 else 0
        return (loss_0 * class_multiplier[subtask][0] + loss_1 * class_multiplier[subtask][1] + loss_2 * class_multiplier[subtask][2]) / ((loss_0!=0) + (loss_1!=0) + (loss_2 != 0))

def early_stop(vals, crit = "max"):
    if len(vals) < 10:
        return False
    avg1 = sum(vals[-5:]) / 5
    avg2 = sum(vals[-15:]) / 15
    if avg1 < avg2:
        return True
    return False
    # if len(vals) < 8:
    #     return False
    # if vals.index(max(vals) if crit == "max" else min(vals)) + 8 < len(vals):
    #     return True
    # return False

def add_epoch_data(pred, target, epoch_data, key):
    results = classification_report(y_pred=pred, y_true=target, output_dict=True, digits=4, zero_division=1)
    epoch_data[key]["f1"].append(results["macro avg"]["f1-score"])
    epoch_data[key]["precision"].append(results["macro avg"]["precision"])
    epoch_data[key]["recall"].append(results["macro avg"]["recall"])
    epoch_data[key]["acc"].append(results["accuracy"])
    return epoch_data

def train_nn(train_options):
    if train_options["load_vocab"]:
        if train_options["use_cached_loaders"] and cached_loaders["vocab"] is not None:
            wordtoidx = cached_loaders["vocab"]
            embedding = cached_loaders["embedding"]
        else:
            wordtoidx = load_saved_vocab(train_options["vocab_path"])
            embedding = GloveEmbedding(train_options["embedding_dim"], wordtoidx, train_options["embedding_path"], False)
            cached_loaders["vocab"] = wordtoidx
            cached_loaders["embedding"] = embedding
    else:
        embedding, wordtoidx = load_embeddings_n_words(train_options["train_data_path"], train_options["embedding_path"], embedding_dim = train_options["embedding_dim"])
        save_vocab(wordtoidx, train_options["vocab_path"])

    print("Loading Complete.")

    if train_options["model_type"] == "attention":
        model = AttentionModel(embedding, train_options["embedding_dim"], out_dim=train_options["out_dim"], use_word_dropout=train_options["use_word_dropout"])
    elif train_options["model_type"] in ("attn", "lstm", "attn_cat"):
        model = AttentionExp(embedding, train_options["embedding_dim"], out_dim=train_options["out_dim"], use_word_dropout=train_options["use_word_dropout"], model_type=train_options["model_type"])
    else:
        model = RNNModel(embedding, train_options["embedding_dim"], out_dim = train_options["out_dim"], use_word_dropout=train_options["use_word_dropout"])
    model.to(device)

    if train_options["load_model"]:
        load_model(model, train_options["model_path"])

    if not train_options["train_model"]:
        return model, None

    if train_options["use_cached_loaders"] and cached_loaders[train_options["sub_task"]]["train_loader"] is not None:
        train_loader = cached_loaders[train_options["sub_task"]]["train_loader"]
        val_loader = cached_loaders[train_options["sub_task"]]["val_loader"]
        test_loader = cached_loaders[train_options["sub_task"]]["test_loader"]
    else:
        train_loader, val_loader = get_train_val_dataloader(train_options["train_data_path"], wordtoidx, train_options["batch_size"], train_options["sub_task"])
        test_loader = get_test_dataloader(train_options["test_tweet_path"], train_options["test_label_path"], wordtoidx, train_options["batch_size"])
        cached_loaders[train_options["sub_task"]]["train_loader"] = train_loader
        cached_loaders[train_options["sub_task"]]["val_loader"] = val_loader
        cached_loaders[train_options["sub_task"]]["test_loader"] = test_loader

    optimizer = Adam(model.parameters(), lr=train_options["lr"])
    epoch_stats = {"val": {"loss": [], "f1": [], "precision": [], "recall": [], "acc": []}, "train": {"loss": [], "f1": [], "precision": [], "recall": [], "acc": []}}

    if train_options["out_dim"] > 1:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCELoss()

    max_acc, temp, _, _, _ = get_validation_metrics(model, val_loader, loss_fn, subtask=train_options["sub_task"])

    for epoch_num in range(1, train_options["epochs"] + 1):
        print(f'Epoch: {epoch_num}')
        model.train()

        all_pred = torch.zeros(0)
        all_target = torch.zeros(0)
        total_train_loss = 0

        for X, Y in train_loader:
            Y_pred = model(X.to(device)).cpu()
            Y_pred = Y_pred.squeeze()
            loss = calculate_weighted_loss(Y_pred, Y, loss_fn, train_options["sub_task"])
            if len(Y_pred.shape) == 1:
                all_pred = torch.cat((all_pred, Y_pred > 0.5), dim=0)
                all_target = torch.cat((all_target, Y), dim=0)
            else:
                all_pred = torch.cat((all_pred, Y_pred.argmax(dim = -1)), dim=0)
                all_target = torch.cat((all_target, Y), dim=0)

            total_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_acc, val_loss, val_pred, val_rec, val_f1 = get_validation_metrics(model, val_loader, loss_fn, epoch_stats, subtask=train_options["sub_task"])

        add_epoch_data(all_pred, all_target, epoch_stats, "train")
        epoch_stats["train"]["loss"].append(total_train_loss / len(train_loader))

        print(f"Val Accuracy {val_acc}, val loss {val_f1}")
        print(f"train accuracy: {epoch_stats['train']['acc'][-1]}, total loss: {epoch_stats['train']['loss'][-1]}")

        if len(epoch_stats["val"]["f1"]) <= 1 or (train_options["save_model"] and val_loss < min(epoch_stats["val"]["loss"][:-1])):
            print("Saving Model")
            save_model(model, train_options["model_path"])

        # if early_stop(epoch_stats["val"]["loss"], "min"):
        #     print("EARLY STOPPING...")
        #     break

    if train_options["save_model"]:
        model = load_model(model, train_options["model_path"])

    print("--------------TEST STATS.----------------")
    test_acc, test_loss, test_prec, test_recall, test_f1 = get_validation_metrics(model, test_loader, loss_fn, subtask=train_options["sub_task"])
    print(f"Test accuracy: {test_acc}")
    return model, epoch_stats, test_acc, test_prec, test_recall, test_f1

cached_loaders = {
    "subtask_a": {
        "train_loader": None,
        "val_loader": None,
        "test_loader": None
    },
    "subtask_b": {
        "train_loader": None,
        "val_loader": None,
        "test_loader": None
    },
    "subtask_c": {
        "train_loader": None,
        "val_loader": None,
        "test_loader": None
    },
    "vocab": None,
    "embedding": None
}

train_options = {
    "embedding_dim": 300,
    "embedding_type": "glove",
    "use_cached_loaders": True,
    "embedding_path": "data/glove822/glove.6B.{dims}d.txt",
    "base_path": "",
    "train_model": True,
    "load_model": False,
    "save_model": True,
    "load_vocab": False,
    "use_word_dropout": True,
    "vocab_path": "data/vocab.json",
    "model_path": "saved_models/birnn_300.model",
    "train_data_path": "data/OLIDv1.0/olid-training-v1.0_clean3.tsv",
    "test_tweet_path": "data/OLIDv1.0/testset-levela_clean3.tsv",
    "test_label_path": "data/OLIDv1.0/labels-levela.csv",
    "out_dim": 1,
    "sub_task": "subtask_a",
    "batch_size": 32,
    "lr": 0.001,
    "epochs": 100,
    "model_type": "rnn" # attention | rnn
}

# train_options["sub_task"] = "subtask_c"
# train_options["test_tweet_path"] = "data/OLIDv1.0/testset-levelc_clean.tsv"
# train_options["test_label_path"] = "data/OLIDv1.0/labels-levelc.csv"
# train_options["model_path"] = "saved_models/birnn_300_taskc_att.model"
# train_options["out_dim"] = 3

def train_taska(model_type = "rnn"): # model_types = rnn | attention
    global train_options
    train_options["model_path"] = "saved_models/birnn_300_taska.model"
    train_options["model_type"] = model_type
    train_options = change_path_to_absolute(train_options)
    return train_nn(train_options)

def train_taskb(model_type = "rnn"): # model_types = rnn | attention
    global train_options
    train_options["sub_task"] = "subtask_b"
    train_options["test_tweet_path"] = "data/OLIDv1.0/testset-levelb_clean3.tsv"
    train_options["test_label_path"] = "data/OLIDv1.0/labels-levelb.csv"
    train_options["model_path"] = "saved_models/birnn_300_taskb.model"
    train_options["model_type"] = model_type
    train_options["out_dim"] = 1
    train_options = change_path_to_absolute(train_options)
    return train_nn(train_options)

def train_taskc(model_type = "rnn"): # model_types = rnn | attention
    global train_options
    train_options["sub_task"] = "subtask_c"
    train_options["test_tweet_path"] = "data/OLIDv1.0/testset-levelc_clean3.tsv"
    train_options["test_label_path"] = "data/OLIDv1.0/labels-levelc.csv"
    train_options["model_path"] = "saved_models/birnn_300_taskc.model"
    train_options["model_type"] = model_type
    train_options["out_dim"] = 3
    train_options = change_path_to_absolute(train_options)
    return train_nn(train_options)

def plot_attention_on_words(words, attention_mat):
    attention_mat = np.round(attention_mat.numpy(), decimals=2)
    row_labels = words[:]
    col_labels = words[:]
    fig, ax = plt.subplots()
    im = ax.imshow(attention_mat)
    ax.set_xticks(np.arange(len(row_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(row_labels)
    ax.set_yticklabels(col_labels)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, attention_mat[i, j],
                           ha="center", va="center", color="b")

    ax.set_title("Attentions heatmap b/w words.")
    # fig.tight_layout()
    plt.show()

def plot_attention(model, tweet_idx, train_options):
    model.eval()
    with torch.no_grad():
        if train_options["model_type"] == "attention":
            pp = Preprocessor()
            wordtoidx = load_saved_vocab(train_options["vocab_path"])
            tweets, labels = pp.get_test_data(train_options["test_tweet_path"], train_options["test_label_path"])
            tweets = tweets[tweet_idx:tweet_idx+1]
            labels = labels[tweet_idx:tweet_idx+1]
            dataset = TweetDataset(tweets, labels, wordtoidx=wordtoidx)
            tw_vec, lb_vec = dataset.__getitem__(0)
            attn_weights = model.get_attention_weights(tw_vec.unsqueeze(0))
            all_words = tweets[0].split()
            plot_attention_on_words(all_words, attn_weights[0, :len(all_words), :len(all_words)])
def train_n_plot_attention_taskc():
    train_options["load_vocab"] = True
    train_options["save_model"] = False
    train_options["load_model"] = True
    train_options["epochs"] = 0
    train_options["train_model"] = False

    model, epoch_stats = train_taskc(model_type="attention")
    plot_attention(model, 61, train_options)
    plot_attention(model, 55, train_options)
    plot_attention(model, 4, train_options)
    plot_attention(model, 9, train_options)
    plot_attention(model, 11, train_options)
    plot_attention(model, 136, train_options)
    plot_attention(model, 200, train_options)
    plot_attention(model, 59, train_options)
    print(epoch_stats)

def print_attn_exp_results(results):
    def avg(vals):
        return sum(vals) / len(vals)
    with open("attn_exp_res.txt", "w") as f:
        for subtask in results:
            res = f"_________{subtask}___________"
            f.write(res)
            print(res)
            for model_type in results[subtask]:
                if len(results[subtask][model_type]["f1"]) > 0:
                    mod_res = results[subtask][model_type]
                    res = f"{model_type}: F1: {avg(mod_res['f1'])}, acc: {avg(mod_res['acc'])}, precision: {avg(mod_res['precision'])}, recall: {avg(mod_res['recall'])}, totel_iter: {len(mod_res['acc'])}"
                    f.write(res + "\n\n")
                    print(res)

def run_attention_experiment():
    exp_results = {"subtask_a": {"lstm":{"precision": [], "recall":[], "acc":[], "f1":[]}, "attn": {"precision": [], "recall":[], "acc":[], "f1":[]}, "attn_cat": {"precision": [], "recall":[], "acc":[], "f1":[]}},
                   "subtask_b": {"lstm":{"precision": [], "recall":[], "acc":[], "f1":[]}, "attn": {"precision": [], "recall":[], "acc":[], "f1":[]}, "attn_cat": {"precision": [], "recall":[], "acc":[], "f1":[]}},
                   "subtask_c": {"lstm":{"precision": [], "recall":[], "acc":[], "f1":[]}, "attn": {"precision": [], "recall":[], "acc":[], "f1":[]}, "attn_cat": {"precision": [], "recall":[], "acc":[], "f1":[]}}}

    sub_task_params = {
        "subtask_a": {
            "out_dim": 1,
            "test_tweet_path": "data/OLIDv1.0/testset-levela_clean3.tsv",
            "test_label_path": "data/OLIDv1.0/labels-levela.csv"
        },
        "subtask_b": {
            "out_dim": 1,
            "test_tweet_path": "data/OLIDv1.0/testset-levelb_clean3.tsv",
            "test_label_path": "data/OLIDv1.0/labels-levelb.csv"
        },
        "subtask_c": {
            "out_dim": 3,
            "test_tweet_path": "data/OLIDv1.0/testset-levelc_clean3.tsv",
            "test_label_path": "data/OLIDv1.0/labels-levelc.csv"
        },
    }

    for subtask in ("subtask_a", "subtask_b", "subtask_c"): # for each subtask
        for exp_num in range(10): # 10 experiments
            for model_type in ("lstm", "attn", "attn_cat"):
                print(f"TRAINING: {subtask} on {model_type}")
                train_ops = copy.deepcopy(train_options)
                train_ops["sub_task"] = subtask
                train_ops["test_tweet_path"] = sub_task_params[subtask]["test_tweet_path"]
                train_ops["test_label_path"] = sub_task_params[subtask]["test_label_path"]
                train_ops["model_path"] = "saved_models/birnn_300_taskc.model"
                train_ops["model_type"] = model_type
                train_ops["epochs"] = 40
                train_ops["batch_size"] = 128
                train_ops["lr_rate"] = 0.0005
                train_ops["out_dim"] = sub_task_params[subtask]["out_dim"]
                train_ops["use_cached_loaders"] = True
                train_ops["load_vocab"] = False
                train_ops["load_model"] = False
                train_ops = change_path_to_absolute(train_ops)
                model, epoch_stats, test_acc, test_prec, test_recall, test_f1 = train_nn(train_ops)

                exp_results[subtask][model_type]["acc"].append(test_acc)
                exp_results[subtask][model_type]["precision"].append(test_prec)
                exp_results[subtask][model_type]["recall"].append(test_recall)
                exp_results[subtask][model_type]["f1"].append(test_f1)

                print_attn_exp_results(exp_results)
    return exp_results

def run_paper_experiment():
    exp_results = {"subtask_a": {"rnn":{"precision": [], "recall":[], "acc":[], "f1":[]}, "attention": {"precision": [], "recall":[], "acc":[], "f1":[]}},
                   "subtask_b": {"rnn":{"precision": [], "recall":[], "acc":[], "f1":[]}, "attention": {"precision": [], "recall":[], "acc":[], "f1":[]}},
                   "subtask_c": {"rnn":{"precision": [], "recall":[], "acc":[], "f1":[]}, "attention": {"precision": [], "recall":[], "acc":[], "f1":[]}}}
    sub_task_params = {
        "subtask_a": {
            "out_dim": 1,
            "test_tweet_path": "data/OLIDv1.0/testset-levela_clean3.tsv",
            "test_label_path": "data/OLIDv1.0/labels-levela.csv"
        },
        "subtask_b": {
            "out_dim": 1,
            "test_tweet_path": "data/OLIDv1.0/testset-levelb_clean3.tsv",
            "test_label_path": "data/OLIDv1.0/labels-levelb.csv"
        },
        "subtask_c": {
            "out_dim": 3,
            "test_tweet_path": "data/OLIDv1.0/testset-levelc_clean3.tsv",
            "test_label_path": "data/OLIDv1.0/labels-levelc.csv"
        },
    }

    for subtask in ("subtask_a", "subtask_b", "subtask_c"): # for each subtask
        for exp_num in range(10):
            for model_type in ("rnn", "attention"):
                print(f"TRAINING: {subtask} on {model_type}")
                train_ops = copy.deepcopy(train_options)
                train_ops["sub_task"] = subtask
                train_ops["test_tweet_path"] = sub_task_params[subtask]["test_tweet_path"]
                train_ops["test_label_path"] = sub_task_params[subtask]["test_label_path"]
                train_ops["model_path"] = "saved_models/birnn_300_taskc.model"
                train_ops["model_type"] = model_type
                train_ops["epochs"] = 40
                train_ops["batch_size"] = 128
                train_ops["lr_rate"] = 0.001
                train_ops["out_dim"] = sub_task_params[subtask]["out_dim"]
                train_ops["use_cached_loaders"] = False
                train_ops["load_vocab"] = False
                train_ops["load_model"] = False
                train_ops = change_path_to_absolute(train_ops)
                model, epoch_stats, test_acc, test_prec, test_recall, test_f1 = train_nn(train_ops)

                exp_results[subtask][model_type]["acc"].append(test_acc)
                exp_results[subtask][model_type]["precision"].append(test_prec)
                exp_results[subtask][model_type]["recall"].append(test_recall)
                exp_results[subtask][model_type]["f1"].append(test_f1)

                print_attn_exp_results(exp_results)

# exp_results = run_attention_experiment()
exp_results = run_paper_experiment()
