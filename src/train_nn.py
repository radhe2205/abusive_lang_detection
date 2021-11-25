import json

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from preprocessing import Preprocessor
from src.models.bi_rnn import RNNModel
from src.models.embedding import GloveEmbedding
from src.tweet_dataset import TweetDataset
from src.utils import change_path_to_absolute, save_model, load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_train_dataloader(dataset_path, wordtoidx, batch_size, task = "subtask_a"):
    pp = Preprocessor()
    tweets, labels = pp.get_train_data(dataset_path, task)
    dataset = TweetDataset(tweets, labels, wordtoidx)
    return DataLoader(dataset = dataset, shuffle=True, batch_size = batch_size)

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

def check_val_accuracy(model, loader, loss_fn):
    total_correct = 0
    total_loss = 0
    model.eval()
    fp = 0; tp = 0; fn = 0

    with torch.no_grad():
        for X,Y,w_len in loader:
            Y_pred = model(X.to(device), w_len.to(device)).cpu()
            Y_pred = Y_pred.squeeze()
            total_loss += calculate_weighted_loss(Y_pred, Y, loss_fn).item()
            if len(Y_pred.shape) == 1:
                pred = (Y_pred > 0.5)
                total_correct += (Y == (Y_pred > 0.5)).sum().item()
                tp += (Y[pred == 1] == 1).sum()
                fp += (pred[Y == 0] == 1).sum()
                fn += (pred[Y == 1] == 0).sum()
            else:
                pred = Y_pred.argmax(dim = -1)
                total_correct += (Y == Y_pred.argmax(dim=-1)).sum().item()

    precision = tp / (fp + tp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    print(f"precision: {precision}, recall: {recall}, f1 score: {f1_score}")
    return total_correct / loader.dataset.__len__(), total_loss / loader.dataset.__len__()

def calculate_weighted_loss(pred, target, loss_fn):
    if type(loss_fn) == nn.BCELoss:
        loss_1 = loss_fn(pred[target==1], target[target == 1])
        loss_0 = loss_fn(pred[target==0], target[target == 0])
        return loss_1 * 2 + loss_0


def train_nn(train_options):
    if train_options["load_vocab"]:
        wordtoidx = load_saved_vocab(train_options["vocab_path"])
        embedding = GloveEmbedding(train_options["embedding_dim"], wordtoidx, train_options["embedding_path"], False)
    else:
        embedding, wordtoidx = load_embeddings_n_words(train_options["train_data_path"], train_options["embedding_path"], embedding_dim = train_options["embedding_dim"])
        save_vocab(wordtoidx, train_options["vocab_path"])

    model = RNNModel(embedding, train_options["embedding_dim"], out_dim = train_options["out_dim"])
    model.to(device)

    if train_options["load_model"]:
        load_model(model, train_options["model_path"])

    if not train_options["train_model"]:
        return model

    train_loader = get_train_dataloader(train_options["train_data_path"], wordtoidx, train_options["batch_size"], train_options["sub_task"])
    test_loader = get_test_dataloader(train_options["test_tweet_path"], train_options["test_label_path"], wordtoidx, train_options["batch_size"])

    optimizer = Adam(model.parameters(), lr=train_options["lr"])

    if train_options["out_dim"] > 1:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCELoss()

    max_acc, temp = check_val_accuracy(model, test_loader, loss_fn)

    for epoch_num in range(1, train_options["epochs"] + 1):
        print(f'Epoch: {epoch_num}')
        model.train()
        total_correct = 0
        total_loss = 0
        total_items = train_loader.dataset.__len__()
        for X, Y, w_len in train_loader:
            Y_pred = model(X.to(device), w_len.to(device)).cpu()
            Y_pred = Y_pred.squeeze()
            loss = calculate_weighted_loss(Y_pred, Y, loss_fn)
            if len(Y_pred.shape) == 1:
                total_correct += (Y == (Y_pred>0.5)).sum().item()
            else:
                total_correct += (Y==Y_pred.argmax(dim = -1)).sum().item()
            total_loss+= loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_acc, test_loss = check_val_accuracy(model, test_loader, loss_fn)
        print(f"Test Accuracy {test_acc}, test loss {test_loss}")
        print(f"train accuracy: {total_correct / total_items}, total loss: {total_loss / total_items}")
        if max_acc <= test_acc:
            save_model(model, train_options["model_path"])
            max_acc = test_acc

train_options = {
    "embedding_dim": 300,
    "embedding_type": "glove",
    "embedding_path": "data/glove822/glove.6B.{dims}d.txt",
    "base_path": "",
    "train_model": True,
    "load_model": True,
    "save_model": True,
    "load_vocab": True,
    "vocab_path": "data/vocab.json",
    "model_path": "saved_models/birnn_300.model",
    "train_data_path": "data/OLIDv1.0/olid-training-v1.0.tsv",
    "test_tweet_path": "data/OLIDv1.0/testset-levela.tsv",
    "test_label_path": "data/OLIDv1.0/labels-levela.csv",
    "out_dim": 1,
    "sub_task": "subtask_a",
    "batch_size": 128,
    "lr": 0.001,
    "epochs": 100
}

train_options = change_path_to_absolute(train_options)

train_nn(train_options)
