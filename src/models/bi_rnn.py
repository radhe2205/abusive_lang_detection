import torch
from torch import nn

# model_arch = {
#     "RNN_layers": [
#     {"layer_type": "lstm", "in_dim": 50, "hidden_dim": 100, "bi_dir": True, "dropout": 0.5, "num_layers": 2}],
#
#     "Classification_Layers": [
#         {"layer_type": "dropout", "val": 0.4},
#         {"layer_type":"batch_norm1", "in_dim": 100},
#         {"layer_type": "linear", "in_dim": 100, "out_dim": 1, "activation": "sigmoid"},
#     ]
# }
# def get_lstm_layer(config):
#     seq_layer = nn.LSTM
#     if (config["layer_type"] == "lstm"):
#         seq_layer = nn.LSTM
#     elif (config["layer_type"] == "gru"):
#         seq_layer = nn.GRU
#
#     return seq_layer(input_size=config["in_dim"], hidden_size = "hidden_dim", bidirectional=config["bi_dir"], dropout=config["dropout"])
#
# def get_linear_layer(config):
#     fc = nn.Linear()
#
# def create_nn_layers(config):
#

class RNNModel(nn.Module):
    def __init__(self, embeddings, in_dim, num_layers = 1, hidden_size = 100, out_dim = 1):
        super(RNNModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.out_dim = out_dim

        self.embeddings = embeddings

        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.5)

        self.linear_layers = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 2),
            nn.Dropout(0.4),
            nn.Linear(self.hidden_size * 2, self.out_dim),
            (nn.Sigmoid(), nn.Identity())[out_dim >= 2]
        )

        # Weight Initialization
        for layer in self.linear_layers:
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer.weight.data)

    # number of words in tweet are limited, will use padded fixed length sequence.
    def forward(self, samples):
        word_embs = self.embeddings.get_embeddings(samples)
        o, (h,c) = self.rnn(word_embs)
        # o = o[:,-1,:]
        o = torch.cat((o[:,-1,:self.hidden_size], o[:, 0, self.hidden_size:]), dim=-1)
        return self.linear_layers(o)

    # TO be used to save/load the model with different parameter
    def get_model_name(self):
        return "bi_lstm_{emb_dim}_{num_layers}_{hid_size}_{out_dim}".format(emb_dim = self.embedding_dim,
                                                                            num_layers = self.num_layers,
                                                                            hid_size = self.hidden_size,
                                                                            out_dim = self.out_dim)
