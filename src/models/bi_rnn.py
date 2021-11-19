import torch
from torch import nn

from src.utils import load_embeddings_from_path


class BiLSTMModel(nn.Module):
    def __init__(self, embedding_dim, embedding_path, embedding_type = "glove",
                 num_layers = 2, hidden_size = 100, out_dim = 1):
        super(BiLSTMModel, self).__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.out_dim = out_dim

        self.embeddings = load_embeddings_from_path(embedding_dim, embedding_type, embedding_path)

        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True, dropout=0.2)

        self.linear_layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.BatchNorm1d(self.hidden_size * 2),
            nn.Linear(self.hidden_size * 2, self.out_dim),
            (nn.Sigmoid(), nn.Softmax())[out_dim > 2]
        )

        # Weight Initialization
        for layer in self.linear_layers:
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer.weight.data)

    # number of words in tweet are limited, will use padded fixed length sequence.
    def forward(self, samples):
        # TODO
        # Use last cell bidirectional output of lstm
        # Pass to linear layer
        pass

    # TO be used to save/load the model with different parameter
    def get_model_name(self):
        return "bi_lstm_{emb_dim}_{num_layers}_{hid_size}_{out_dim}".format(emb_dim = self.embedding_dim,
                                                                            num_layers = self.num_layers,
                                                                            hid_size = self.hidden_size,
                                                                            out_dim = self.out_dim)
