import torch
from torch import nn

class AttentionParam(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(AttentionParam, self).__init__()
        self.in_dim = in_feature
        self.out_dim = out_feature
        self.query = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.key = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.value = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.tanh = nn.Tanh()

        nn.init.xavier_normal_(self.query.weight)
        nn.init.xavier_normal_(self.key.weight)
        nn.init.xavier_normal_(self.value.weight)

    def forward(self, samples): # batch x seq_len x 2*hidden_size
        q = self.query(samples).reshape(samples.shape[1], samples.shape[0], -1)
        k = self.key(samples).reshape(samples.shape[1], samples.shape[0], -1)
        v = self.value(samples).reshape(samples.shape[1], samples.shape[0], -1)

        return self.tanh(q), self.tanh(k), self.tanh(v)



class AttentionModel(nn.Module):
    def __init__(self, embeddings, in_dim, num_layers = 1, hidden_size = 100, out_dim = 1):
        super(AttentionModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.out_dim = out_dim

        self.embeddings = embeddings

        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.5)

        self.attn_param = AttentionParam(self.hidden_size * 2, self.hidden_size)
        self.attn_param = AttentionParam(embeddings.fixed_embedding.weight.shape[-1], self.hidden_size)

        self.multi_attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=4, dropout=0.) # Batch First = False

        self.rnn2 = nn.LSTM(input_size = self.hidden_size, hidden_size=hidden_size, num_layers = 1, batch_first=True, bidirectional=True, dropout=0.5)

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
    def forward(self, samples, w_lens):
        word_embs = self.embeddings.get_embeddings(samples, w_lens)
        o, (h,c) = self.rnn(word_embs)
        q,k,v = self.attn_param(o)
        o, o_weights= self.multi_attn(q,k,v)
        o = o.reshape(o.shape[1], o.shape[0], -1)
        o, (h,c) = self.rnn2(o)
        # o = o[:,-1,:]
        o = torch.cat((o[:,-1,:self.hidden_size], o[:, 0, self.hidden_size:]), dim=-1)
        return self.linear_layers(o)

    # TO be used to save/load the model with different parameter
    def get_model_name(self):
        return "attention_{emb_dim}_{num_layers}_{hid_size}_{out_dim}".format(emb_dim = self.embedding_dim,
                                                                            num_layers = self.num_layers,
                                                                            hid_size = self.hidden_size,
                                                                            out_dim = self.out_dim)
