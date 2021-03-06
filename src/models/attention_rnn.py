import torch
from torch import nn

class AttentionModel(nn.Module):
    def __init__(self, embeddings, in_dim, num_layers = 1, hidden_size = 100, out_dim = 1, use_word_dropout = False):
        super(AttentionModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.use_word_dropout = use_word_dropout

        self.embeddings = embeddings

        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.5)

        # self.attn_param = AttentionParam(self.hidden_size * 2, self.hidden_size)
        # self.attn_param = AttentionParam(embeddings.fixed_embedding.weight.shape[-1], self.hidden_size)

        self.multi_attn = nn.MultiheadAttention(embed_dim=self.hidden_size * 2, num_heads=4, dropout=0.5) # Batch First = False

        self.rnn2 = nn.LSTM(input_size = self.hidden_size * 2, hidden_size=hidden_size, num_layers = 1, batch_first=True, bidirectional=True, dropout=0.5)

        self.word_dropout = nn.Dropout(0.1)

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

    def get_attention_weights(self, samples):
        if self.use_word_dropout:
            padding_idxes = samples == -1
            drp_inp = torch.ones(samples.shape).to(samples.device)
            drp_idxes = self.word_dropout(drp_inp) == 0
            samples[drp_idxes] = self.embeddings.unk_idx
            samples[padding_idxes] = -1

        word_embs = self.embeddings.get_embeddings(samples)
        o, (h,c) = self.rnn(word_embs)
        # q,k,v = self.attn_param(o)
        o = o.transpose(0,1)
        o, o_weights= self.multi_attn(o,o,o)
        return o_weights

    # number of words in tweet are limited, will use padded fixed length sequence.
    def forward(self, samples):
        if self.use_word_dropout:
            padding_idxes = samples == -1
            drp_inp = torch.ones(samples.shape).to(samples.device)
            drp_idxes = self.word_dropout(drp_inp) == 0
            samples[drp_idxes] = self.embeddings.unk_idx
            samples[padding_idxes] = -1

        word_embs = self.embeddings.get_embeddings(samples)
        o, (h,c) = self.rnn(word_embs)
        # q,k,v = self.attn_param(o)
        o = o.transpose(0,1)
        o, o_weights= self.multi_attn(o,o,o)
        o = o.transpose(0,1)
        # o = o.reshape(o.shape[1], o.shape[0], -1)
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
