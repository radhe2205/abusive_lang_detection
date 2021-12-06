from torch import nn
import torch

class CharLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, d=50):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, d, padding_idx=input_size-1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=d, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True, dropout=0.5)
        self.linear_layers = nn.Sequential(
            nn.BatchNorm1d(hidden_size*2),
            nn.Linear(hidden_size*2, output_size),
            nn.Sigmoid()
        )

        for layer in self.linear_layers:
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer.weight.data)

    def forward(self, input_seq):
        embedding = self.embedding(input_seq)
        o, hidden_state = self.lstm(embedding)
        output = torch.cat((o[:,-1,:self.hidden_size], o[:, 0, self.hidden_size:]), dim=-1)
        output = self.linear_layers(output)
        return output