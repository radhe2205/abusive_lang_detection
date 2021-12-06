from torch import nn
import torch
import numpy as np

class TextDataset(nn.Module):
    def __init__(self, X,Y):
        self.X = torch.from_numpy(X.toarray())
        self.Y = Y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]