
import torch
from torch import nn
import numpy as np

class LogisticRegressionModel(nn.Module):
    def __init__(self,input_dim):
        super(LogisticRegression, self).__init__()

        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(in_features=self.input_dim,out_features=1),
            nn.Sigmoid()        
        )
    
    def forward(self, x):
        output = self.model(x.float())
        return output