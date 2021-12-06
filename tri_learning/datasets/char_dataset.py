from torch import nn

class CharDataset(nn.Module):
    def __init__(self, tweets, labels):
        self.tweets = tweets
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.tweets[index], self.labels[index]

