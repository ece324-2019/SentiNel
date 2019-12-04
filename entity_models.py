import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()

        self.embed = nn.Embedding(len(vocab), embedding_dim)
        self.embed.from_pretrained(vocab.vectors)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_filters, (filter_sizes[0], embedding_dim)),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, n_filters, (filter_sizes[1], embedding_dim)),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(100, 1)
        )

    def forward(self, x, length=None):

        x = self.embed(x)
        x = x.permute(1, 0, 2)
        x = x.unsqueeze(1)
        x_1 = self.conv1(x)
        x_1, _ = torch.max(x_1, 2)
        x_2 = self.conv2(x)
        x_2, _ = torch.max(x_2, 2)
        x = torch.cat([x_1, x_2], 1).squeeze()
        x = self.linear(x)

        return x.squeeze()

class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(RNN, self).__init__()

        self.embed = nn.Embedding(len(vocab), embedding_dim)
        self.embed.from_pretrained(vocab.vectors)
        self.rnn = nn.GRU(embedding_dim, hidden, bidirectional=True)
        self.linear = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())


    def forward(self, x, length=None):

        x = self.embed(x)
        x = x.permute(1, 0, 2)
        x = x.unsqueeze(1)
        x = self.linear(x)

        return x.squeeze()


class LSTM(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()

        self.embed = nn.Embedding(len(vocab), embedding_dim)
        self.embed.from_pretrained(vocab.vectors)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_filters, (filter_sizes[0], embedding_dim)),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, n_filters, (filter_sizes[1], embedding_dim)),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(100, 1)
        )

    def forward(self, x, length=None):

        x = self.embed(x)
        x = x.permute(1, 0, 2)
        x = x.unsqueeze(1)
        x_1 = self.conv1(x)
        x_1, _ = torch.max(x_1, 2)
        x_2 = self.conv2(x)
        x_2, _ = torch.max(x_2, 2)
        x = torch.cat([x_1, x_2], 1).squeeze()
        x = self.linear(x)

        return x.squeeze()

class biLSTM(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()

        self.embed = nn.Embedding(len(vocab), embedding_dim)
        self.embed.from_pretrained(vocab.vectors)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_filters, (filter_sizes[0], embedding_dim)),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, n_filters, (filter_sizes[1], embedding_dim)),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(100, 1)
        )

    def forward(self, x, length=None):

        x = self.embed(x)
        x = x.permute(1, 0, 2)
        x = x.unsqueeze(1)
        x_1 = self.conv1(x)
        x_1, _ = torch.max(x_1, 2)
        x_2 = self.conv2(x)
        x_2, _ = torch.max(x_2, 2)
        x = torch.cat([x_1, x_2], 1).squeeze()
        x = self.linear(x)

        return x.squeeze()
