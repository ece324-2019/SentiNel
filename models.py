import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):

    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        #x = [sentence length, batch size]
        embedded = self.embedding(x)
        average = embedded.mean(0) # [sentence length, batch size, embedding_dim]
        output = self.fc(average).squeeze(1)
        return output

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