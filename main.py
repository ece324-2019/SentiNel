import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy
import argparse
import os
import matplotlib.pyplot as plt

from models import *

def load_model(lr,vocab,name): #loads model, loss function and optimizer
    if name == "cnn":
        model = CNN(args.emb_dim,vocab, args.num_filt,[2,4])
    else:
        model = RNN(args.emb_dim, vocab, args.rnn_hidden_dim)

    loss_fnc = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    return model, loss_fnc, optimizer

def evaluate(model, loader):
    total_corr = 0
    for i, vbatch in enumerate(loader):
        feats, length = vbatch.text
        label = vbatch.label
        prediction = model(feats,length)
        corr = (prediction > 0.5).squeeze().long() == label.long()

        total_corr += int(corr.sum())
    return float(total_corr)/len(loader.dataset)

def validlosscalc(model,loss_fnc, loader):
    accum_loss = 0.0
    for i, vbatch in enumerate(loader):
        feats, length = vbatch.text
        label = vbatch.label
        prediction = model(feats,length)
        batch_loss = loss_fnc(input=prediction, target=label.float())

        accum_loss += batch_loss.item()

    return accum_loss/len(loader)

def main(args):
    torch.manual_seed(1)

    TEXT = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)

    overfit_data = data.TabularDataset(path='data/overfit.tsv',format='tsv',skip_header=True,fields=[('label', LABELS),('text', TEXT)])

    overfit_iter = data.BucketIterator((overfit_data), batch_size=(args.batch_size),sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    TEXT.build_vocab(overfit_data)

    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab

    print("Shape of Vocab:",TEXT.vocab.vectors.shape)

    model, loss_fnc, optimizer = load_model(args.lr,vocab,args.model)

    trainplotacc = []
    trainplotloss = []


    for epoch in range(args.epochs):
        accum_loss = 0.0
        for i, batch in enumerate(overfit_iter):
            optimizer.zero_grad()
            feats, length = batch.text
            label = batch.label


            predictions = model(feats,length)
            batch_loss = loss_fnc(input=predictions, target=label.float())

            accum_loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()
        train_acc = evaluate(model, overfit_iter)
        train_loss = accum_loss/len(overfit_iter)

        trainplotacc.append(train_acc)
        trainplotloss.append(train_loss)
        print("Epoch: {} |Train Acc:{}| Train Loss: {}".format(epoch + 1, train_acc,train_loss))

    print("Test Accuracy = ", train_acc)
    plt.plot(trainplotacc, 'b', label="Train")
    plt.title("Accuracy vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.plot(trainplotloss, 'b', label="Train")
    plt.title("Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='cnn', help="Model type: rnn,cnn (Default: cnn)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()

    main(args)
