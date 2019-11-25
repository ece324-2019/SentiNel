import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

from models import *

def load_model(lr,vocab,name): #loads model, loss function and optimizer
    if name == 'baseline':
        model = Baseline(100, vocab)
    elif name == "cnn":
        model = CNN(args.emb_dim,vocab, args.num_filt,[2,4])
    else:
        model = RNN(args.emb_dim, vocab, args.rnn_hidden_dim)

    if torch.cuda.is_available():
        print("Using Cuda")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        model.cuda()
    loss_fnc = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    return model, loss_fnc, optimizer

def evaluate(model, loader):
    total_corr = 0
    for i, vbatch in enumerate(loader):
        feats, length = vbatch.text
        label = vbatch.label
        prediction = model(feats,length)
        prediction = torch.sigmoid(prediction)
        for j in range(len(prediction)):
            if (prediction[j] > 0.50) and (label[j] == 1):
                total_corr += 1
            elif (prediction[j] <= 0.50) and (label[j] == 0):
                total_corr += 1
    return float(total_corr)/len(loader.dataset)

def confustionmatrix(model,loader):
    truth = []
    couldbe = []
    for i, vbatch in enumerate(loader):
        feats, length = vbatch.text
        label = vbatch.label
        prediction = model(feats, length)
        # prediction = torch.sigmoid(prediction)
        for j in range(len(prediction)):
            truth += [int(label[j])]
            if (prediction[j] > 0.50):
                couldbe += [1]
            elif (prediction[j] <= 0.50):
                couldbe += [0]
    return truth, couldbe

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

    train_data, val_data, test_data = data.TabularDataset.splits(path='data/', train='train.tsv',
                                                                 validation='validation.tsv', test='test.tsv',
                                                                 format='tsv',
                                                                 skip_header=True,
                                                                 fields=[('label', LABELS),('text', TEXT)])

    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    overfit_data = data.TabularDataset(path='data/overfit.tsv',format='tsv',skip_header=True,fields=[('label', LABELS),('text', TEXT)])

    overfit_iter = data.Iterator((overfit_data), batch_size=(args.batch_size),sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    TEXT.build_vocab(overfit_data, train_data, val_data, test_data)

    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab

    print("Shape of Vocab:",TEXT.vocab.vectors.shape)

    model, loss_fnc, optimizer = load_model(args.lr,vocab,args.model)

    # train_iter = overfit_iter

    trainplotacc = []
    validplotacc = []
    trainplotloss = []
    validplotloss = []
    for epoch in range(args.epochs):
        accum_loss = 0.0
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            feats, length = batch.text
            label = batch.label

            predictions = model(feats,length)
            batch_loss = loss_fnc(input=predictions, target=label.float())

            accum_loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

        train_acc = evaluate(model, train_iter)
        train_loss = accum_loss/len(train_iter)
        valid_acc = evaluate(model, val_iter)
        valid_loss = validlosscalc(model, loss_fnc, val_iter)

        trainplotacc.append(train_acc)
        validplotacc.append(valid_acc)
        trainplotloss.append(train_loss)
        validplotloss += [valid_loss]
        print("Epoch: {} | Train Acc:{}| Train Loss: {} | Valid Acc:{}| Valid Loss: {}".format(epoch + 1, train_acc, train_loss, valid_acc, valid_loss))

    # truth, couldbe = confustionmatrix(model,val_iter)
    # a = confusion_matrix(couldbe,truth)
    # df_cm = pd.DataFrame(a, range(2),range(2))
    # sn.set(font_scale=1.4)  # for label size
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    # plt.show()

    print("Test Accuracy = ", evaluate(model, test_iter))
    plt.plot(trainplotacc, 'b--', label="Train")
    plt.plot(validplotacc, 'r', label="Valid")
    # plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    plt.title("Accuracy vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.plot(trainplotloss, 'b--', label="Train")
    plt.plot(validplotloss, 'r', label="Valid")
    plt.title("Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model', type=str, default='cnn', help="Model type: rnn,cnn,baseline (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()

    main(args)
