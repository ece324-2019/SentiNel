import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy
import argparse
import os
import matplotlib.pyplot as plt

from entity_models import *

def load_model(lr, vocab, name): #loads model, loss function and optimizer
    if name == 'cnn':
        model = Baseline(100, vocab)
    elif name == "rnn":
        model = CNN(args.emb_dim,vocab, args.num_filt,[2,4])
    elif name == "bi_rnn":
        model = RNN(args.emb_dim, vocab, args.rnn_hidden_dim)
    elif name == "lstm":
        model =
    elif name == "bi_lstm":
        model =
    loss_fnc = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    return model, loss_fnc, optimizer

def evaluate(model, loader):
    total_corr = 0
    for i, vbatch in enumerate(loader):
        feats, length = vbatch.text
        label = vbatch.label
        prediction = model(feats,length)
        # prediction = torch.sigmoid(prediction)
        for j in range(len(prediction)):
            if (prediction[j] > 0.50) and (label[j] == 1):
                total_corr += 1
            elif (prediction[j] <= 0.50) and (label[j] == 0):
                total_corr += 1


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

    train_data, val_data, test_data = data.TabularDataset.splits(
        path='data/', train='train.tsv',
        validation='validation.tsv', test='test.tsv', format='tsv',
        skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    overfit_data, val_data, test_data = data.TabularDataset.splits(
        path='data/', train='overfit.tsv',
        validation='validation.tsv', test='test.tsv', format='tsv',
        skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    # 3.2.3
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    overfit_iter, val_iter, test_iter = data.BucketIterator.splits(
        (overfit_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    print("Shape of Vocab:",TEXT.vocab.vectors.shape)

    model, loss_fnc, optimizer = load_model(args.lr,vocab,args.model)

    trainplotacc = []
    trainplotloss = []

    train_accs = []
    t_loss_array = []
    valid_accs = []
    v_loss_array = []
    tvals = []
    timearray = []

    t = 0
    start = time.time()
    for epoch in range(0, epochs):
        accum_loss = 0
        v_accum_loss = 0
        t_accum_loss = 0
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()

            batch_input, batch_input_length = batch.text
            predictions = model(batch_input, batch_input_length)
            batch_loss = loss_fnc(input=predictions, target=batch.label.float())
            accum_loss = accum_loss + batch_loss.item()
            batch_loss.backward()
            optimizer.step()
        train_loss = accum_loss / (len(train_iter))
        train_acc = eval(model, train_iter)
        train_accs = train_accs + [train_acc]
        t_loss_array = t_loss_array + [train_loss]
        # t_loss_array = t_loss_array + [batch_loss.item()]
        accum_loss = 0

        for j, v_batch in enumerate(val_iter):
            v_batch_input, v_batch_input_length = v_batch.text
            predictions = model(v_batch_input, v_batch_input_length)
            v_batch_loss = loss_fnc(input=predictions, target=v_batch.label.float())
            v_accum_loss = v_accum_loss + v_batch_loss.item()

        valid_acc = eval(model, val_iter)
        valid_accs = valid_accs + [valid_acc]
        valid_loss = v_accum_loss / (len(val_iter))
        v_loss_array = v_loss_array + [valid_loss]
        # v_loss_array = v_loss_array + [v_batch_loss.item()]
        v_accum_loss = 0
        end = time.time()

        for k, t_batch in enumerate(test_iter):
            t_batch_input, t_batch_input_length = t_batch.text
            predictions = model(t_batch_input, t_batch_input_length)
            t_batch_loss = loss_fnc(input=predictions, target=t_batch.label.float())
            t_accum_loss = t_accum_loss + t_batch_loss.item()

        test_loss = t_accum_loss / (len(test_iter))
        t_accum_loss = 0

        print(
            "Epoch: {}, Step: {} | Train Loss: {} | Valid loss: {}| test loss: {}| Valid acc: {} | train acc:{} ".format(
                epoch + 1, t + 1, train_loss,
                valid_loss, test_loss, valid_acc, train_acc))
        tvals = tvals + [t + 1]
        timearray = timearray + [end - start]
        t = t + 1

    test_accuracy = eval(model, test_iter)
    # torch.save(model, 'model_rnn.pt')
    print("Test accuracy = ", test_accuracy)
    print("Time taken to execute: ", timearray[len(timearray) - 1])
    plt.figure()
    lines = plt.plot(tvals, train_accs, 'r--', valid_accs, 'b')
    # plt.plot(tvals, train_accs, 'r')
    # plt.plot(tvals, valid_accs, 'b')
    plt.title("Accuracy " + " vs." + " Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(lines[0:2], ['Training', 'Validation'])
    plt.show()
    plt.close()

    plt.figure()
    lines = plt.plot(tvals, t_loss_array, 'r--', v_loss_array, 'b')
    # plt.plot(tvals, t_loss_array, 'r')
    # plt.plot(tvals, v_loss_array, 'b')
    plt.title("Loss " + " vs." + " Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(lines[0:2], ['Training', 'Validation'])
    plt.show()
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='lstm_rnn', help="Model type: rnn,cnn,baseline (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)
    args = parser.parse_args()

    main(args)