import torch
from torchtext import data
import argparse
import spacy
import torchtext


def tokenizer(text):
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en(text)]

def main(args):
    TEXT = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)

    train_data, val_data, test_data = data.TabularDataset.splits(path='data/', train='train.tsv',
                                                                 validation='validation.tsv', test='test.tsv',
                                                                 format='tsv',
                                                                 skip_header=True,
                                                                 fields=[('text', TEXT), ('label', LABELS)])
    overfit_data = data.TabularDataset(path='data/overfit.tsv', format='tsv', skip_header=True,
                                       fields=[('text', TEXT), ('label', LABELS)])

    TEXT.build_vocab(overfit_data, train_data, val_data, test_data)

    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab

    cnn = torch.load('model_cnn.pt')

    while (1):
        print("Enter a sentence")
        sentence = input()
        tokens = tokenizer(sentence)
        token_ints = [vocab.stoi[tok] for tok in tokens]
        token_tensor = torch.LongTensor(token_ints).view(-1, 1)  # Shape is [sentence_len, 1]
        lengths = torch.Tensor([len(token_ints)])

        prediction = cnn(token_tensor, lengths)
        # prediction = torch.sigmoid(prediction)
        print(prediction)
        if prediction > 0.6:
            print("Model CNN: positive", round(float(prediction), 3))
        elif prediction < 0.4:
            print("Model CNN: negative", round(float(prediction), 3))
        else:
        	print("Model CNN: neutral", round(float(prediction), 3))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='cnn', help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()

    main(args)

