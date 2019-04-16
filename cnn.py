# python cnn.py train
# reference: https://github.com/Shawn1993/cnn-text-classification-pytorch

import os
import math
import sys
import gzip
import spacy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from torchtext import data

EARLY_STOP = 1000
EVAL_INTERVAL = 100
SAVE_INTERVAL = 500
EPOCH = 256

class CNN_Text(nn.Module):

    def __init__(self, vocab, emb_dim, n_filters=100, filter_sizes=None, class_num=2, dropout=0.5):
        super(CNN_Text, self).__init__()

        c_in = 1
        c_out = n_filters
        Ks = filter_sizes

        self.embed = nn.Embedding(len(vocab), emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        #self.convs1 = nn.ModuleList([nn.Conv2d(c_in, c_out, (K, emb_dim)) for K in Ks])
        self.convs1 = nn.ModuleList([nn.Conv2d(c_in, c_out, (K, emb_dim), padding=(K-1,0)) for K in Ks])

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks)*c_out, class_num)

    def forward(self, x):
        x = self.embed(x)

        x = x.unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]

        x = [F.max_pool1d(i, i.shape[2]).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        x = self.dropout(x)
        y = self.fc1(x)

        return y

def evaluate(model, data_iter):
    
    #model.cuda()
    model.eval()

    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.sentence, batch.label

        feature.data.t_()
        logit = model(feature)
        loss = F.cross_entropy(logit, target)

        avg_loss += loss.data
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects.item()/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy

def train_m(train_iter, model):
   # model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    steps=0
    best_acc = 0
    last_step=0
    model.train()
    for epoch in range(1, EPOCH + 1):
        for batch in train_iter:
            inputs, target = batch.sentence, batch.label

            inputs.data.t_()

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = F.cross_entropy(outputs, target)
            loss.backward()
            optimizer.step()

            steps += 1

            corrects = (torch.max(outputs, 1)[1].data == target.data).sum()
            accuracy = 100.0 * corrects.item()/len(inputs)
            print('\rBatch[{}] - loss: {:.6f} acc: {:.4f}%({}/{})'.format(steps, loss.data.item(), accuracy, corrects, len(inputs)))

            if steps%EVAL_INTERVAL == 0:
                dev_acc = evaluate(model, dev_iter)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    torch.save(model.state_dict(), 'model.pt')
                else:
                    if steps - last_step >= EARLY_STOP:
                        print('early stop by {} steps.'.format(EARLY_STOP))
                        return
            elif steps % SAVE_INTERVAL == 0:
                torch.save(model.state_dict(), 'model.pt')


if __name__ == "__main__":

    mode = sys.argv[1]

    Ks = [3, 4, 5]

    comment = data.Field(
                    sequential=True,
                    tokenize='spacy',
                    lower=True)
    label = data.Field(
                    sequential=False, use_vocab=False)
    train = data.TabularDataset(
                            path='data/rt-polarity.train', format='tsv',skip_header=True,
                            fields=[('sentence', comment), ('label', label)])
    dev = data.TabularDataset(
                            path='data/rt-polarity.dev', format='tsv',skip_header=True,
                            fields=[('sentence', comment), ('label', label)])
    test = data.TabularDataset(
                            path='data/rt-polarity.test', format='tsv',skip_header=True,
                            fields=[('sentence', comment), ('label', label)])
    comment.build_vocab(train, vectors="glove.6B.300d")

    train_iter, dev_iter, test_iter = data.Iterator.splits(
            (train, dev, test), sort_key=lambda x: len(x.sentence),
            batch_sizes=(32, 256, 256))

    vocab = comment.vocab
    emb_dim = vocab.vectors.shape[1]

    # build model
    cnn = CNN_Text(vocab, emb_dim=emb_dim, filter_sizes=Ks)

    if mode == "train":  
        train_m(train_iter, cnn)
    else:
        cnn.load_state_dict(torch.load("model.pt"))
        evaluate(cnn,test_iter)

    print('Finished...')

