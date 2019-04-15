import argparse
import os
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from tqdm import tqdm


def load_embeddings(emb_file, emb_dim = 300):

  with open(emb_file, 'rt', encoding='utf-8') as f:
    word2idx = {}
    emb_weights = {}

    # Process the remaining lines
    #line[0] contains the word while line[1:] contains the vector strings
    for line in tqdm(f):
      line = line.strip().split()
      if line:
        dim = len(line) - 1
        if dim == emb_dim:
          word = line[0]
          rep = [float(x) for x in line[1:]]
          emb_weights[word] = rep
          word2idx[word] = len(word2idx)

  emb = np.zeros((len(emb_weights)+1, emb_dim))
  for word, idx in word2idx.items():
    emb[idx] = emb_weights[word]

  emb_weights = emb    

  # Create a random vector for 'unk' tokens
  mean = emb_weights[1:-1].mean(0)
  std = emb_weights[1:-1].std(0)
  rand = np.random.RandomState(5246)
  vec = rand.normal(mean, std)
  emb_weights[-1] = vec

  # Convert numpy array to PyTorch tensor
  emb_weights = torch.FloatTensor(emb_weights)
  
  word2idx['<UNK>'] = len(word2idx)

  return emb_weights, word2idx


def process_input(inp_file, word2idx):

  with open(inp_file, errors='replace') as f:
    texts = f.readlines()

  texts_id = []
  labels = []
  for line in texts[1:]:
    try:
      text, label = line.strip().split('\t')
    except:
      print(line)
      raise ValueError('Error reading line')
    labels.append(int(label))
    text = text.strip().split()
    tokens_id = []
    for token in text:
      # Try working with only lowercase inputs
      # Should give marginal improvements
      token = token.lower()
      if token in word2idx:
        tokens_id.append(word2idx[token])
      else:
        tokens_id.append(word2idx['<UNK>'])
    texts_id.append(tokens_id)

  return texts_id, labels


class Classifier(nn.Module):

  def __init__(self, emb_weights, hidden_size, num_classes, dropout):

    super(Classifier, self).__init__()
    #self.token_embedder = token_embedder
    #self.char_embedder = char_embedder
    #self.emb_dim = token_embedder.emb_dim + char_embedder.num_filters
    self.emb = nn.Embedding.from_pretrained(emb_weights)
    self.emb_dim = emb_weights.shape[1]
    self.hidden_size = hidden_size
    self.output_size = num_classes

    self.bilstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_size,
      batch_first=True, bidirectional=True)
    self.out = nn.Linear(2*self.hidden_size, num_classes)
    if dropout:
      self.dropout = nn.Dropout(dropout)
    else:
      self.dropout = lambda nop: nop

    for param in self.emb.parameters():
      param.requires_grad = False

  def forward(self, tokens, lengths):

    emb_token = self.emb(tokens)    
    lstm_out = self.dropout(self._pack_unpack_lstm(emb_token, lengths, self.bilstm)) # B x 2H
    #lstm_out = self._pack_unpack_lstm(emb_token, lengths, self.bilstm)
    #print(lstm_out.shape)
    out = self.out(lstm_out)

    return out


  #@classmethod
  def _pack_unpack_lstm(self, input, lengths, lstm):
    sorted_len, idx = lengths.sort(0, descending=True)
    sorted_input = input.index_select(0, idx)
    _, reverse_idx = idx.sort(0, descending=False)

    packed = nn.utils.rnn.pack_padded_sequence(
      sorted_input, sorted_len.data.tolist(), batch_first=True)
    _, (output, _) = lstm(packed)
    output = output.transpose(1,0).contiguous().view(-1,2*self.hidden_size)

    output = output.index_select(0, reverse_idx)
    
    return output


class EpochGen(object):

  def __init__(self, data, batch_size=32, shuffle=True, tensor_type=torch.LongTensor):
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.data_word = data[0]
    self.data_tag = data[1]
    self.n_samples = len(data[0])
    self.idx = np.arange(self.n_samples)
    self.tensor_type = tensor_type

  
  def process_input_for_length(self, sequences):
    """
    Assemble and pad data.
    """
    lengths = Variable(torch.tensor([len(seq) for seq in sequences]))
    max_length = max(len(seq) for seq in sequences)

    def _padded(seq, max_length):
      _padded_seq = self.tensor_type(max_length).zero_()
      _padded_seq[:len(seq)] = self.tensor_type(seq)
      return _padded_seq
    sequences = Variable(torch.stack(
        [_padded(seq, max_length) for seq in sequences]))

    return (sequences, lengths)

  
  def __len__(self):
    
    return (self.n_samples + self.batch_size - 1)//self.batch_size


  def __iter__(self):
    """
    Generate batches from data.
    All outputs are in torch.autograd.Variable, for convenience.
    """
    if self.shuffle:
      np.random.shuffle(self.idx)

    for start_ind in range(0, self.n_samples - 1, self.batch_size):
      batch_idx = self.idx[start_ind:start_ind+self.batch_size]
      word = [self.data_word[idx] for idx in batch_idx]
      label = [self.data_tag[idx] for idx in batch_idx]
      inp = self.process_input_for_length(word)
      label = torch.tensor(label)

      batch = (inp, label)
      yield batch

    return


def train_model(args):
    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file

    print("Loading word vectors")
    if args.load_emb_file:
      with open(args.load_emb_file, 'rb') as f:
        emb_file = torch.load(f)
        emb_weights = emb_file['emb_weights']
        word2idx = emb_file['word2idx']
    elif args.emb_file_txt:
      emb_weights, word2idx = load_embeddings(args.emb_file_txt)
    else:
      raise ValueError("At least one of 'load_emb_file' or 'emb_weights' must be provided.")
    
    print("Preparing training data")
    data = process_input(args.train_file, word2idx)
    data = EpochGen(data)

    print("Preparing validation data")
    val_data = process_input(args.val_file, word2idx)
    val_data = EpochGen(val_data)

    print("Preparing model")
    model = Classifier(emb_weights, hidden_size=300, num_classes=2, dropout=0.3)
    criterion = nn.CrossEntropyLoss()
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_params, lr=0.001, betas=(0.9, 0.999),
      eps=1e-8, weight_decay=0)

    # loop over epochs
    for i in range(args.epochs):
      print('Epoch: %1d / %1d'%(i+1, args.epochs))
      # train model
      correct = 0
      total = 0
      model.train()
      model.cuda()
      for inp, labels in tqdm(data):
        tokens, lengths = inp
        tokens = tokens.cuda()
        lengths = lengths.cuda()
        labels = labels.cuda().view(-1)
        outputs = model(tokens, lengths)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.shape[0]
        correct += (predicted == labels).sum().item()

      print('Training Accuracy:', correct/total)

      # perform validation
      correct = 0
      total = 0
      model.eval()
      for inp, labels in val_data:
        tokens, lengths = inp
        tokens = tokens.cuda()
        lengths = lengths.cuda()
        labels = labels.cuda().view(-1)
        outputs = model(tokens, lengths)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.shape[0]
        correct += (predicted == labels).sum().item()

      print('Validation Accuracy:', correct/total)

    if args.output_file:
      # save model
      items_to_save = {
        'word2idx': word2idx,
        'emb_weights_dim': emb_weights.shape,
        'model_state_dict': model.state_dict(),
      }

      with open(args.output_file, 'wb') as f:
        torch.save(items_to_save, f)

    print('Finished...')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--train_file", type=str)
  parser.add_argument("--val_file", type=str)
  parser.add_argument("--load_emb_file", type=str)
  parser.add_argument("--emb_file_txt", type=str)
  parser.add_argument("--output_file", type=str)
  parser.add_argument("--epochs", type=int)

  args = parser.parse_args()

  train_model(args)


if __name__ == "__main__":
    main()
