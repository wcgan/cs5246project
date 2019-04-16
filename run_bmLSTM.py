# from encoder import Model
import os
import html
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
# from utils import binary, train_with_reg_cv

def train_with_reg_cv(trX, trY, vaX, vaY, teX=None, teY=None, penalty='l1',
        C=2**np.arange(-8, 1).astype(np.float), seed=42):
    scores = []
    for i, c in enumerate(C):
        model = LogisticRegression(C=c, penalty=penalty, random_state=seed+i)
        model.fit(trX, trY)
        score = model.score(vaX, vaY)
        scores.append(score)
    c = C[np.argmax(scores)]
    model = LogisticRegression(C=c, penalty=penalty, random_state=seed+len(C))
    model.fit(trX, trY)
    nnotzero = np.sum(model.coef_ != 0)
    if teX is not None and teY is not None:
        score = model.score(teX, teY)*100.
    else:
        score = model.score(vaX, vaY)*100.
    return score, c, nnotzero

def load_data(path):
    data = pd.read_csv(path,sep='\t')
    X = data['sentence'].values.tolist()
    Y = data['label'].values
    return X, Y

def binary(data_dir='data/', tra, dev, test):
    """
    Most standard models make use of a preprocessed/tokenized/lowercased version
    of Stanford Sentiment Treebank. Our model extracts features from a version
    of the dataset using the raw text instead which we've included in the data
    folder.
    """
    trX, trY = load_data(os.path.join(data_dir, tra))
    vaX, vaY = load_data(os.path.join(data_dir, dev))
    teX, teY = load_data(os.path.join(data_dir, test))
    return trX, vaX, teX, trY, vaY, teY

def find_trainable_variables(key):
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, ".*{}.*".format(key))

def preprocess(text, front_pad='\n ', end_pad=' '):
    text = html.unescape(text)
    text = text.replace('\n', ' ').strip()
    text = front_pad+text+end_pad
    text = text.encode()
    return text

def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    batches = n // size
    if n % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])

class HParams(object):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

global nloaded
nloaded = 0

def load_params(shape, dtype, *args, **kwargs):
    global nloaded
    nloaded += 1
    return params[nloaded - 1]

def embd(X, ndim, scope='embedding'):
    with tf.variable_scope(scope):
        embd = tf.get_variable(
            "w", [hps.nvocab, ndim], initializer=load_params)
        h = tf.nn.embedding_lookup(embd, X)
        return h

def fc(x, nout, act, wn=False, bias=True, scope='fc'):
    with tf.variable_scope(scope):
        nin = x.get_shape()[-1].value
        w = tf.get_variable("w", [nin, nout], initializer=load_params)
        if wn:
            g = tf.get_variable("g", [nout], initializer=load_params)
        if wn:
            w = tf.nn.l2_normalize(w, dim=0) * g
        z = tf.matmul(x, w)
        if bias:
            b = tf.get_variable("b", [nout], initializer=load_params)
            z = z+b
        h = act(z)
        return h

def mlstm(inputs, c, h, M, ndim, scope='lstm', wn=False):
    nin = inputs[0].get_shape()[1].value
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, ndim * 4], initializer=load_params)
        wh = tf.get_variable("wh", [ndim, ndim * 4], initializer=load_params)
        wmx = tf.get_variable("wmx", [nin, ndim], initializer=load_params)
        wmh = tf.get_variable("wmh", [ndim, ndim], initializer=load_params)
        b = tf.get_variable("b", [ndim * 4], initializer=load_params)
        if wn:
            gx = tf.get_variable("gx", [ndim * 4], initializer=load_params)
            gh = tf.get_variable("gh", [ndim * 4], initializer=load_params)
            gmx = tf.get_variable("gmx", [ndim], initializer=load_params)
            gmh = tf.get_variable("gmh", [ndim], initializer=load_params)

    if wn:
        wx = tf.nn.l2_normalize(wx, dim=0) * gx
        wh = tf.nn.l2_normalize(wh, dim=0) * gh
        wmx = tf.nn.l2_normalize(wmx, dim=0) * gmx
        wmh = tf.nn.l2_normalize(wmh, dim=0) * gmh

    cs = []
    for idx, x in enumerate(inputs):
        m = tf.matmul(x, wmx)*tf.matmul(h, wmh)
        z = tf.matmul(x, wx) + tf.matmul(m, wh) + b
        i, f, o, u = tf.split(z, 4, 1)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        if M is not None:
            ct = f*c + i*u
            ht = o*tf.tanh(ct)
            m = M[:, idx, :]
            c = ct*m + c*(1-m)
            h = ht*m + h*(1-m)
        else:
            c = f*c + i*u
            h = o*tf.tanh(c)
        inputs[idx] = h
        cs.append(c)
    cs = tf.stack(cs)
    return inputs, cs, c, h

def model(X, S, M=None, reuse=False):
    nsteps = X.get_shape()[1]
    cstart, hstart = tf.unstack(S, num=hps.nstates)
    with tf.variable_scope('model', reuse=reuse):
        words = embd(X, hps.nembd)
        inputs = tf.unstack(words, nsteps, 1)
        hs, cells, cfinal, hfinal = mlstm(
            inputs, cstart, hstart, M, hps.nhidden, scope='rnn', wn=hps.rnn_wn)
        hs = tf.reshape(tf.concat(hs, 1), [-1, hps.nhidden])
        logits = fc(
            hs, hps.nvocab, act=lambda x: x, wn=hps.out_wn, scope='out')
    states = tf.stack([cfinal, hfinal], 0)
    return cells, states, logits

def ceil_round_step(n, step):
    return int(np.ceil(n/step)*step)

def batch_pad(xs, nbatch, nsteps):
    xmb = np.zeros((nbatch, nsteps), dtype=np.int32)
    mmb = np.ones((nbatch, nsteps, 1), dtype=np.float32)
    for i, x in enumerate(xs):
        l = len(x)
        npad = nsteps-l
        xmb[i, -l:] = list(x)
        mmb[i, :npad] = 0
    return xmb, mmb

class Model(object):

    def __init__(self, nbatch=128, nsteps=64):
        global hps
        hps = HParams(
            load_path='model_params/params.jl',
            nhidden=4096,
            nembd=64,
            nsteps=nsteps,
            nbatch=nbatch,
            nstates=2,
            nvocab=256,
            out_wn=False,
            rnn_wn=True,
            rnn_type='mlstm',
            embd_wn=True,
        )
        global params
        params = [np.load('model/%d.npy'%i) for i in range(15)]
        params[2] = np.concatenate(params[2:6], axis=1)
        params[3:6] = []

        X = tf.placeholder(tf.int32, [None, hps.nsteps])
        M = tf.placeholder(tf.float32, [None, hps.nsteps, 1])
        S = tf.placeholder(tf.float32, [hps.nstates, None, hps.nhidden])
        cells, states, logits = model(X, S, M, reuse=False)

        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)

        def seq_rep(xmb, mmb, smb):
            return sess.run(states, {X: xmb, M: mmb, S: smb})

        def seq_cells(xmb, mmb, smb):
            return sess.run(cells, {X: xmb, M: mmb, S: smb})

        def transform(xs):
            tstart = time.time()
            xs = [preprocess(x) for x in xs]
            lens = np.asarray([len(x) for x in xs])
            sorted_idxs = np.argsort(lens)
            unsort_idxs = np.argsort(sorted_idxs)
            sorted_xs = [xs[i] for i in sorted_idxs]
            maxlen = np.max(lens)
            offset = 0
            n = len(xs)
            smb = np.zeros((2, n, hps.nhidden), dtype=np.float32)
            for step in range(0, ceil_round_step(maxlen, nsteps), nsteps):
                start = step
                end = step+nsteps
                xsubseq = [x[start:end] for x in sorted_xs]
                ndone = sum([x == b'' for x in xsubseq])
                offset += ndone
                xsubseq = xsubseq[ndone:]
                sorted_xs = sorted_xs[ndone:]
                nsubseq = len(xsubseq)
                xmb, mmb = batch_pad(xsubseq, nsubseq, nsteps)
                for batch in range(0, nsubseq, nbatch):
                    start = batch
                    end = batch+nbatch
                    batch_smb = seq_rep(
                        xmb[start:end], mmb[start:end],
                        smb[:, offset+start:offset+end, :])
                    smb[:, offset+start:offset+end, :] = batch_smb
            features = smb[0, unsort_idxs, :]
            print('%0.3f seconds to transform %d examples' %
                  (time.time() - tstart, n))
            return features

        def cell_transform(xs, indexes=None):
            Fs = []
            xs = [preprocess(x) for x in xs]
            for xmb in tqdm(
                    iter_data(xs, size=hps.nbatch), ncols=80, leave=False,
                    total=len(xs)//hps.nbatch):
                smb = np.zeros((2, hps.nbatch, hps.nhidden))
                n = len(xmb)
                xmb, mmb = batch_pad(xmb, hps.nbatch, hps.nsteps)
                smb = sess.run(cells, {X: xmb, S: smb, M: mmb})
                smb = smb[:, :n, :]
                if indexes is not None:
                    smb = smb[:, :, indexes]
                Fs.append(smb)
            Fs = np.concatenate(Fs, axis=1).transpose(1, 0, 2)
            return Fs

        self.transform = transform
        self.cell_transform = cell_transform


model = Model(nbatch=64, nsteps=1000)

trX, vaX, teX, trY, vaY, teY = binary(tra='rt-polarity.train', dev='rt-polarity.dev', test='rt-polarity.test')
trXt = model.transform(trX)
vaXt = model.transform(vaX)
teXt = model.transform(teX)

# classification results
full_rep_acc, c, nnotzero = train_with_reg_cv(trXt, trY, vaXt, vaY, teXt, teY)
print('%05.2f test accuracy'%full_rep_acc)
print('%05.2f regularization coef'%c)
print('%05d features used'%nnotzero)