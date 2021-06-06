# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import data
from sklearn.utils import shuffle
import model
import os
# cuda device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='3'


def get_args():
    parser = argparse.ArgumentParser(description='Pretrain transformer Model')
    parser.add_argument('--data', type=str, default='./data/DNA',
                        help='location of the data corpus')
    parser.add_argument('--emsize', type=int, default=256,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=512,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=8,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=202,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=2000, metavar='N',
                    help='report interval')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')
    parser.add_argument('--nhead', type=int, default=8,
                        help='the number of heads in the encoder/decoder of the transformer model')
    return parser.parse_args()

''' Set the random seed manually for reproducibility '''
args = get_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
args.cuda = True
device = torch.device("cuda" if args.cuda else "cpu")

''' load data '''
corpus = data.Corpus(args.data)
idss = []
ids = []
words = corpus.dictionary.item()
np.save('original_K6_256_cls.npy',words)
for word in words:
    ids.append(corpus.dictionary.word2idx[word])
idss.append(torch.tensor(ids).type(torch.int64))
ids = torch.cat(idss)

''' batchify data '''
def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz,-1).t().contiguous()
    return data

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i + 1: i + seq_len + 1].view(-1)
    return data.to(device), target.to(device)

def evaluate(data_source):
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output,feature,_= model(data)
            output = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)

eval_batch_size = 128
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)

''' build the model '''
ntokens = len(corpus.dictionary)
model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout)
model.to(device)
criterion = nn.CrossEntropyLoss()
lr = 1.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output,feature,attn= model(data)
        output = output.view(-1, ntokens)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()     
        total_loss += loss.item()           
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

import time
starttime = time.time()
lr = args.lr
best_val_loss = None
''' train the model '''
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            save_p = args.save + '%s_new.pt'%epoch
            with open(save_p, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
endtime = time.time()
dtime = endtime - starttime
print("训练时间：%.8s s" % dtime)  
