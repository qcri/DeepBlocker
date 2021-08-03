import torch
import torch.optim as optim
from torch import nn
import options
from utils import *
import random
import os

from data.dataset import *
from data.vocab import *
from utils import Utils
from models.seq2seq import *

import pdb
from time import time

train_start = time()

arg_parser = options.build_parser()
opts = arg_parser.parse_args()

if opts.deterministic:
    SEED = 123456
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

datasets = ['Amazon-Google', 'Walmart-Amazon', 'Abt-Buy',
            'DBLP-GoogleScholar', 'DBLP-ACM', 'Fodors-Zagats']

print('[MAIN] Loading vocab.')
VOCAB = VocabFromData()
VOCAB.buildWordDict(opts.data, opts.word_list, opts.min_word_freq)
VOCAB.buildWordEmbeddings(opts.word_list, opts.word_emb)
print('Vocab size:', len(VOCAB.w2i), len(VOCAB.i2w), len(VOCAB.embs))
print('Elapsed time:', time() - train_start)

print('[MAIN] Loading data.')
DATA = SequenceData.ReadDataInRows(opts.data, VOCAB)
print('Number of training instances:', len(DATA))
print('Elapsed time:', time() - train_start)

print('[MAIN] Building model.')
if opts.model_arch == 0:
    encoder = LSTMEncoder(VOCAB.size[1], opts.rnn_hidden_size, VOCAB.size[0],
                                gpu=opts.gpu, dropout=opts.drate, n_layers=opts.rnn_layers)
    decoder = LSTMDecoder(VOCAB.size[1], opts.rnn_hidden_size, VOCAB.size[0],
                                gpu=opts.gpu, dropout=opts.drate, n_layers=opts.rnn_layers)
elif opts.model_arch == 1:
    encoder = GRUEncoder(VOCAB.size[1], opts.rnn_hidden_size, VOCAB.size[0],
                                gpu=opts.gpu, dropout=opts.drate, n_layers=opts.rnn_layers)
    decoder = GRUDecoder(VOCAB.size[1], opts.rnn_hidden_size, VOCAB.size[0],
                                gpu=opts.gpu, dropout=opts.drate, n_layers=opts.rnn_layers)
elif opts.model_arch == 2:
    encoder = BiLSTMEncoder(VOCAB.size[1], opts.rnn_hidden_size, VOCAB.size[0],
                                gpu=opts.gpu, dropout=opts.drate, n_layers=opts.rnn_layers)
    decoder = LSTMDecoder(VOCAB.size[1], opts.rnn_hidden_size, VOCAB.size[0],
                                gpu=opts.gpu, dropout=opts.drate, n_layers=opts.rnn_layers)
elif opts.model_arch == 3:
    encoder = LSTMEncoder(VOCAB.size[1], opts.rnn_hidden_size, VOCAB.size[0],
                                gpu=opts.gpu, dropout=opts.drate, n_layers=opts.rnn_layers)
    decoder = LSTMDecoderWithContext(VOCAB.size[1], opts.rnn_hidden_size, VOCAB.size[0],
                                gpu=opts.gpu, dropout=opts.drate, n_layers=opts.rnn_layers)
else:
    raise Exception('Error: model arch id not found:', opts.model_arch)

with_context = opts.model_arch == 3
encoder.load_existing_embs(VOCAB.convertToTensor())
decoder.load_existing_embs(VOCAB.convertToTensor())
model = Seq2seq(encoder, decoder, opts.gpu, with_context)
if opts.gpu:
    model = model.cuda()
print('Elapsed time:', time() - train_start)
print(model)

def train(model, data, vocab, opts):
    Utils.CreateDirectory(opts.model_path)
    Utils.OutputTrainingConfig(vars(opts), opts.model_path + '/config.txt')
    model.train()

    min_loss = float('inf')
    min_epoch = -1

    optimizer = optim.Adam(model.parameters(), lr=opts.lrate, weight_decay=opts.weight_decay)
    lr_decay = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opts.lrdecay)

    pad_idx = vocab.w2i['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    sorted_data = sorted(data, key=lambda x:len(x[0]), reverse=True)

    n_epochs = opts.num_epochs
    batch_size = opts.batch_size
    n_batches = int(len(sorted_data) / batch_size)
    if len(sorted_data) % batch_size != 0:
        n_batches += 1
    batch_order = [i for i in range(n_batches)]

    for epoch in range(n_epochs):
        print('[MAIN] Start epoch:', epoch)
        optimizer.zero_grad()
        lr_decay.step()

        random.shuffle(batch_order)

        epoch_start_time = time()
        accu_loss = 0
        total = 0
        batch_count = 0
        for idx in batch_order:
            optimizer.zero_grad()

            cur_batch = sorted_data[batch_size * idx : batch_size * idx + batch_size]
            src_batch, tgt_batch = zip(*cur_batch)

            padded_src_batch = Utils.PadSequence(src_batch, to_tensor=True, reverse=opts.reverse_src)
            padded_tgt_batch = Utils.PadSequence(tgt_batch, to_tensor=True)
            if opts.gpu:
                padded_src_batch = padded_src_batch.cuda()
                padded_tgt_batch = padded_tgt_batch.cuda()

            outputs = model(padded_src_batch, padded_tgt_batch, opts.teacher_forcing_ratio)

            outputs_for_loss = outputs[:,1:,:].contiguous().view(-1, outputs.shape[-1])
            tgt_for_loss = padded_tgt_batch[:,1:].contiguous().view(-1)
            loss = criterion(outputs_for_loss, tgt_for_loss)

            loss.backward()
            optimizer.step()

            accu_loss += loss.item()
            if opts.verbose and batch_count > 0 and batch_count % 10 == 0:
                print(('Batch: {batch:4d} | Loss: {loss:.4f} | Time elapsed: {time:7.2f}').format(batch=batch_count,
                    loss=accu_loss / (batch_count + 1), time=(time() - epoch_start_time)))
            batch_count += 1

        if accu_loss < min_loss:
            min_loss = accu_loss
            min_epoch = epoch
            torch.save(model.state_dict(), opts.model_path + '/model.bin')
        print('Elapsed Time:', time() - train_start)

    print('Min loss:', min_loss, 'epoch:', min_epoch)

train(model, DATA, VOCAB, opts)

train_end = time()
print('Total training time:', train_end - train_start)
