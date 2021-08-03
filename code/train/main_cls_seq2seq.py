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
from models.classification import SelfTeaching

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

datasets = ['Amazon-Google', 'Walmart-Amazon', 'Abt-Buy', 'DBLP-GoogleScholar', 'DBLP-ACM']

print('[MAIN] Loading vocab.')
VOCAB = VocabFromData()
VOCAB.buildWordDict(opts.data_seq2seq, opts.min_word_freq)
VOCAB.buildWordEmbeddings(opts.word_list, opts.word_emb)
print('Vocab size:', len(VOCAB.w2i), len(VOCAB.i2w), len(VOCAB.embs))
print('Elapsed time:', time() - train_start)

print('[MAIN] Loading data.')
DATA = SequenceData.ReadDataWithLabelInRows(opts.data, VOCAB, 'label')
print('Number of training instances:', len(DATA))
print('Elapsed time:', time() - train_start)

print('[MAIN] Building model.')
if opts.model_arch == 0:
    encoder = LSTMEncoder(VOCAB.size[1], opts.rnn_hidden_size, VOCAB.size[0],
                                gpu=opts.gpu, dropout=0, n_layers=opts.rnn_layers)
    decoder = LSTMDecoderWithContext(VOCAB.size[1], opts.rnn_hidden_size, VOCAB.size[0],
                                gpu=opts.gpu, dropout=0, n_layers=opts.rnn_layers)
    seq2seq = Seq2seq(encoder, decoder, opts.gpu, True)
    seq2seq_model_states = torch.load(opts.seq2seq_model)
    seq2seq.load_state_dict(seq2seq_model_states)
    encoder.load_existing_embs(VOCAB.convertToTensor())
    decoder.load_existing_embs(VOCAB.convertToTensor())
    seq2seq.eval()
    for param in seq2seq.parameters():
        param.requires_grad = False

    model = SelfTeaching(opts.prime_enc_dims, opts.aux_enc_dims, opts.cls_enc_dims, opts.drate)
    if opts.gpu:
        seq2seq = seq2seq.cuda()
        model = model.cuda()
    print(seq2seq)
    print(model)
else:
    raise Exception('Error: model arch id not found:', opts.model_arch)

def train(model, seq2seq_model, data, vocab, opts):
    Utils.CreateDirectory(opts.model_path)
    Utils.OutputTrainingConfig(vars(opts), opts.model_path + '/config.txt')
    model.train()

    min_loss = float('inf')
    min_epoch = -1

    optimizer = optim.Adam(model.parameters(), lr=opts.lrate, weight_decay=opts.weight_decay)
    lr_decay = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opts.lrdecay)

    pos_weight = torch.tensor([opts.pos_weight])
    if opts.gpu:
        pos_weight = pos_weight.cuda()

    if opts.model_arch == 0:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    sorted_data = sorted(data, key=lambda x:len(x[0]), reverse=True)

    n_epochs = opts.num_epochs
    batch_size = opts.batch_size
    n_batches = int(len(sorted_data) / batch_size)
    if len(sorted_data) % batch_size == 0:
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
        correct = 0
        tpSum = 0
        fpSum = 0
        fnSum = 0
        for idx in batch_order:
            optimizer.zero_grad()

            cur_batch = sorted_data[batch_size * idx : batch_size * idx + batch_size]
            src_batch, tgt_batch, label_batch = zip(*cur_batch)

            padded_src_batch = Utils.PadSequence(src_batch, to_tensor=True, reverse=opts.reverse_src)
            padded_tgt_batch = Utils.PadSequence(tgt_batch, to_tensor=True, reverse=opts.reverse_src)
            label_batch_tensor = torch.from_numpy(np.asarray(label_batch, dtype=np.float32)).contiguous()
            if opts.gpu:
                padded_src_batch = padded_src_batch.cuda()
                padded_tgt_batch = padded_tgt_batch.cuda()
                label_batch_tensor = label_batch_tensor.cuda()

            src_lstm_rep = seq2seq.lstm_encode(padded_src_batch)
            tgt_lstm_rep = seq2seq.lstm_encode(padded_tgt_batch)

            src_lstm_rep = src_lstm_rep[0].squeeze()
            tgt_lstm_rep = tgt_lstm_rep[0].squeeze()

            pred_tensor = model(src_lstm_rep, tgt_lstm_rep)
            loss = criterion(pred_tensor.squeeze(), label_batch_tensor)

            loss.backward()
            optimizer.step()
            accu_loss += loss.item()

            total += batch_size
            pred_class = pred_tensor.squeeze() > 0.5
            correct += torch.sum(pred_class == label_batch_tensor.byte())
            tp = torch.dot(pred_class.float(), label_batch_tensor)
            fn = label_batch_tensor.sum() - tp
            fp = pred_class.sum().float() - tp
            tpSum += tp
            fpSum += fp
            fnSum += fn

            if opts.verbose and batch_count > 0 and batch_count % 10 == 0:
                prec_score = tpSum / (tpSum + fpSum)
                recall_score = tpSum / (tpSum + fnSum)
                print(('Batch: {batch:4d} | Loss: {loss:.4f} | Prec.: {prec:.4f} | Rec.: {recall:.4f} | Time elapsed: {time:7.2f}').format(batch=batch_count,
                    loss=accu_loss / (batch_count + 1), prec=prec_score, recall=recall_score, time=(time() - epoch_start_time)))

            batch_count += 1

        if accu_loss < min_loss:
            min_loss = accu_loss
            min_epoch = epoch
            torch.save(model.state_dict(), opts.model_path + '/model.bin')

    print('Min loss:', min_loss, 'epoch:', min_epoch)

train(model, seq2seq, DATA, VOCAB, opts)

train_end = time()
print('Total training time:', train_end - train_start)
