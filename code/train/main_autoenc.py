import torch
import torch.optim as optim
from torch import nn
import options
from utils import *
import random

from data.dataset import *
from data.vocab import *
from data.utils import *
from utils import Utils
from models.autoencoder import AutoEncoder

import pdb
from time import time

train_start = time()

# Parse argument
arg_parser = options.build_parser()
opts = arg_parser.parse_args()

print('[MAIN] Loading vocab.')
VOCAB = Vocab()
VOCAB.buildWordDict(opts.word_list)
VOCAB.buildWordEmbeddings(opts.word_emb)

print('[MAIN] Loading data.')
if opts.single_table:
    DATA = Data.readData([opts.data + 'table.csv'], eval(opts.sel_cols))
else:
    tlists = [opts.data + 'tableA_ori_tok.csv', opts.data + 'tableB_ori_tok.csv']
    DATA = Data.readData(tlists, eval(opts.sel_cols))

if opts.model_arch == 0:
    enc_dims = eval(opts.encoder_dims)
    dec_dims = eval(opts.decoder_dims)
    if not isinstance(enc_dims, list):
        raise Exception('The encoder dims should be in the format of a list.')
    if not isinstance(dec_dims, list):
        raise Exception('The decoder dims should be in the format of a list.')
    model = AutoEncoder(enc_dims, dec_dims, opts.drate)

if opts.gpu:
    model = model.cuda()
print(model)

def train(model, data_raw, vocab, opts):
    Utils.CreateDirectory(opts.model_path)
    model.train()

    min_loss = float('inf')
    min_epoch = -1

    optimizer = optim.Adam(model.parameters(), lr=opts.lrate, weight_decay=opts.weight_decay)
    lr_decay = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opts.lrdecay)
    criterion = nn.MSELoss()
    n_epochs = opts.num_epochs
    if opts.model_arch == 0:
        if opts.concat:
            data = Data.convertToEmbeddingsByConcat(data_raw, vocab)
        elif opts.sif:
            word_freq_dict = Data.readWordFreq(opts.data + '/word_freq.txt')
            if opts.large_dataset:
                data = Data.convertToEmbeddingsBySIFForLargeDatasets(data_raw, vocab, word_freq_dict, opts.sif_param)
            else:
                data = Data.convertToEmbeddingsBySIF(data_raw, vocab, word_freq_dict, opts.sif_param)
            if opts.rm_pc:
                print('Calc SIF first pc')
                data_numpy = np.asarray(data, dtype=np.float32)
                first_pc = compute_pc(data_numpy, 1)
        else:
            data = Data.convertToEmbeddingsByAveraging(data_raw, vocab)
    train_order = [i for i in range(len(data))]
    print(len(train_order))

    for epoch in range(n_epochs):
        print('[MAIN] Start epoch:', epoch)
        optimizer.zero_grad()
        lr_decay.step()

        random.shuffle(train_order)
        batch_size = opts.batch_size
        batch_cnt = int(len(train_order) / batch_size)
        if len(train_order) % batch_size != 0:
            batch_cnt += 1

        start_time = time()
        accu_loss = 0
        for i in range(batch_cnt):
            optimizer.zero_grad()
            cur_batch_idx = train_order[i * batch_size : (i + 1) * batch_size]
            cur_batch = []
            for idx in cur_batch_idx:
                cur_batch.append(data[idx])
            if opts.rm_pc:
                cur_batch_numpy = np.asarray(cur_batch, dtype=np.float32)
                cur_batch_numpy_rm_pc = cur_batch_numpy - cur_batch_numpy.dot(first_pc.transpose()) * first_pc
                cur_tensor_batch = torch.from_numpy(cur_batch_numpy_rm_pc)
            else:
                cur_tensor_batch = torch.from_numpy(np.asarray(cur_batch, dtype=np.float32))
            if opts.denoise:
                bernoulli_prob_matrix = torch.zeros(cur_tensor_batch.size()) + opts.noise_rate
                noise_mask = torch.bernoulli(bernoulli_prob_matrix)
                noise_tensor_batch = cur_tensor_batch * (1 - noise_mask)
                # gassian_noise = torch.randn(cur_tensor_batch.size()) * 0.0
                # noise_tensor_batch = cur_tensor_batch + gassian_noise
            if opts.gpu:
                cur_tensor_batch = cur_tensor_batch.cuda()
                if opts.denoise:
                    noise_tensor_batch = noise_tensor_batch.cuda()

            if opts.denoise:
                recover_res = model(noise_tensor_batch)
            else:
                recover_res = model(cur_tensor_batch)
            # pdb.set_trace()
            loss = criterion(recover_res, cur_tensor_batch)
            loss.backward()
            optimizer.step()
            accu_loss += loss.item()

            if opts.verbose and i > 0 and i % 10 == 0:
                print(('Batch: {batch:4d} | Loss: {loss:.4f} | Time elapsed: {time:7.2f}').format(batch=i, loss=accu_loss / (i + 1), time=(time() - start_time)))

        if accu_loss < min_loss:
            min_loss = accu_loss
            min_epoch = epoch
            torch.save(model.state_dict(), opts.model_path + '/model.bin')

    print('Min loss:', min_loss, 'epoch:', min_epoch)

if opts.train:
    train(model, DATA, VOCAB, opts)

train_end = time()
print('Total training time:', train_end - train_start)
