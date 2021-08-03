import numpy as np
import pandas as pd
import torch

import sys
sys.path.append('../../em_repr/em_repr/')

from data.dataset import *
from data.vocab import *
from models.seq2seq import *
from exps.exp_utils import *
from utils import *

import exp_options
from time import time

arg_parser = exp_options.build_parser()
opts = arg_parser.parse_args()

# ori_data_dir = '/u/h/a/hanli/research/em_repr/datasets/'
# train_data_dir = opts.data_dir
ori_data_dir = opts.table_data_dir
train_data_dir = opts.train_data_dir
datasets = ['Amazon-Google', 'Walmart-Amazon', 'Abt-Buy', 'DBLP-GoogleScholar', 'DBLP-ACM']

cur_dataset = opts.dataset_id
word_list = ori_data_dir + 'fast_word_list_small_datasets.txt'
word_emb = ori_data_dir + 'fast_word_embed_vec_small_datasets.npy'
ltable_path = ori_data_dir + datasets[cur_dataset] + '/tableA_trunc_tok.csv'
rtable_path = ori_data_dir + datasets[cur_dataset] + '/tableB_trunc_tok.csv'
gold_path = ori_data_dir + datasets[cur_dataset] + '/gold.csv'

train_path = train_data_dir + datasets[cur_dataset] + '/train.csv'

start_time = time()
tableA = pd.read_csv(ltable_path)
tableB = pd.read_csv(rtable_path)

VOCAB = VocabFromData()
VOCAB.buildWordDict(train_path, opts.min_word_freq)
VOCAB.buildWordEmbeddings(word_list, word_emb)
print('Vocab size:', len(VOCAB.w2i), len(VOCAB.i2w), len(VOCAB.embs))
print('Elapsed time:', time() - start_time)

DATA = SequenceData.ReadDataInRows(train_path, VOCAB)
print('Number of training instances:', len(DATA))
print('Elapsed time:', time() - start_time)

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
model = Seq2seq(encoder, decoder, opts.gpu, with_context)
model_states = torch.load(opts.model_path + '/' + datasets[cur_dataset] + '/model.bin')
model.load_state_dict(model_states)
encoder.load_existing_embs(VOCAB.convertToTensor())
decoder.load_existing_embs(VOCAB.convertToTensor())

model.eval()
if opts.gpu:
    model = model.cuda()
print('Elapsed time:', time() - start_time)
print(model)

# rep_tensor = torch.zeros(len(DATA), opts.rnn_hidden_size)
# with torch.no_grad():
#     for i in range(len(DATA)):
#         if i % 1000 == 0:
#             print(i)
#         src = np.asarray(DATA[i][0][::-1])
#         src_tensor = torch.from_numpy(src).type(torch.LongTensor).unsqueeze(0)
#         if opts.gpu:
#             src_tensor = src_tensor.cuda()
#         hidden = model.encode(src_tensor)
#         # rep = (torch.sum(hidden, dim=0) / hidden.shape[0]).squeeze()
#         rep = hidden[0].squeeze()
#         rep_tensor[i] = rep

with torch.no_grad():
    rep_tensor = torch.zeros(len(DATA), opts.rnn_hidden_size)
    sorted_data = [(i, DATA[i][0]) for i in range(len(DATA))]
    sorted_data = sorted(sorted_data, key=lambda x:len(x[1]), reverse=True)
    batch_size = opts.batch_size
    n_batches = int(len(sorted_data) / batch_size)
    if len(sorted_data) % batch_size == 0:
        n_batches += 1
    for batch_idx in range(n_batches):
        if batch_idx % 100 == 0:
            print(batch_idx)

        cur_batch_tup = sorted_data[batch_size * batch_idx : batch_size * batch_idx + batch_size]
        idx_batch, cur_batch = zip(*cur_batch_tup)
        padded_src_batch = Utils.PadSequence(cur_batch, to_tensor=True, reverse=True)
        if opts.gpu:
            padded_src_batch = padded_src_batch.cuda()

        if opts.model_arch == 0 or opts.model_arch == 3:
            hidden = model.lstm_encode(padded_src_batch)
        elif opts.model_arch == 1:
            hidden = model.gru_encode(padded_src_batch)
        elif opts.model_arch == 2:
            hidden = model.bilstm_encode(padded_src_batch)
            # hidden = hidden.view(opts.rnn_layers, 2, -1, int(opts.rnn_hidden_size / 2))
            # hidden = torch.cat([hidden[:,0,:,:], hidden[:,1,:,:]], dim=2)
        else:
            assert opts.model_arch <= 3
        # rep = (torch.sum(hidden, dim=0) / hidden.shape[0]).squeeze()
        rep = hidden[0].squeeze()

        for idx in range(len(idx_batch)):
            rep_tensor[idx_batch[idx]] = rep[idx]

    gold = pd.read_csv(gold_path)
    print('Gold csv size: ', len(gold))
    gold_set = getGoldset(gold)
    print('Gold set size:', len(gold_set))
    gold_matrix = buildGoldMatrix(gold_set, tableA, tableB)

    tableA_enc = rep_tensor[:len(tableA)].contiguous()
    tableB_enc = rep_tensor[len(tableA):].contiguous()
    sim_matrix = calcCosineSim(tableA_enc, tableB_enc)

    thres_list = eval(opts.cos_thres_list)
    res_list = []
    plot_data_list = []
    block_time_list = []

    preprocess_time = time() - start_time
    for thres in thres_list:
        print('---thres', thres, '---')
        block_start_time = time()
        cand_matrix, cand_set = getCandMatrix(sim_matrix, thres)
        # print(torch.sum(cand_matrix))
        res = calcMeasures(cand_matrix, gold_matrix, tableA, tableB)
        block_end_time = time()
        res_list.append(res)
        plot_data_list.append([thres, res[0], res[3]])
        block_time_list.append(block_end_time - block_start_time + preprocess_time)

    for ii in range(len(res_list)):
        tup = res_list[ii]
        print(thres_list[ii], tup[0], tup[1], tup[2])
    for value in block_time_list:
        print(value)
    for tup in plot_data_list:
        print(tup[1], tup[2])
