import numpy as np
import pandas as pd
import torch

import sys
sys.path.append('../../em_repr/em_repr/')

from data.dataset import *
from data.vocab import *
from models.seq2seq import *
from lsh.lsh_blocking import *
from utils import *

import exp_options
from time import time

def getGoldset(gold_csv, reverse=False):
    gold_set = set()
    for tup in gold_csv.itertuples(index=False):
        idx_tuple = tuple(tup)
        if reverse:
            gold_set.add((idx_tuple[1], idx_tuple[0]))
        else:
            gold_set.add(idx_tuple)
    return gold_set

def buildGoldMap(gold_csv, reverse=True):
    gold_dict = {}
    for (left, right) in gold_csv.itertuples(index=False):
        if reverse:
            if right not in gold_dict:
                gold_dict[right] = []
            gold_dict[right].append(left)
        else:
            if left not in gold_dict:
                gold_dict[left] = []
            gold_dict[left].append(right)
    return gold_dict

arg_parser = exp_options.build_parser()
opts = arg_parser.parse_args()

ori_data_dir = '../../em_repr/datasets/'
train_data_dir = opts.data_dir
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
gold = pd.read_csv(gold_path)

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

tableA_np = rep_tensor[:len(tableA)].contiguous().detach().numpy()
tableB_np = rep_tensor[len(tableA):].contiguous().detach().numpy()

query_table = tableA_np
index_table = tableB_np
reverse_order_set = True
reverse_order_map = not reverse_order_set

gold_set = getGoldset(gold, reverse=reverse_order_set)
print('Gold set size:', len(gold_set))

gold_dict = buildGoldMap(gold, reverse=reverse_order_map)
print('Unique key in gold map:', len(gold_dict))
total_pair = 0
for key in gold_dict:
    total_pair += len(gold_dict[key])
print('Total pairs in gold map:', total_pair)

def lsh_blocking_impl(K, L):
    lsh_engine = create_lsh(K, L)
    return lsh_engine

preprocess_time = time() - start_time

def lsh_eval(K, L, id_to_vec_dict):
    build_id_to_vec_dict(index_table, id_to_vec_dict)
    print(len(id_to_vec_dict))

    lsh_engine = lsh_blocking_impl(K, L)
    index_data(lsh_engine, index_table, dimension=index_table.shape[1])
    result = compute_stats(lsh_engine, index_table, query_table, gold_dict, dimension=index_table.shape[1])

    return result

lsh_config = eval(opts.lsh_config)
res_list = []
block_time_list = []
for K, L in lsh_config:
    block_start = time()
    id_to_vec_dict.clear()
    cur_res = lsh_eval(K, L, id_to_vec_dict)
    block_end = time()
    res_list.append(cur_res)
    block_time_list.append(block_end - block_start + preprocess_time)
    print('---K=', K, 'L=', L, '---')
    print('PC:', cur_res[0], 'RR:', cur_res[1])
for tup in res_list:
    print(tup[0], tup[1], 2 * tup[0] * tup[1] / (tup[0] + tup[1]))
for value in block_time_list:
    print(value)
for tup in res_list:
    print(tup[0], 100 - tup[1])
