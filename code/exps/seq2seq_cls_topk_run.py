import numpy as np
import pandas as pd
import torch

import sys
sys.path.append('../../em_repr/em_repr/')

from data.dataset import *
from data.vocab import *
from models.seq2seq import *
from models.classification import SelfTeaching
from exps.exp_utils import *
from utils import *

import exp_options
from time import time

arg_parser = exp_options.build_parser()
opts = arg_parser.parse_args()

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
                                gpu=opts.gpu, dropout=0, n_layers=opts.rnn_layers)
    decoder = LSTMDecoderWithContext(VOCAB.size[1], opts.rnn_hidden_size, VOCAB.size[0],
                                gpu=opts.gpu, dropout=0, n_layers=opts.rnn_layers)

    seq2seq = Seq2seq(encoder, decoder, opts.gpu, True)
    seq2seq_model_states = torch.load(opts.seq2seq_model + '/' + datasets[opts.dataset_id] + '/model.bin')
    seq2seq.load_state_dict(seq2seq_model_states)
    encoder.load_existing_embs(VOCAB.convertToTensor())
    decoder.load_existing_embs(VOCAB.convertToTensor())
    seq2seq.eval()

    model_states = torch.load(opts.cls_model + '/'+ datasets[opts.dataset_id] + '/model.bin')
    prime_enc_dims= '[(150, 800), (800, 600)]'
    aux_enc_dims= '[(150, 800), (800, 600)]'
    cls_enc_dims= '[(600, 100), (100, 1)]'
    # prime_enc_dims= '[(150, 600), (600, 1000)]'
    # aux_enc_dims= '[(150, 600), (600, 1000)]'
    # cls_enc_dims= '[(1000, 200), (200, 1)]'
    model = SelfTeaching(prime_enc_dims, aux_enc_dims, cls_enc_dims, drop=0.05)
    model.load_state_dict(model_states)
    model.eval()

    if opts.gpu:
        seq2seq = seq2seq.cuda()
        model = model.cuda()
    print(seq2seq)
    print(model)
else:
    raise Exception('Error: model arch id not found:', opts.model_arch)


with torch.no_grad():
    rep_tensor = torch.zeros(len(DATA), 600)
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

        if opts.model_arch == 0:
            hidden = seq2seq.lstm_encode(padded_src_batch)
            # hidden = hidden.view(opts.rnn_layers, 2, -1, int(opts.rnn_hidden_size / 2))
            # hidden = torch.cat([hidden[:,0,:,:], hidden[:,1,:,:]], dim=2)
        else:
            assert opts.model_arch <= 3
        # rep = (torch.sum(hidden, dim=0) / hidden.shape[0]).squeeze()
        rep_lstm = hidden[0].squeeze()
        rep = model.encode(rep_lstm)

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

    thres_list = [(i + 1) / 50 for i in range(50)]
    res_list = []
    plot_data_list = []
    block_time_list = []

    preprocess_time = time() - start_time
    for thres_percent in thres_list:
        block_start_time = time()
        print('---topk:', thres_percent, '---')
        thres = int(thres_percent * len(tableB))
        print(thres)
        if thres <= 0:
            continue
        topk_res = torch.topk(sim_matrix, thres)[1]
        cand_set = geneTopkCandSet(topk_res)
        reduction_ratio = 1 - len(cand_set) / len(tableA) / len(tableB)
        pair_completeness = len(cand_set & gold_set) / len(gold_set)
        cand_size_ratio = 1 - reduction_ratio
        print('PC:', pair_completeness, 'RR:', reduction_ratio)
        F_score = 0
        if reduction_ratio > 0 and cand_size_ratio > 0:
            F_score = 2 / (1 / reduction_ratio + 1 / pair_completeness)
        res = [pair_completeness * 100, reduction_ratio * 100, F_score * 100, cand_size_ratio * 100]
        block_end_time = time()
        res_list.append(res)
        plot_data_list.append([thres, res[0], res[3]])
        block_time_list.append(block_end_time - block_start_time + preprocess_time)

    for tup in res_list:
        print(tup[0], tup[1], tup[2])
    for value in block_time_list:
        print(value)
    for tup in plot_data_list:
        print(tup[1], tup[2])
