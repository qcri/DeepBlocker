import numpy as np
import pandas as pd
import torch
import math
import torch.nn.functional as F
import time
import pdb

import sys
sys.path.append('../../em_repr/')

from models.autoencoder import *
from models.classification import *
from data.dataset import Data
from data.vocab import Vocab
from exps.exp_utils import *
import exp_options

arg_parser = exp_options.build_parser()
opts = arg_parser.parse_args()

# data_dir = '/u/h/a/hanli/research/em_repr/datasets/'
data_dir = opts.data_dir
datasets = ['Music']

cur_dataset = opts.dataset_id
word_list = '../../em_repr/em_repr/structured_data/datasets/fast_music_word_list_large.txt'
word_emb = '../../em_repr/em_repr/structured_data/datasets/fast_music_word_embed_large.npy'
ltable_path = data_dir + datasets[cur_dataset] + '/tableA_ori_tok_sample.csv'
rtable_path = data_dir + datasets[cur_dataset] + '/tableB_ori_tok_sample.csv'
gold_path = data_dir + datasets[cur_dataset] + '/gold.csv'
if opts.sif:
    word_freq_path = data_dir + datasets[cur_dataset] + '/word_freq.txt'
    word_freq_dict = Data.readWordFreq(word_freq_path)

start_time = time.time()

vocab = Vocab()
vocab.buildWordDict(word_list)
vocab.buildWordEmbeddings(word_emb)

if opts.structured:
    sel_cols = [['title', 'release', 'artist_name', 'duration', 'year']
               ]
elif opts.textual:
    sel_cols = [['description'], ['proddescrlong']]
else:
    raise Exception('Should specify the type of data.')

tableA = Data.readData([ltable_path], sel_cols[cur_dataset])
tableB = Data.readData([rtable_path], sel_cols[cur_dataset])

if opts.sif:
    tableA_emb = Data.convertToEmbeddingsBySIFForLargeDatasets(tableA, vocab, word_freq_dict, opts.sif_param)
    tableB_emb = Data.convertToEmbeddingsBySIFForLargeDatasets(tableB, vocab, word_freq_dict, opts.sif_param)
else:
    tableA_emb = Data.convertToEmbeddingsByAveraging(tableA, vocab)
    tableB_emb = Data.convertToEmbeddingsByAveraging(tableB, vocab)

if opts.model_arch == 0:
    # model_states = torch.load('/u/h/a/hanli/research/em_repr/saved_model/autoencoder/'
    #                         + datasets[cur_dataset] + '/model.bin')
    model_states = torch.load(opts.model_path)
    enc_dims = [(300, 800), (800, 600)]
    dec_dims = [(600, 800), (800, 300)]

    model = AutoEncoder(enc_dims, dec_dims, drop=0.05)
    model.load_state_dict(model_states)
    model.eval()

    tableA_tensor = torch.from_numpy(np.asarray(tableA_emb, dtype=np.float32))
    tableB_tensor = torch.from_numpy(np.asarray(tableB_emb, dtype=np.float32))
    print(tableA_tensor.shape, tableB_tensor.shape)

    tableA_enc = model.encode(tableA_tensor)
    tableB_enc = model.encode(tableB_tensor)
elif opts.model_arch == 1:
    # model_states = torch.load('/u/h/a/hanli/research/em_repr/saved_model/classification/self_teach_model_cmp_title/arch0/'
    #                     + datasets[cur_dataset] + opts.model_path)
    # model_states = torch.load('/u/h/a/hanli/research/em_repr/saved_model/arch0/'
    #                     + datasets[cur_dataset] + opts.model_path)
    model_states = torch.load(opts.model_path)
    prime_enc_dims= '[(300, 400), (400, 600)]'
    aux_enc_dims= '[(300, 400), (400, 600)]'
    cls_enc_dims= '[(600, 100), (100, 1)]'
    model = SelfTeaching(prime_enc_dims, aux_enc_dims, cls_enc_dims, drop=0.05)
    model.load_state_dict(model_states)
    model.eval()

    tableA_tensor = torch.from_numpy(np.asarray(tableA_emb, dtype=np.float32))
    tableB_tensor = torch.from_numpy(np.asarray(tableB_emb, dtype=np.float32))
    print(tableA_tensor.shape, tableB_tensor.shape)

    tableA_enc = model.encode(tableA_tensor)
    tableB_enc = model.encode(tableB_tensor)

    # tableA_enc = model.encode(tableA_tensor)
    # tableB_enc = model.encode(tableB_tensor)
elif opts.model_arch == 2:
    model_states = torch.load('../../em_repr/em_repr/saved_model/arch1/'
                        + datasets[cur_dataset] + opts.model_path)
    enc_dims = [(300, 800), (800, 600)]
    dec_dims = [(600, 800), (800, 300)]
    prime_enc_dims= '[(600, 800), (800, 600)]'
    aux_enc_dims= '[(600, 800), (800, 600)]'
    cls_enc_dims= '[(600, 100), (100, 1)]'
    # model = SelfTeaching(prime_enc_dims, aux_enc_dims, cls_enc_dims, drop=0.05)
    model = JointSelfTeaching(enc_dims, dec_dims,
                    prime_enc_dims, cls_enc_dims, drop=0.05)
    model.load_state_dict(model_states)
    model.eval()

    tableA_tensor = torch.from_numpy(np.asarray(tableA_emb, dtype=np.float32))
    tableB_tensor = torch.from_numpy(np.asarray(tableB_emb, dtype=np.float32))
    print(tableA_tensor.shape, tableB_tensor.shape)

    tableA_enc = model.encode(tableA_tensor)
    tableB_enc = model.encode(tableB_tensor)
elif opts.model_arch == 3:
    # autoenc_model_states = torch.load('/u/h/a/hanli/research/em_repr/saved_model/autoencoder/'
    #                     + datasets[cur_dataset] + opts.pretrained_autoenc_model_path)
    autoenc_model_states = torch.load(opts.pretrained_autoenc_model_path)
    autoenc_enc_dims = [(300, 800), (800, 600)]
    autoenc_dec_dims = [(600, 800), (800, 300)]

    autoenc_model = AutoEncoder(autoenc_enc_dims, autoenc_dec_dims, drop=0.05)
    autoenc_model.load_state_dict(autoenc_model_states)
    autoenc_model.eval()

    # model_states = torch.load('/u/h/a/hanli/research/em_repr/saved_model/classification/self_teach_model_cmp_title/arch2/'
    #                     + datasets[cur_dataset] + opts.model_path)
    # model_states = torch.load('/u/h/a/hanli/research/em_repr/saved_model/arch2/' + datasets[cur_dataset] + opts.model_path)
    model_states = torch.load(opts.model_path)
    prime_enc_dims= '[(600, 800), (800, 600)]'
    aux_enc_dims= '[(600, 800), (800, 600)]'
    cls_enc_dims= '[(600, 100), (100, 1)]'
    model = SelfTeaching(prime_enc_dims, aux_enc_dims, cls_enc_dims, drop=0.05)
    model.load_state_dict(model_states)
    model.eval()

    tableA_tensor = torch.from_numpy(np.asarray(tableA_emb, dtype=np.float32))
    tableB_tensor = torch.from_numpy(np.asarray(tableB_emb, dtype=np.float32))
    print(tableA_tensor.shape, tableB_tensor.shape)

    tableA_enc = model.encode(autoenc_model.encode(tableA_tensor))
    tableB_enc = model.encode(autoenc_model.encode(tableB_tensor))
elif opts.model_arch == 4:
    autoenc_model_states = torch.load('../../em_repr/em_repr/saved_model/arch3/'
                        + datasets[cur_dataset] + opts.pretrained_autoenc_model_path)
    autoenc_enc_dims = [(300, 800), (800, 600)]
    autoenc_dec_dims = [(600, 800), (800, 300)]

    autoenc_model = AutoEncoder(autoenc_enc_dims, autoenc_dec_dims, drop=0.05)
    autoenc_model.load_state_dict(autoenc_model_states)
    autoenc_model.eval()

    model_states = torch.load('../../em_repr/em_repr/saved_model/arch3/'
                        + datasets[cur_dataset] + opts.model_path)
    prime_enc_dims= '[(600, 800), (800, 600)]'
    aux_enc_dims= '[(600, 800), (800, 600)]'
    cls_enc_dims= '[(600, 100), (100, 1)]'
    model = SelfTeaching(prime_enc_dims, aux_enc_dims, cls_enc_dims, drop=0.05)
    model.load_state_dict(model_states)
    model.eval()

    tableA_tensor = torch.from_numpy(np.asarray(tableA_emb, dtype=np.float32))
    tableB_tensor = torch.from_numpy(np.asarray(tableB_emb, dtype=np.float32))
    print(tableA_tensor.shape, tableB_tensor.shape)

    tableA_enc = model.encode(autoenc_model.encode(tableA_tensor))
    tableB_enc = model.encode(autoenc_model.encode(tableB_tensor))
    # tableA_enc = autoenc_model.encode(tableA_tensor)
    # tableB_enc = autoenc_model.encode(tableB_tensor)
else:
    tableA_enc= torch.from_numpy(np.asarray(tableA_emb, dtype=np.float32))
    tableB_enc = torch.from_numpy(np.asarray(tableB_emb, dtype=np.float32))
print(tableA_enc.shape, tableB_enc.shape)

gold = pd.read_csv(gold_path)
print('Gold csv size: ', len(gold))
gold_set = getGoldset(gold)
print('Gold set size:', len(gold_set))

# gold_matrix = buildGoldMatrix(gold_set, tableA, tableB)

slice_size = opts.slice_size
num_slice_A = int(len(tableA_enc) / slice_size)
num_slice_B = int(len(tableB_enc) / slice_size)
if len(tableA_enc) % slice_size != 0:
    num_slice_A += 1
if len(tableB_enc) % slice_size != 0:
    num_slice_B += 1
print('TableA slices:', num_slice_A)
print('TableB slices:', num_slice_B)

# thres_list = [x / 10 + 0.1 for x in range(10)]
thres_list = eval(opts.cos_thres_list)
res_list = []
plot_data_list = []
block_time_list = []

preprocess_time = time.time() - start_time

for thres in thres_list:
    print('---thres', thres, '---')
    block_start_time = time.time()
    gold_total = 0
    cand_set_total = 0
    for i in range(num_slice_A):
        print('---------------Slice index:', i, '---------------')
        tableA_slice = tableA_enc[i * slice_size:(i + 1) * slice_size]
        for j in range(i, num_slice_B):
            tableB_slice = tableB_enc[j * slice_size:(j + 1) * slice_size]
            # # Approach 1
            # tableA_slice = tableA_slice.cuda()
            # tableB_slice = tableB_slice.cuda()
            # sim_matrix = calcCosineSim(tableA_slice, tableB_slice)
            # cand_matrix, cand_set_tensor = getCandMatrix(sim_matrix, thres)
            # cand_set_tensor = cand_set_tensor + torch.tensor([i * slice_size, j * slice_size]).cuda()
            # cand_set = set(map(tuple, cand_set_tensor.cpu().detach().numpy()))
            # gold_count = len(cand_set & gold_set)
            # gold_total += gold_count
            # cand_set_total += len(cand_set)
            # print('Candset size:', len(cand_set), 'Gold match:', gold_count, 'Blocktime:', time.time() - block_start_time)

            # # Approach 2
            tableA_slice = tableA_slice.cuda()
            tableB_slice = tableB_slice.cuda()
            sim_matrix = calcCosineSim(tableA_slice, tableB_slice)
            cand_matrix = getCandMatrixForLargeDataset(sim_matrix, thres)
            gold_matrix = buildGoldMatrixForLargeDataset(gold_set, slice_size, i, j)
            gold_matrix = gold_matrix.cuda()
            cand_size = torch.sum(cand_matrix).item()
            cand_set_total += cand_size
            gold_count = torch.sum(cand_matrix * gold_matrix).item()
            gold_total += gold_count
        print('Candset size:', cand_size, 'Gold match:', gold_count, 'Blocktime:', time.time() - block_start_time)

    block_end_time = time.time()
    print('Candset total:', cand_set_total, 'Gold total:', gold_total)
    print('Blocking time:', block_end_time - block_start_time)
    res_list.append(res)
    plot_data_list.append([thres, res[0], res[3]])
    block_time_list.append(block_end_time - block_start_time + preprocess_time)

for tup in res_list:
    print(tup[0], tup[1], tup[2])
for value in block_time_list:
    print(value)
for tup in plot_data_list:
    print(tup[1], tup[2])
for tup in res_list:
    print(tup[2])
