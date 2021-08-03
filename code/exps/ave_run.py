import numpy as np
import pandas as pd
import torch
import math
import torch.nn.functional as F
import time

import sys
sys.path.append('../../em_repr/em_repr/')

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
datasets = ['Amazon-Google', 'Walmart-Amazon', 'Abt-Buy',
            'DBLP-GoogleScholar', 'DBLP-ACM', 'Fodors-Zagats', 'Hospital']

cur_dataset = opts.dataset_id
word_list = '../../em_repr/structured_data/datasets/fast_word_list_small_datasets.txt'
word_emb = '../../em_repr/structured_data/datasets/fast_word_embed_vec_small_datasets.npy'
if opts.word_list and opts.word_emb:
    word_list = opts.word_list
    word_emb = opts.word_emb
    print(word_list)
    print(word_emb)
# if opts.structured:
#     ltable_path = data_dir + datasets[cur_dataset] + '/tableA_ori_tok.csv'
#     rtable_path = data_dir + datasets[cur_dataset] + '/tableB_ori_tok.csv'
# elif opts.textual:
#     ltable_path = data_dir + datasets[cur_dataset] + '/tableA_top20_tok.csv'
#     rtable_path = data_dir + datasets[cur_dataset] + '/tableB_top20_tok.csv'
# gold_path = data_dir + datasets[cur_dataset] + '/gold.csv'
ltable_path = data_dir + datasets[cur_dataset] + '/tableA_ori_tok.csv'
rtable_path = data_dir + datasets[cur_dataset] + '/tableB_ori_tok.csv'
gold_path = data_dir + datasets[cur_dataset] + '/gold.csv'
if opts.sif:
    word_freq_path = '../../em_repr/structured_data/datasets/' \
            + datasets[cur_dataset] + '/word_freq.txt'
    word_freq_dict = Data.readWordFreq(word_freq_path)

start_time = time.time()

vocab = Vocab()
vocab.buildWordDict(word_list)
vocab.buildWordEmbeddings(word_emb)

if opts.structured:
    sel_cols = [['title', 'manufacturer', 'price'],
                ['title', 'category', 'brand', 'modelno', 'price'],
                ['name', 'description', 'price'],
                ['title', 'authors', 'venue', 'year'],
                ['title', 'authors', 'venue', 'year'],
                ['name', 'addr', 'city', 'phone', 'type', 'class'],
                ['name','degree','email','job','dept','location','company']
               ]
    # sel_cols = [['title'],
    #             ['title'],
    #             ['name'],
    #             ['title'],
    #             ['title']
    #            ]
elif opts.textual:
    sel_cols = [['description'], ['proddescrlong']]
elif opts.textual_with_title:
    sel_cols = [['title', 'description'],
                ['title', 'proddescrlong']]
else:
    raise Exception('Should specify the type of data.')

tableA = Data.readData([ltable_path], sel_cols[cur_dataset])
tableB = Data.readData([rtable_path], sel_cols[cur_dataset])

if opts.sif:
    tableA_emb = Data.convertToEmbeddingsBySIF(tableA, vocab, word_freq_dict, opts.sif_param)
    tableB_emb = Data.convertToEmbeddingsBySIF(tableB, vocab, word_freq_dict, opts.sif_param)
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
    model_states = torch.load('../../em_repr/saved_model/arch1/'
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
    autoenc_model_states = torch.load('../../em_repr/saved_model/arch3/'
                        + datasets[cur_dataset] + opts.pretrained_autoenc_model_path)
    autoenc_enc_dims = [(300, 800), (800, 600)]
    autoenc_dec_dims = [(600, 800), (800, 300)]

    autoenc_model = AutoEncoder(autoenc_enc_dims, autoenc_dec_dims, drop=0.05)
    autoenc_model.load_state_dict(autoenc_model_states)
    autoenc_model.eval()

    model_states = torch.load('../../em_repr/saved_model/arch3/'
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

gold_matrix = buildGoldMatrix(gold_set, tableA, tableB)
sim_matrix = calcCosineSim(tableA_enc, tableB_enc)

# thres_list = eval(opts.cos_thres_list)
thres_list = [1 - (i + 1) / 50 for i in range(50 - 1)]
res_list = []
plot_data_list = []
block_time_list = []

preprocess_time = time.time() - start_time

for thres in thres_list:
    print('---thres', thres, '---')
    block_start_time = time.time()
    cand_matrix, cand_set = getCandMatrix(sim_matrix, thres)
    res = calcMeasures(cand_matrix, gold_matrix, tableA, tableB)
    cand_set = set(map(tuple, cand_set.numpy()))
    print('Cand set size:', len(cand_set))
    if opts.single_table:
        cand_set = cleanCandidateSet(cand_set)
        print('candset:', len(cand_set))
    if opts.single_table:
        reduction_ratio = 1 - len(cand_set) / (len(tableA) * (len(tableA) - 1) / 2)
    else:
        reduction_ratio = 1 - len(cand_set) / len(tableA) / len(tableB)
    pair_completeness = len(cand_set & gold_set) / len(gold_set)

    cand_size_ratio = 1 - reduction_ratio
    F_score = 0
    if reduction_ratio > 0 and cand_size_ratio > 0:
        F_score = 2 / (1 / reduction_ratio + 1 / pair_completeness)
    res = [pair_completeness * 100, reduction_ratio * 100, F_score * 100, cand_size_ratio * 100]
    print('PC:', pair_completeness, 'RR:', reduction_ratio, 'F1:', F_score)
    block_end_time = time.time()
    res_list.append(res)
    plot_data_list.append([thres, res[0], res[3]])
    block_time_list.append(block_end_time - block_start_time + preprocess_time)

    if pair_completeness == 1.0:
        break

for tup in res_list:
    print(tup[0], tup[1], tup[2])
for value in block_time_list:
    print(value)
for tup in plot_data_list:
    print(tup[1], tup[2])
for tup in res_list:
    print(tup[2])
