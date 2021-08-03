import numpy as np
import pandas as pd
import torch
import math
import torch.nn.functional as F
import time
from sklearn.decomposition import PCA

import sys
sys.path.append('../../em_repr/em_repr/')

from models.autoencoder import *
from models.classification import *
from data.dataset import Data
from data.vocab import Vocab
from data.utils import *
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
    # sel_cols = [['title'],
    #             ['title'],
    #             ['name'],
    #             ['title'],
    #             ['title'],
    #             ['name']
    #            ]
    sel_cols = [['title', 'manufacturer', 'price'],
                ['title', 'category', 'brand', 'modelno', 'price'],
                ['name', 'description', 'price'],
                ['title', 'authors', 'venue', 'year'],
                ['title', 'authors', 'venue', 'year'],
                ['name', 'addr', 'city', 'phone', 'type', 'class'],
                ['name','degree','email','job','dept','location','company']
               ]
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
    tableA_emb = np.asarray(tableA_emb, dtype=np.float32)
    tableB_emb = np.asarray(tableB_emb, dtype=np.float32)
    if opts.rm_pc:
        tableA_emb = remove_pc(tableA_emb, 1)
        tableB_emb = remove_pc(tableB_emb, 1)
else:
    tableA_emb = Data.convertToEmbeddingsByAveraging(tableA, vocab)
    tableB_emb = Data.convertToEmbeddingsByAveraging(tableB, vocab)
    tableA_emb = np.asarray(tableA_emb, dtype=np.float32)
    tableB_emb = np.asarray(tableB_emb, dtype=np.float32)

if opts.model_arch == 0:
    # model_states = torch.load('/u/h/a/hanli/research/em_repr/saved_model/autoencoder/'
    #                         + datasets[cur_dataset] + '/model.bin')
    model_states = torch.load(opts.model_path)
    enc_dims = [(300, 800), (800, 600)]
    dec_dims = [(600, 800), (800, 300)]

    model = AutoEncoder(enc_dims, dec_dims, drop=0.05)
    model.load_state_dict(model_states)
    model.eval()

    # tableA_tensor = torch.from_numpy(np.asarray(tableA_emb, dtype=np.float32))
    # tableB_tensor = torch.from_numpy(np.asarray(tableB_emb, dtype=np.float32))
    tableA_tensor = torch.from_numpy(tableA_emb)
    tableB_tensor = torch.from_numpy(tableB_emb)
    print(tableA_tensor.shape, tableB_tensor.shape)

    tableA_enc = model.encode(tableA_tensor)
    tableB_enc = model.encode(tableB_tensor)
elif opts.model_arch == 1:
    # model_states = torch.load('/u/h/a/hanli/research/em_repr/saved_model/classification/self_teach_model_cmp_title/arch0/'
    #                     + datasets[cur_dataset] + opts.model_path)
    # model_states = torch.load('/u/h/a/hanli/research/em_repr/saved_model/arch0/'
    #                     + datasets[cur_dataset] + opts.model_path)
    model_states = torch.load(opts.model_path)
    prime_enc_dims = '[(300, 400), (400, 600)]'
    aux_enc_dims= '[(300, 400), (400, 600)]'
    cls_enc_dims= '[(600, 100), (100, 1)]'
    # prime_enc_dims = eval(opts.prime_enc_dims)
    # aux_enc_dims = eval(opts.aux_enc_dims)
    # cls_enc_dims = eval(opts.cls_enc_dims)
    model = SelfTeaching(prime_enc_dims, aux_enc_dims, cls_enc_dims, drop=0.05)
    model.load_state_dict(model_states)
    model.eval()

    tableA_tensor = torch.from_numpy(tableA_emb)
    tableB_tensor = torch.from_numpy(tableB_emb)
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

    tableA_tensor = torch.from_numpy(tableA_emb)
    tableB_tensor = torch.from_numpy(tableB_emb)
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

    tableA_tensor = torch.from_numpy(tableA_emb)
    tableB_tensor = torch.from_numpy(tableB_emb)
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

    tableA_tensor = torch.from_numpy(tableA_emb)
    tableB_tensor = torch.from_numpy(tableB_emb)
    print(tableA_tensor.shape, tableB_tensor.shape)

    tableA_enc = model.encode(autoenc_model.encode(tableA_tensor))
    tableB_enc = model.encode(autoenc_model.encode(tableB_tensor))
    # tableA_enc = autoenc_model.encode(tableA_tensor)
    # tableB_enc = autoenc_model.encode(tableB_tensor)
elif opts.model_arch == 5:
    # model_states = torch.load('/u/h/a/hanli/research/em_repr/saved_model/classification/self_teach_model_cmp_title/arch0/'
    #                     + datasets[cur_dataset] + opts.model_path)
    # model_states = torch.load('/u/h/a/hanli/research/em_repr/saved_model/arch0/'
    #                     + datasets[cur_dataset] + opts.model_path)
    model_states = torch.load(opts.model_path)
    prime_enc_dims = '[(300, 400), (400, 600)]'
    aux_enc_dims= '[(300, 400), (400, 600)]'
    cls_enc_dims= '[(600, 100), (100, 3)]'
    # prime_enc_dims = eval(opts.prime_enc_dims)
    # aux_enc_dims = eval(opts.aux_enc_dims)
    # cls_enc_dims = eval(opts.cls_enc_dims)
    model = SelfTeaching(prime_enc_dims, aux_enc_dims, cls_enc_dims, drop=0.05)
    model.load_state_dict(model_states)
    model.eval()

    tableA_tensor = torch.from_numpy(tableA_emb)
    tableB_tensor = torch.from_numpy(tableB_emb)
    print(tableA_tensor.shape, tableB_tensor.shape)

    tableA_enc = model.encode(tableA_tensor)
    tableB_enc = model.encode(tableB_tensor)
elif opts.model_arch == 6:
    pca = PCA(opts.pca_dim)
    # tableA_numpy = np.asarray(tableA_emb, dtype=np.float32)
    # tableB_numpy = np.asarray(tableB_emb, dtype=np.float32)
    table_numpy = np.concatenate((tableA_emb, tableB_emb), axis=0)
    table_enc = pca.fit_transform(table_numpy)
    tableA_enc = torch.from_numpy(table_enc[:len(tableA)])
    tableB_enc = torch.from_numpy(table_enc[len(tableA):])
else:
    tableA_enc = torch.from_numpy(tableA_emb)
    tableB_enc = torch.from_numpy(tableB_emb)
print(tableA_enc.shape, tableB_enc.shape)

gold = pd.read_csv(gold_path)
print('Gold csv size: ', len(gold))
gold_set = getGoldset(gold)
print('Gold set size:', len(gold_set))

# gold_matrix = buildGoldMatrix(gold_set, tableA, tableB)
sim_matrix = calcCosineSim(tableA_enc, tableB_enc).cuda()
print(sim_matrix)

# thres_list = [x / 10 + 0.1 for x in range(10)]
thres_list = eval(opts.topk_config)
res_list = []
plot_data_list = []
block_time_list = []

preprocess_time = time.time() - start_time
print('preprocessing times:', preprocess_time)

# thres_list = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.015]
thres_list = [(i + 1) / 50 for i in range(50 - 1)]
# thres_list = [int((i + 1) * 50) for i in range(20)]
# thres_list = [2, 6, 11, 21, 51, 101, 151, 201, 251, 301, 351, 401, 451, 501]
for thres_percent in thres_list:
    print('---topk:', thres_percent, '---')
    thres = int(thres_percent * len(tableB))
    # thres = thres_percent
    print(thres)
    if thres <= 0:
        continue
    if thres > len(tableB):
        break

    block_start_time = time.time()
    topk_res = torch.topk(sim_matrix, thres)[1]
    # cand_set = geneTopkCandSet(topk_res)
    cand_set = geneTopkCandSetTensor(topk_res, thres, len(tableA))
    print(len(cand_set))
    if opts.single_table:
        cand_set = cleanCandidateSet(cand_set)
        print('candset:', len(cand_set))
    if opts.output_candset:
        outputCandset(cand_set, opts.output_candset_path + '/candset_' + str(thres_percent) + '.csv')

    if opts.single_table:
        reduction_ratio = 1 - len(cand_set) / (len(tableA) * (len(tableA) - 1) / 2)
    else:
        reduction_ratio = 1 - len(cand_set) / len(tableA) / len(tableB)
    pair_completeness = len(cand_set & gold_set) / len(gold_set)

    cand_size_ratio = 1 - reduction_ratio
    F_score = 0
    if reduction_ratio > 0 and pair_completeness > 0:
        F_score = 2 / (1 / reduction_ratio + 1 / pair_completeness)
    print('PC:', pair_completeness, 'RR:', reduction_ratio, 'F1:', F_score)
    res = [pair_completeness * 100, reduction_ratio * 100, F_score * 100, cand_size_ratio * 100]
    block_end_time = time.time()
    print('blocking time:', block_end_time - block_start_time + preprocess_time)
    res_list.append(res)
    plot_data_list.append([thres, res[0], res[3]])
    block_time_list.append(block_end_time - block_start_time + preprocess_time)

    if pair_completeness == 1.0:
        break
    # if thres > 1000:
        # break

for tup in res_list:
    print(tup[0], tup[1], tup[2])
for tup in plot_data_list:
    print(tup[1], tup[2])
for tup in res_list:
    print(tup[2])
print(preprocess_time)
for value in block_time_list:
    print(value)
