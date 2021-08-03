import numpy as np
import pandas as pd
import torch
import math
import torch.nn.functional as F

import sys
sys.path.append('../../em_repr/')

from lsh.lsh_blocking import *
from models.autoencoder import *
from models.classification import *
from data.dataset import Data
from data.vocab import Vocab
import exp_options
import time

arg_parser = exp_options.build_parser()
opts = arg_parser.parse_args()


data_dir = opts.data_dir
datasets = ['Amazon-Google', 'Walmart-Amazon', 'Abt-Buy',
            'DBLP-GoogleScholar', 'DBLP-ACM', 'Fodors-Zagats',
            'Music', 'Hospital']

cur_dataset = opts.dataset_id
word_list = data_dir + 'fast_word_list_small_datasets.txt'
word_emb = data_dir + 'fast_word_embed_vec_small_datasets.npy'
if opts.word_list and opts.word_emb:
    word_list = opts.word_list
    word_emb = opts.word_emb
    print(word_list)
    print(word_emb)
ltable_path = data_dir + datasets[cur_dataset] + '/tableA_ori_tok.csv'
rtable_path = data_dir + datasets[cur_dataset] + '/tableB_ori_tok.csv'
gold_path = data_dir + datasets[cur_dataset] + '/gold.csv'
if opts.sif:
    word_freq_path = data_dir + datasets[cur_dataset] + '/word_freq.txt'
    word_freq_dict = Data.readWordFreq(word_freq_path)

start_time = time.time()

vocab = Vocab()
vocab.buildWordDict(word_list)
vocab.buildWordEmbeddings(word_emb)

gold = pd.read_csv(gold_path)
print('Gold csv size:', len(gold))

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

# Average all words.
# sel_cols = [['title', 'description', 'manufacturer', 'price'], ['title', 'authors', 'venue', 'year']]
if opts.structured:
    sel_cols = [['title', 'manufacturer', 'price'],
                ['title', 'category', 'brand', 'modelno', 'price'],
                ['name', 'description', 'price'],
                ['title', 'authors', 'venue', 'year'],
                ['title', 'authors', 'venue', 'year'],
                ['name', 'addr', 'city', 'phone', 'type', 'class'],
                ['title', 'release', 'artist_name', 'duration', 'year'],
                ['name','degree','email','job','dept','location','company']
               ]
elif opts.textual:
    sel_cols = [['description'], ['proddescrlong']]
elif opts.textual_with_title:
    sel_cols = [['title', 'description'],
                ['title', 'proddescrlong']]
else:
    raise Exception('Should specify the type of data.')

print(sel_cols[cur_dataset])
tableA = Data.readData([ltable_path], sel_cols[cur_dataset])
tableB = Data.readData([rtable_path], sel_cols[cur_dataset])

if opts.sif:
    tableA_emb = Data.convertToEmbeddingsBySIF(tableA, vocab, word_freq_dict, opts.sif_param)
    tableB_emb = Data.convertToEmbeddingsBySIF(tableB, vocab, word_freq_dict, opts.sif_param)
else:
    tableA_emb = Data.convertToEmbeddingsByAveraging(tableA, vocab)
    tableB_emb = Data.convertToEmbeddingsByAveraging(tableB, vocab)

if opts.model_arch == 0:
    model_states = torch.load(opts.model_path)
    enc_dims = [(300, 800), (800, 600)]
    dec_dims = [(600, 800), (800, 300)]

    model = AutoEncoder(enc_dims, dec_dims, drop=0.05)
    model.load_state_dict(model_states)
    model.eval()

    tableA_tensor = torch.from_numpy(np.asarray(tableA_emb, dtype=np.float32))
    tableB_tensor = torch.from_numpy(np.asarray(tableB_emb, dtype=np.float32))
    print(tableA_tensor.shape, tableB_tensor.shape)

    tableA_np = model.encode(tableA_tensor).detach().numpy()
    tableB_np = model.encode(tableB_tensor).detach().numpy()
elif opts.model_arch == 1:
    model_states = torch.load('../../em_repr/saved_model/classification/'
                        + datasets[cur_dataset] + opts.model_path)
    prime_enc_dims= '[(300, 400), (400, 600)]'
    aux_enc_dims= '[(300, 400), (400, 600)]'
    cls_enc_dims= '[(600, 100), (100, 1)]'
    model = SelfTeaching(prime_enc_dims, aux_enc_dims, cls_enc_dims, drop=0.05)
    model.load_state_dict(model_states)
    model.eval()

    tableA_tensor = torch.from_numpy(np.asarray(tableA_emb, dtype=np.float32))
    tableB_tensor = torch.from_numpy(np.asarray(tableB_emb, dtype=np.float32))
    print(tableA_tensor.shape, tableB_tensor.shape)

    tableA_np = model.encode(tableA_tensor).detach().numpy()
    tableB_np = model.encode(tableB_tensor).detach().numpy()
elif opts.model_arch == 2:
    model_states = torch.load('../../em_repr/saved_model/classification/'
                        + datasets[cur_dataset] + opts.model_path)
    enc_dims = [(300, 800), (800, 600)]
    dec_dims = [(600, 800), (800, 300)]
    prime_enc_dims= '[(300, 400), (400, 600)]'
    aux_enc_dims= '[(300, 400), (400, 600)]'
    cls_enc_dims= '[(600, 100), (100, 1)]'
    # model = SelfTeaching(prime_enc_dims, aux_enc_dims, cls_enc_dims, drop=0.05)
    model = JointSelfTeaching(enc_dims, dec_dims,
                    prime_enc_dims, cls_enc_dims, drop=0.05)
    model.load_state_dict(model_states)
    model.eval()

    tableA_tensor = torch.from_numpy(np.asarray(tableA_emb, dtype=np.float32))
    tableB_tensor = torch.from_numpy(np.asarray(tableB_emb, dtype=np.float32))
    print(tableA_tensor.shape, tableB_tensor.shape)

    tableA_np = model.encode(tableA_tensor).detach().numpy()
    tableB_np = model.encode(tableB_tensor).detach().numpy()
elif opts.model_arch == 3:
    autoenc_model_states = torch.load(opts.pretrained_autoenc_model_path)
    autoenc_enc_dims = [(300, 800), (800, 600)]
    autoenc_dec_dims = [(600, 800), (800, 300)]

    autoenc_model = AutoEncoder(autoenc_enc_dims, autoenc_dec_dims, drop=0.05)
    autoenc_model.load_state_dict(autoenc_model_states)
    autoenc_model.eval()

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

    tableA_np = model.encode(autoenc_model.encode(tableA_tensor)).detach().numpy()
    tableB_np = model.encode(autoenc_model.encode(tableB_tensor)).detach().numpy()
else:
    tableA_np = np.asarray(tableA_emb, dtype=np.float32)
    tableB_np = np.asarray(tableB_emb, dtype=np.float32)
print(tableA_np.shape, tableB_np.shape)

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

preprocess_time = time.time() - start_time

def lsh_eval(K, L, id_to_vec_dict, single_table):
    build_id_to_vec_dict(index_table, id_to_vec_dict)
    print(len(id_to_vec_dict))

    lsh_engine = lsh_blocking_impl(K, L)
    index_data(lsh_engine, index_table, dimension=index_table.shape[1])
    result = compute_stats(lsh_engine, index_table, query_table,
        gold_dict, single_table, dimension=index_table.shape[1])

    return result

lsh_config = eval(opts.lsh_config)
res_list = []
block_time_list = []
for K, L in lsh_config:
    block_start = time.time()
    id_to_vec_dict.clear()
    cur_res = lsh_eval(K, L, id_to_vec_dict, opts.single_table)
    block_end = time.time()
    res_list.append(cur_res)
    block_time_list.append(block_end - block_start + preprocess_time)
    print('---K=', K, 'L=', L, '---')
    print('PC:', cur_res[0], 'RR:', cur_res[1])
for tup in res_list:
    print(tup[0], tup[1])
for value in block_time_list:
    print(value)
for tup in res_list:
    print(tup[0], 100 - tup[1])
for tup in res_list:
    print(tup[2])
