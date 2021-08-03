import numpy as np
import pandas as pd
import torch
import math
import torch.nn.functional as F

from lsh.lsh_blocking import *
from data.dataset import Data
from data.vocab import Vocab

data_dir = '../datasets/'
datasets = ['Amazon-Google', 'DBLP-GoogleScholar']

cur_dataset = 1
word_list = data_dir + 'fast_word_list_small_datasets.txt'
word_emb = data_dir + 'fast_word_embed_vec_small_datasets.npy'
ltable_path = data_dir + datasets[cur_dataset] + '/tableA_ori_tok.csv'
rtable_path = data_dir + datasets[cur_dataset] + '/tableB_ori_tok.csv'
gold_path = data_dir + datasets[cur_dataset] + '/gold.csv'

vocab = Vocab()
vocab.readWordDict(word_list)
vocab.readWordEmbeddings(word_emb)

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
sel_cols = [['title', 'description', 'manufacturer', 'price'], ['title', 'authors', 'venue', 'year']]
tableA = Data.readData([ltable_path], sel_cols[cur_dataset])
tableB = Data.readData([rtable_path], sel_cols[cur_dataset])
tableA_emb = Data.convertToEmbeddingsByAveraging(tableA, vocab)
tableB_emb = Data.convertToEmbeddingsByAveraging(tableB, vocab)

tableA_np = np.asarray(tableA_emb, dtype=np.float32)
tableB_np = np.asarray(tableB_emb, dtype=np.float32)

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

build_id_to_vec_dict(index_table, id_to_vec_dict)

def lsh_blocking_impl(K, L):
    lsh_engine = create_lsh(K, L)
    return lsh_engine

lsh_engine = lsh_blocking_impl(8, 2)
index_data(lsh_engine, index_table)
compute_stats_old(lsh_engine, index_table, query_table, gold_set)
