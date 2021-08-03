import numpy as np
import pandas as pd
import random
import time

from lsh import lsh

random_seeds = [778, 3527, 3647, 6917, 12539, 12684, 16778, 19300, 27564, 42453]
np.random.seed(random_seeds[0])

id_to_vec_dict = {}

def id_to_vec(id_val):
    return id_to_vec_dict[id_val]

def build_id_to_vec_dict(table_np, id_to_vec_d):
    size = table_np.shape[0]
    for i in range(size):
        id_to_vec_d[i] = table_np[i]

def create_lsh(K, L):
    #create L projects of size K each
    lsh_engine = lsh.index(float('inf'), K, L)
    return lsh_engine

# #No need to index both datasets
# #Index the smaller one and query the bigger one
# def index_data(lsh_engine, data, dataset1_size, dimension=300):
#     for i in range(dataset1_size):
#         #the following doesnt seem to work after some numpy update
#         #vector = data[i]['vec'].reshape((dimension, 1))
#         vector = data[i]['vec']
#         vector.shape = (dimension, 1)
#         id_val = data[i]['id']
#         lsh_engine.InsertIntoTable(id_val, vector)

def index_data(lsh_engine, tableA, dimension=300):
    num_tups = tableA.shape[0]
    for i in range(num_tups):
        #the following doesnt seem to work after some numpy update
        #vector = data[i]['vec'].reshape((dimension, 1))
        vector = tableA[i].reshape((dimension, 1))
        lsh_engine.InsertIntoTable(i, vector)


def query_data_non_binary_old(lsh_engine, query_vec, match_id, topK=None, multi_probe=False):
    #old code that gives a distance proxy which is the number of buckets in which the item fell into same bucket as query vector
    #if multi_probe:
    #    matches = lsh_engine.FindMP(query_vec, 2)
    #else:
    #    matches = lsh_engine.Find(query_vec)
    multi_probe_radius = 0
    if multi_probe:
        multi_probe_radius = 2
    matches = lsh_engine.FindExact(query_vec, id_to_vec, multi_probe_radius)
    if topK is None:
        tuple_ids = [elem[0] for elem in matches]
    else:
        #Old code
        #sorted_matches = sorted(matches, key=lambda x: x[1], reverse=True)
        sorted_matches = sorted(matches, key=lambda x: x[1])
        t_matches = len(sorted_matches)
        tuple_ids = [elem[0] for elem in sorted_matches[:topK]]
        # tuple_ids = [elem[0] for elem in sorted_matches]
    return match_id in tuple_ids, len(set(tuple_ids))


def query_data_non_binary(lsh_engine, query_vec, match_ids, topK=None, multi_probe=False):
    #old code that gives a distance proxy which is the number of buckets in which the item fell into same bucket as query vector
    #if multi_probe:
    #    matches = lsh_engine.FindMP(query_vec, 2)
    #else:
    #    matches = lsh_engine.Find(query_vec)
    multi_probe_radius = 0
    if multi_probe:
        multi_probe_radius = 2
    matches = lsh_engine.FindExact(query_vec, id_to_vec, multi_probe_radius)
    if topK is None:
        tuple_ids = [elem[0] for elem in matches]
    else:
        #Old code
        #sorted_matches = sorted(matches, key=lambda x: x[1], reverse=True)
        sorted_matches = sorted(matches, key=lambda x: x[1])
        t_matches = len(sorted_matches)
        tuple_ids = [elem[0] for elem in sorted_matches[:topK]]
        # tuple_ids = [elem[0] for elem in sorted_matches]
    # return match_id in tuple_ids, len(set(tuple_ids))

    match_count = 0
    for match_id in match_ids:
        if match_id in tuple_ids:
            match_count += 1
    return match_count, set(tuple_ids)


# Compute reduction ratio and pair completeness.
# The way implemented in the function for RR and PC is not correct.
# def compute_stats_old(lsh_engine, tableA, tableB, ground_truth_mapping, topK=None, multi_probe=False, dimension=300):
#     total_match = len(ground_truth_mapping)
#     found_match = 0
#     cand_size = 0
#     processed = 0
#     start = time.time()
#     for tuple_id1, tuple_id2 in ground_truth_mapping:
#         processed += 1
#         if processed % 100 == 0:
#             print(processed, time.time() - start)
#         vector2_raw = tableB[tuple_id2]
#         vector2 = vector2_raw.reshape((dimension, 1))
#         result2, block_size = query_data_non_binary_old(lsh_engine, vector2, tuple_id1, topK, multi_probe)
#         cand_size += block_size
#         if result2 is True:
#             found_match += 1
#
#     reduction_ratio = 1 - cand_size / len(tableA) / len(tableB)
#     pair_completeness = found_match / total_match
#     print('Pair Completeness:', pair_completeness)
#     print('Reduction Ratio:', reduction_ratio)
#
#     return pair_completeness, reduction_ratio


def lshCleanCandidateSet(candset):
    cand_new = set()
    for pair in candset:
        if pair[0] < pair[1]:
            cand_new.add(pair)
    return cand_new

# Compute reduction ratio and pair completeness.
def compute_stats(lsh_engine, tableA, tableB, ground_truth_mapping, single_table, topK=None,
            multi_probe=False, dimension=300):
    total_match = 0
    for key in ground_truth_mapping:
        total_match += len(ground_truth_mapping[key])
    found_match = 0
    cand_size = 0
    processed = 0
    # for tuple_id1, tuple_id2 in ground_truth_mapping:
    #     processed += 1
    #     if processed % 200 == 0:
    #         print(processed)
    #     vector2_raw = tableB[tuple_id2]
    #     vector2 = vector2_raw.reshape((dimension, 1))
    #     result2, block_size = query_data_non_binary(lsh_engine, vector2, [tuple_id1], topK, multi_probe)
    #     cand_size += block_size
    #     # if result2 is True:
    #     #     found_match += 1
    #     found_match += result2
    start = time.time()
    cand_set = set()
    for tuple_id2 in range(len(tableB)):
        processed += 1
        if processed % 200 == 0:
            print(processed, time.time() - start)
        vector2_raw = tableB[tuple_id2]
        vector2 = vector2_raw.reshape((dimension, 1))
        tuple_id1_list = []
        if tuple_id2 in ground_truth_mapping:
            tuple_id1_list = ground_truth_mapping[tuple_id2]
        result2, block_subset = query_data_non_binary(lsh_engine, vector2, tuple_id1_list, topK, multi_probe)
        cand_size += len(block_subset)
        for idx in block_subset:
            cand_set.add((tuple_id2, idx))
        found_match += result2

    if single_table:
        cand_new = lshCleanCandidateSet(cand_set)
        reduction_ratio = 1 - len(cand_new) / (len(tableA) * (len(tableA) - 1) / 2)
    else:
        reduction_ratio = 1 - cand_size / len(tableA) / len(tableB)
    pair_completeness = found_match / total_match
    f1_score = 0
    if reduction_ratio > 0 and pair_completeness > 0:
        f1_score = 2 / (1 / reduction_ratio + 1 / pair_completeness)
    print('Pair Completeness:', pair_completeness)
    print('Reduction Ratio:', reduction_ratio)

    return pair_completeness * 100, reduction_ratio * 100, f1_score * 100
