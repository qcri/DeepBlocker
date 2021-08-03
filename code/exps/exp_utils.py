import torch
import pandas as pd
import pdb

def cleanCandidateSet(candset):
    cand_new = set()
    for pair in candset:
        if pair[0] < pair[1]:
            cand_new.add(pair)
    return cand_new

def buildGoldMatrix(gold_set, tableA, tableB):
    gold_matrix = torch.zeros([len(tableA), len(tableB)], dtype=torch.uint8)
    for tup in gold_set:
        gold_matrix[tup[0]][tup[1]] = 1
    return gold_matrix

def buildGoldMatrixForLargeDataset(gold_set, slice_size, i, j):
    gold_matrix = torch.zeros([slice_size, slice_size], dtype=torch.uint8)
    A_lower, A_upper = i * slice_size, (i + 1) * slice_size
    B_lower, B_upper = j * slice_size, (j + 1) * slice_size
    for tup in gold_set:
        if tup[0] >= A_lower and tup[0] < A_upper and tup[1] >= B_lower and tup[1] < B_upper:
            gold_matrix[tup[0] - i * slice_size][tup[1] - j * slice_size] = 1
    return gold_matrix

def getGoldset(gold_csv):
    gold_set = set()
    for tup in gold_csv.itertuples(index=False):
        gold_set.add(tuple(tup))
    return gold_set

def calcCosineSim(tableA_tensor, tableB_tensor):
    C = torch.mm(tableA_tensor, tableB_tensor.t())
    A_norm = torch.norm(tableA_tensor, dim=1).unsqueeze(dim=1).expand_as(C)
    B_norm = torch.norm(tableB_tensor, dim=1).unsqueeze(dim=0).expand_as(C)
    D = C / A_norm / B_norm

    return D.contiguous()


def getCandMatrix(sim_matrix, thres):
    res = sim_matrix > thres
    cand_set = res.nonzero()
    return res, cand_set

def getCandMatrixForLargeDataset(sim_matrix, thres):
    res = sim_matrix > thres
    return res

def geneTopkCandSet(topk_res):
    cand_set = set()
    for i in range(len(topk_res)):
        for idx in topk_res[i]:
            cand_set.add((i, idx.item()))
    return cand_set

def geneTopkCandSetTensor(topk_res, topk, tableA_size):
    topk_res_temp = topk_res.view(-1, topk, 1)
    A_matrix = torch.tensor(list(range(tableA_size))).cuda()
    length = A_matrix.shape[0]
    A_matrix = A_matrix.view(-1, length, 1).expand(-1, length, topk).view(-1, topk, 1)
    cand_matrix = torch.cat([A_matrix, topk_res_temp], dim=2).view(-1, 2)
    cand_set = set(map(tuple, cand_matrix.cpu().detach().numpy()))

    return cand_set


def geneTopkCandSetForLargeDataset(topk_res, topk, slice_size_A, A_shift):
    shifted_topk_res = topk_res + A_shift * slice_size_A
    A_matrix = torch.tensor(list(range(A_shift * slice_size_A, (A_shift + 1) * slice_size_A))).cuda()
    length = A_matrix.shape[0]
    topk_res_temp = shifted_topk_res.view(-1, topk, 1)
    A_matrix = A_matrix.view(-1, length, 1).expand(-1, length, topk).view(-1, topk, 1)
    cand_matrix = torch.cat([A_matrix, topk_res_temp], dim=2).view(-1, 2)
    cand_set = set(map(tuple, cand_matrix.cpu().detach().numpy()))
    return cand_set

    # for i in range(len(topk_res)):
    #     for idx in topk_res[i]:
    #         cand_set.add((i + A_shift * slice_size, idx.item() + B_shift * slice_size))
    # return cand_set


def calcMeasures(cand_matrix, gold_matrix, tableA, tableB):
    reduction_ratio = 1 - torch.sum(cand_matrix).item() / len(tableA) / len(tableB)
    cand_size_ratio = 1 - reduction_ratio
    pair_completeness = torch.sum(cand_matrix * gold_matrix).item() / torch.sum(gold_matrix).item()
    F_score = 0
    if reduction_ratio > 0 and pair_completeness > 0:
        F_score = 2 / (1 / pair_completeness + 1 / reduction_ratio)
    print('Reduction Ratio:', reduction_ratio)
    print('Pair Completeness:', pair_completeness)
    # print(pair_completeness * 100, reduction_ratio * 100)
    return pair_completeness * 100, reduction_ratio * 100, F_score * 100, cand_size_ratio * 100


def outputCandset(candset, output_path):
    cand_list = list(candset)
    id_list = list(range(len(candset)))
    cand_table = pd.DataFrame(cand_list)
    cand_table.insert(0, '_id', id_list)
    cand_table.columns = ['_id', 'ltable_id', 'rtable_id']
    cand_table.to_csv(output_path, index=False)


def outputCandList(candset, output_path):
    cand_list = candset
    id_list = list(range(len(candset)))
    cand_table = pd.DataFrame(cand_list)
    cand_table.insert(0, '_id', id_list)
    cand_table.columns = ['_id', 'ltable_id', 'rtable_id']
    cand_table.to_csv(output_path, index=False)
