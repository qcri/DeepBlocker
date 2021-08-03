import pandas as pd
import numpy as np
from sklearn import preprocessing
from torchtext.data import Field, BucketIterator, TabularDataset
import spacy
import pdb
import math

from data.utils import *

class Data:
    @staticmethod
    def readData(table_paths, sel_cols):
        table_list = []
        for path in table_paths:
            table = pd.read_csv(path)
            table_list.append(table)
        full_table = pd.concat(table_list)

        cols = full_table.columns
        for sel_col in sel_cols:
            if sel_col not in cols:
                raise Exception('Selected column ' + '\'' + sel_col + '\'' + ' not in the schema.')

        temp = []
        for sel_col in sel_cols:
            temp.append(list(full_table[sel_col]))

        ret = []
        length = len(full_table)
        for i in range(length):
            row = []
            for j in range(len(temp)):
                row.append(temp[j][i])
            ret.append(row)
        return ret

    @staticmethod
    def readLabeledData(table_path, label_col_name):
        table = pd.read_csv(table_path)

        label_col = list(table[label_col_name])
        cols = list(table.columns)
        cols.remove(label_col_name)
        ret = []
        for col in cols:
            ret.append([[v] for v in list(table[col])])

        return ret, label_col

    @staticmethod
    def createFormattedData(table_path, label_col_name, vocab, opts):
        data_raw, label_col = Data.readLabeledData(table_path, label_col_name)

        def format_data(data, label):
            num_srcs = len(data)
            temp = []
            for i in range(num_srcs):
                if opts.sif:
                    word_freq_dict = Data.readWordFreq(opts.word_freq_file)
                    # temp.append(Data.convertToEmbeddingsBySIF(data[i], vocab, word_freq_dict, opts.sif_param))
                    if opts.large_dataset:
                        temp.append(Data.convertToEmbeddingsBySIFForLargeDatasets(data[i], vocab, word_freq_dict, opts.sif_param))
                    else:
                        temp.append(Data.convertToEmbeddingsBySIF(data[i], vocab, word_freq_dict, opts.sif_param))
                else:
                    temp.append(Data.convertToEmbeddingsByAveraging(data[i], vocab))
            temp.append(label)

            return list(zip(*temp))

        data = format_data(data_raw, label_col)
        return data


    @staticmethod
    def convertToEmbeddingsByConcat(data, vocab):
        '''
            Concat the embedding of each attribute. This may not work
            when the number of attributes is large.
        '''
        ret = []
        emb_size = vocab.size[1]
        for row in data:
            temp = []
            for attr in row:
                arr = np.zeros(emb_size)
                if attr and not pd.isnull(attr):
                    words = str(attr).strip().split(' ')
                    count = 0
                    for w in words:
                        if w:
                            arr += vocab.getEmbedding(w)
                            count += 1
                    if count > 0:
                        arr /= count
                temp.append(arr)
            ret.append(np.concatenate(temp))

        return ret

    @staticmethod
    def readWordFreq(file_path):
        word_freq_dict = {}
        with open(file_path, 'r') as fin:
            for line in fin:
                parts = line.strip().split(' ')
                if len(parts) != 2:
                    continue
                word_freq_dict[parts[0]] = float(parts[1])
        fin.close()

        return word_freq_dict


    @staticmethod
    def convertToEmbeddingsBySIF(data, vocab, word_freq_dict, sif_param):
        '''
            Take the average of all words (in all attributes) in a tuple.
        '''
        ret = []
        emb_size = vocab.size[1]
        linecount = 0
        for row in data:
            if linecount % 20000 == 0:
                print('Processing:', linecount)
            linecount += 1
            arr = np.zeros(emb_size)
            count = 0
            for attr in row:
                if attr:
                    if pd.isnull(attr):
                        continue
                    words = str(attr).strip().split(' ')
                    for w in words:
                        if w and w in word_freq_dict and vocab.containsWord(w):
                            sif_score = sif_param / (sif_param + word_freq_dict[w])
                            arr += vocab.getEmbedding(w) * sif_score
                            count += 1
            if count > 0:
                arr /= math.sqrt(count)

            ret.append(arr)

        return ret

    @staticmethod
    def convertToEmbeddingsBySIFForLargeDatasets(data, vocab, word_freq_dict, sif_param):
        '''
            Take the average of all words (in all attributes) in a tuple.
        '''
        ret = []
        emb_size = vocab.size[1]
        linecount = 0
        for row in data:
            if linecount % 20000 == 0:
                print('Processing:', linecount)
            linecount += 1
            arr = np.zeros(emb_size)
            count = 0
            for attr in row:
                if attr:
                    if pd.isnull(attr):
                        continue
                    words = str(attr).strip().split(' ')
                    for w_raw in words:
                        w = w_raw.strip()
                        if w and vocab.containsWord(w):
                            sif_score = sif_param / (sif_param + word_freq_dict[w])
                            arr += vocab.getEmbedding(w) * sif_score
                            count += 1
            if count > 0:
                arr /= math.sqrt(count)

            ret.append(arr)

        return ret


    @staticmethod
    def convertToEmbeddingsByAveraging(data, vocab):
        '''
            Take the average of all words (in all attributes) in a tuple.
        '''
        ret = []
        emb_size = vocab.size[1]
        linecount = 0
        for row in data:
            if linecount % 20000 == 0:
                print('Processing:', linecount)
            linecount += 1
            arr = np.zeros(emb_size)
            count = 0
            for attr in row:
                if attr:
                    if pd.isnull(attr):
                        continue
                    words = str(attr).strip().split(' ')
                    for w in words:
                        if w and vocab.containsWord(w):
                            arr += vocab.getEmbedding(w)
                            count += 1
            arr /= count

            ret.append(arr)

        return ret

class SequenceData:
    @staticmethod
    def ReadDataInColumns(table_path, vocab):
        table = pd.read_csv(table_path)
        schema = list(table.columns)
        fields = {}
        for attr in schema:
            fields[attr] = []
        for attr in schema:
            col_data = list(table[attr])
            for value in col_data:
                if pd.isnull(value):
                    fields[attr].append(vocab.w2i[str(value)])
                else:
                    words = value.strip().split(' ')
                    idx_list = []
                    for w in words:
                        if w in vocab.w2i:
                            idx_list.append(vocab.w2i[w])
                    fields[attr].append(idx_list)
        return fields

    @staticmethod
    def ReadDataInRows(table_path, vocab):
        table = pd.read_csv(table_path)
        ret_list = []
        linecount = 0
        for item in table.itertuples(index=False):
            linecount += 1
            ret = []
            tup = tuple(item)
            for value in tup:
                if pd.isnull(value):
                    ret.append(vocab.w2i[str(value)])
                else:
                    words = value.strip().split(' ')
                    idx_list = []
                    for w in words:
                        if w in vocab.w2i:
                            idx_list.append(vocab.w2i[w])
                    ret.append(idx_list)
            for idx_list in ret:
                if len(idx_list) < 2:
                    print(ret, linecount)
                    for temp in ret:
                        temp.append(vocab.w2i['<pad>'])
                    print(ret, linecount)
                    break

            ret_list.append(ret)
        return ret_list

    @staticmethod
    def ReadDataWithLabelInRows(table_path, vocab, label):
        table = pd.read_csv(table_path)
        schema = list(table.columns)
        assert label in schema
        label_idx = -1
        for i in range(len(schema)):
            if schema[i] == label:
                label_idx = i
        ret_list = []
        for item in table.itertuples(index=False):
            ret = []
            tup = tuple(item)
            for i in range(len(tup)):
                value = tup[i]
                if i == label_idx:
                    ret.append(value)
                else:
                    if pd.isnull(value):
                        ret.append(vocab.w2i[str(value)])
                    else:
                        words = value.strip().split(' ')
                        idx_list = []
                        for w in words:
                            if w in vocab.w2i:
                                idx_list.append(vocab.w2i[w])
                        ret.append(idx_list)
            ret_list.append(ret)
        return ret_list

    @staticmethod
    def ReverseSeqInField(field_data):
        for i in range(len(field_data)):
            field_data[i] = field_data[i][::-1]

# Implemented using torchtext.
# class Seq2seqData:
#     @staticmethod
#     def readData(path, reverse_input=True, vocab_min_freq=2):
#         spacy_tok = spacy.load('en')
#
#         def tokenize(text):
#             return [tok.text for tok in spacy_tok.tokenizer(text)]
#         def tokenize_reverse(text):
#             return [tok.text for tok in spacy_tok.tokenizer(text)][::-1]
#
#         src_field = Field(tokenize=tokenize, init_token='<sos>', eos_token='<eos>', lower=True)
#         tgt_field = Field(tokenize=tokenize, init_token='<sos>', eos_token='<eos>', lower=True)
#         if reverse_input:
#             src_field = Field(tokenize=tokenize_reverse, init_token='<sos>', eos_token='<eos>', lower=True)
#
#         data = TabularDataset(path, 'csv',
#             [('src_field', src_field), ('tgt_field', tgt_field)], skip_header=True)
#
#         src_field.build_vocab(data, min_freq=vocab_min_freq)
#         tgt_field.build_vocab(data, min_freq=vocab_min_freq)
#
#         return data, src_field, tgt_field
