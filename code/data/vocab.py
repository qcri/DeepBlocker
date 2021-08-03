import numpy as np
import pandas as pd
import torch
import pdb

class Vocab:
    def __init__(self):
        self.embs = None
        self.dict = None
        self.size = None

    def buildWordEmbeddings(self, path):
        self.embs = np.load(path)
        self.size = self.embs.shape
        return


    def buildWordDict(self, path):
        self.dict = {}
        with open(path, 'r') as fin:
            linecount = 0
            for line in fin:
                self.dict[line.strip()] = linecount
                linecount += 1
        fin.close()
        return

    def containsWord(self, word):
        return word in self.dict

    def getIndex(self, word):
        return self.dict[word]

    def getEmbedding(self, word):
        return self.embs[self.dict[word]]

    def convertToTensor(self):
        return torch.from_numpy(self.embs).type(torch.FloatTensor)


class VocabFromData(Vocab):
    def __init__(self):
        super(VocabFromData, self).__init__()
        self.w2i = {}
        self.i2w = {}

    def buildWordEmbeddings(self, complete_word_list, complete_word_emb):
        complete_embs = np.load(complete_word_emb)
        complete_dict = {}
        with open(complete_word_list, 'r') as fin:
            linecount = 0
            for line in fin:
                complete_dict[line.strip()] = linecount
                linecount += 1
        fin.close()

        index_list = []
        for i in range(1, len(self.i2w)):
            index_list.append(complete_dict[self.i2w[i]])
        self.embs = complete_embs[index_list]
        self.embs = np.insert(self.embs, 0, [0 for i in range(self.embs.shape[1])], axis=0)
        self.size = self.embs.shape

    def buildWordDict(self, table_path, word_list_path, min_freq):
        self.i2w = {}
        self.w2i = {}

        def insertWord(word, index):
            self.i2w[index] = word
            self.w2i[word] = index

        insertWord('<pad>', 0)
        word_freq_dict = {}
        table = pd.read_csv(table_path)
        for col in table:
            col_data = table[col]
            for value in col_data:
                if pd.isnull(value):
                    str_value = str(value)
                    if str_value not in word_freq_dict:
                        word_freq_dict[str_value] = 0
                    word_freq_dict[str_value] += 1
                else:
                    words = value.strip().split(' ')
                    for w in words:
                        if w not in word_freq_dict:
                            word_freq_dict[w] = 0
                        word_freq_dict[w] += 1

        word_set = set()
        with open(word_list_path, 'r') as fin:
            for line in fin:
                word_set.add(line.strip())
        fin.close()

        word_list = []
        for word in word_freq_dict:
            if word in word_set and word_freq_dict[word] >= min_freq:
                word_list.append(word)

        word_list = sorted(word_list)
        for word in word_list:
            insertWord(word, len(self.w2i))
