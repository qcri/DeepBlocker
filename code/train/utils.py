import torch
import numpy as np
import os
from torchtext.data.utils import interleave_keys

class Utils:
    @staticmethod
    def CreateDirectory(path):
        if os.path.exists(path):
            response = input('Model path already exists: ' + path + '\nDo you want to remove the model? (y/n)\n')
            if response.startswith('y'):
                os.system('rm -rf ' + path)
                os.makedirs(path)
            else:
                exit()
        else:
            os.makedirs(path)

    @staticmethod
    def OutputTrainingConfig(config, path):
        with open(path, 'w') as fin:
            for key in config:
                fin.write(str(key) + ':' + str(config[key]) + '\n')
        fin.close()

    @staticmethod
    def PadSequence(batch, to_tensor=True, reverse=False):
        max_len = max([len(x) for x in batch])
        padded_batch = np.zeros([len(batch), max_len])

        for i in range(len(batch)):
            cur_len = len(batch[i])
            padded_batch[i][:cur_len] = batch[i]
            if reverse:
                padded_batch[i] = padded_batch[i][::-1]

        if to_tensor:
            return torch.from_numpy(padded_batch).type(torch.LongTensor)
        else:
            return padded_batch
