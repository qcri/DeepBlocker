import torch
from torch import nn
from models.autoencoder import AutoEncoder
import pdb

class Encoder(nn.Module):
    def __init__(self, enc_dims=None, drop=0):
        super(Encoder, self).__init__()
        enc_dims = eval(enc_dims)
        if not isinstance(enc_dims, list):
            raise Exception('The encoder dims should be in the format of a list.')
        self.encoder = nn.Sequential()
        if not enc_dims:
            raise Exception('Encoder dims must be non-empty.')
        self.assemble(self.encoder, enc_dims, drop)

    def assemble(self, module, param, drop):
         order = 0
         for i in range(len(param) - 1):
             dim = param[i]
             module.add_module(str(order), nn.Linear(dim[0], dim[1]))
             order += 1
             module.add_module(str(order), nn.ReLU())
             # module.add_module(str(order), nn.Tanh())
             order += 1
             module.add_module(str(order), nn.Dropout(drop))
             order += 1
         module.add_module(str(order), nn.Linear(param[-1][0], param[-1][1]))

    def forward(self, input):
        rep = self.encoder(input)

        return rep


class HighWayLayer(nn.Module):
    def __init__(self, inSize, outSize, dropout):
        super(HighWayLayer, self).__init__()

        self.linear = nn.Linear(inSize, outSize)
        self.gate = nn.Linear(inSize, outSize)
        # self.bn = nn.BatchNorm1d(outSize)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # linear = self.bn(self.linear(self.drop(x)))
        linear = self.linear(self.drop(x))
        activation = nn.ReLU(linear)
        # gate = F.sigmoid(self.bn(self.gate(self.drop(x))))
        gate = F.sigmoid(self.gate(self.drop(x)))

        return gate * activation + (1 - gate) * x


class SelfTeaching(nn.Module):
    '''
        Self teach the model by a classification task to determine if an attribute value
        is from a tuple. The first part is an autoencoder.
    '''
    def __init__(self, prime_enc_dims=None, aux_enc_dims=None, cls_enc_dims=None, drop=0):
        super(SelfTeaching, self).__init__()
        self.prime_enc = Encoder(prime_enc_dims, drop)
        # self.aux_enc = Encoder(aux_enc_dims, drop)
        self.classifier = Encoder(cls_enc_dims, drop)

    def forward(self, prime_input, aux_input):
        prime_res = self.prime_enc(prime_input)
        # aux_res = self.aux_enc(aux_input)
        aux_res = self.prime_enc(aux_input)
        # cls_input = torch.cat([prime_res, aux_res], dim=1)
        cls_input = torch.abs(prime_res - aux_res)
        output = self.classifier(cls_input)

        return output

    def encode(self, input):
        prime_res = self.prime_enc(input)

        return prime_res

class JointSelfTeaching(nn.Module):
    '''
        Jointly self teach the model by minimizing the classification task error and the
        autoencoder error.
    '''
    def __init__(self, autoenc_enc_dims=None, autoenc_dec_dims=None,
                    cls_prime_enc_dims=None, cls_dims=None, drop=0):
        super(JointSelfTeaching, self).__init__()
        self.autoenc = AutoEncoder(autoenc_enc_dims, autoenc_dec_dims, drop)
        self.classifier = SelfTeaching(cls_prime_enc_dims, cls_prime_enc_dims, cls_dims, drop)

    def forward(self, prime_input, aux_input):
        autoenc_prime = self.autoenc.encode(prime_input)
        autoenc_aux = self.autoenc.encode(aux_input)
        autoenc_prime_res = self.autoenc.decode(autoenc_prime)
        autoenc_aux_res = self.autoenc.decode(autoenc_aux)
        cls_res = self.classifier(autoenc_prime, autoenc_aux)

        return autoenc_prime_res, autoenc_aux_res, cls_res

    def encode(self, prime_input):
        enc = self.autoenc.encode(prime_input)
        res = self.classifier.encode(enc)
        return res
        # return enc
