import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, enc_dims=None, dec_dims=None, drop=0):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        if not enc_dims:
            raise Exception('Encoder dims must be non-empty.')
        if not dec_dims:
            raise Exception('Decoder dims must be non-empty.')

        self.assemble(self.encoder, enc_dims, drop)
        self.assemble(self.decoder, dec_dims, drop)

    def assemble(self, module, param, drop):
         order = 0
         for i in range(len(param) - 1):
             dim = param[i]
             module.add_module(str(order), nn.Linear(dim[0], dim[1]))
             order += 1
             # module.add_module(str(order), nn.ReLU())
             module.add_module(str(order), nn.Tanh())
             order += 1
             module.add_module(str(order), nn.Dropout(drop))
             order += 1
         module.add_module(str(order), nn.Linear(param[-1][0], param[-1][1]))

    def forward(self, input):
        rep = self.encoder(input)
        recover = self.decoder(rep)

        return recover

    def encode(self, input):
        enc = self.encoder(input)
        return enc

    def decode(self, input):
        res = self.decoder(input)
        return res
