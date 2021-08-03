import torch
import torch.nn as nn
import pandas as pd
import random
import pdb

class RNNEncoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, vocab_size, n_layers=2, dropout=0.1, gpu=False):
        super(RNNEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.gpu = gpu

        self.embedding_layer = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

    def load_existing_embs(self, existing_embs, requires_grad=False):
        self.embedding_layer.weight.data = existing_embs
        self.embedding_layer.weight.requires_grad = requires_grad


class LSTMEncoder(RNNEncoder):
    def __init__(self, emb_dim, hid_dim, vocab_size, n_layers=2, dropout=0.1, gpu=False):
        super(LSTMEncoder, self).__init__(emb_dim, hid_dim, vocab_size, n_layers, dropout, gpu)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)

    def forward(self, input):
        input_emb = self.embedding_layer(input)
        outputs, (hidden, cell) = self.rnn(input_emb)

        return hidden, cell


class BiLSTMEncoder(RNNEncoder):
    def __init__(self, emb_dim, hid_dim, vocab_size, n_layers=2, dropout=0.1, gpu=False):
        super(BiLSTMEncoder, self).__init__(emb_dim, hid_dim, vocab_size, n_layers, dropout, gpu)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.hidden_linear_trans = nn.Linear(hid_dim * 2, hid_dim)
        self.cell_linear_trans = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, input):
        input_emb = self.embedding_layer(input)
        outputs, (hidden, cell) = self.rnn(input_emb)

        hidden = hidden.view(self.n_layers, 2, -1, self.hid_dim)
        cell = cell.view(self.n_layers, 2, -1, self.hid_dim)
        hidden_cat = torch.cat([hidden[:,0,:,:], hidden[:,1,:,:]], dim=2)
        cell_cat = torch.cat([cell[:,0,:,:], cell[:,1,:,:]], dim=2)
        hidden_cat = hidden_cat.view(-1, hidden_cat.shape[-1])
        cell_cat = cell_cat.view(-1, cell_cat.shape[-1])

        hidden_final = torch.tanh(self.hidden_linear_trans(hidden_cat))
        cell_final = torch.tanh(self.cell_linear_trans(cell_cat))
        hidden_final = hidden_final.view(-1, hidden_final.shape[0], hidden_final.shape[1])
        cell_final = cell_final.view(-1, cell_final.shape[0], cell_final.shape[1])

        return hidden_final, cell_final


class GRUEncoder(RNNEncoder):
    def __init__(self, emb_dim, hid_dim, vocab_size, n_layers=2, dropout=0.1, gpu=False):
        super(GRUEncoder, self).__init__(emb_dim, hid_dim, vocab_size, n_layers, dropout, gpu)

        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)

    def forward(self, input):
        input_emb = self.embedding_layer(input)
        outputs, hidden = self.rnn(input_emb)

        return hidden, None

class RNNDecoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, vocab_size, n_layers=2, dropout=0.1, gpu=False):
        super(RNNDecoder, self).__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.gpu = gpu

        self.embedding_layer = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.out = nn.Linear(hid_dim, vocab_size)

    def load_existing_embs(self, existing_embs, requires_grad=False):
        self.embedding_layer.weight.data = existing_embs
        self.embedding_layer.weight.requires_grad = requires_grad


class LSTMDecoder(RNNDecoder):
    def __init__(self, emb_dim, hid_dim, vocab_size, n_layers=2, dropout=0.1, gpu=False):
        super(LSTMDecoder, self).__init__(emb_dim, hid_dim, vocab_size, n_layers, dropout, gpu)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embs = self.embedding_layer(input)

        output, (hidden, cell) = self.rnn(embs, (hidden, cell))
        prediction = self.out(output.squeeze())

        return prediction, hidden, cell

class LSTMDecoderWithContext(RNNDecoder):
    def __init__(self, emb_dim, hid_dim, vocab_size, n_layers=2, dropout=0.1, gpu=False):
        super(LSTMDecoderWithContext, self).__init__(emb_dim, hid_dim, vocab_size, n_layers, dropout, gpu)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)

    def forward(self, input, hidden, cell, context):
        input = input.unsqueeze(1)
        embs = self.embedding_layer(input)
        emb_context = torch.cat([context, embs], dim=2)

        output, (hidden, cell) = self.rnn(emb_context, (hidden, cell))
        prediction = self.out(output.squeeze())

        return prediction, hidden, cell


class GRUDecoder(RNNDecoder):
    def __init__(self, emb_dim, hid_dim, vocab_size, n_layers=2, dropout=0.1, gpu=False):
        super(GRUDecoder, self).__init__(emb_dim, hid_dim, vocab_size, n_layers, dropout, gpu)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)

    def forward(self, input, hidden, dummy_cell):
        input = input.unsqueeze(1)
        embs = self.embedding_layer(input)

        output, hidden = self.rnn(embs, hidden)
        prediction = self.out(output.squeeze())

        return prediction, hidden, None

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, gpu, with_context=False):
        super(Seq2seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.gpu = gpu
        self.with_context = with_context

    def forward(self, src, tgt, teacher_forcing_ratio=0.8):
        batch_size = tgt.shape[0]
        max_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.vocab_size

        hidden, cell = self.encoder(src)
        outputs = torch.zeros(batch_size, max_len, tgt_vocab_size)
        if self.gpu:
            outputs = outputs.cuda()

        input = tgt[:,0]
        context = hidden.squeeze()
        context = context.view(context.shape[0], -1, context.shape[1])

        for t in range(1, max_len):
            if self.with_context:
                output, hidden, cell = self.decoder(input, hidden, cell, context)
            else:
                output, hidden, cell = self.decoder(input, hidden, cell)

            outputs[:,t,:] = output
            teacher_force = random.random() < teacher_forcing_ratio

            if teacher_force:
                input = tgt[:,t]
            else:
                input = output.max(1)[1]

        return outputs

    def lstm_encode(self, input):
        input_emb = self.encoder.embedding_layer(input)
        outputs, (hidden, cell) = self.encoder.rnn(input_emb)

        return hidden

    def bilstm_encode(self, input):
        input_emb = self.encoder.embedding_layer(input)
        outputs, (hidden, cell) = self.encoder.rnn(input_emb)

        hidden = hidden.view(self.encoder.n_layers, 2, -1, self.encoder.hid_dim)
        hidden_cat = torch.cat([hidden[:,0,:,:], hidden[:,1,:,:]], dim=2)
        hidden_cat = hidden_cat.view(-1, hidden_cat.shape[-1])

        hidden_final = torch.tanh(self.encoder.hidden_linear_trans(hidden_cat))
        hidden_final = hidden_final.view(-1, hidden_final.shape[0], hidden_final.shape[1])

        return hidden_final

    def gru_encode(self, input):
        input_emb = self.encoder.embedding_layer(input)
        outputs, hidden = self.encoder.rnn(input_emb)

        return hidden
