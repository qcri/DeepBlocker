import torch
import torch.optim as optim
from torch import nn
import options
from utils import *
import random
import os

from data.dataset import *
from data.vocab import *
from utils import Utils
from models.classification import SelfTeaching
from models.classification import JointSelfTeaching
from models.autoencoder import AutoEncoder

import pdb
from time import time

train_start = time()

arg_parser = options.build_parser()
opts = arg_parser.parse_args()
datasets = ['Amazon-Google', 'Walmart-Amazon', 'Abt-Buy',
            'DBLP-GoogleScholar', 'DBLP-ACM', 'Music']

print('[MAIN] Loading vocab.')
VOCAB = Vocab()
VOCAB.buildWordDict(opts.word_list)
VOCAB.buildWordEmbeddings(opts.word_emb)

print('[MAIN] Loading data.')
if opts.model_arch <= 5:
    saved_train_path = opts.data + '/train_model_arch_0.npy'
    saved_valid_path = opts.data + '/valid_model_arch_0.npy'
    # if os.path.exists(saved_train_path):
    #     TRAIN = np.load(saved_train_path)
    # else:
    #     TRAIN = Data.createFormattedData(opts.data + '/train.csv', 'label', VOCAB)
    #     np.save(saved_train_path, TRAIN)
    if opts.large_dataset:
        TRAIN = Data.createFormattedData(opts.data, 'label', VOCAB, opts)
    else:
        TRAIN = Data.createFormattedData(opts.data + '/train.csv', 'label', VOCAB, opts)

    # if os.path.exists(saved_valid_path):
    #     VALID = np.load(saved_valid_path)
    # else:
    #     VALID = Data.createFormattedData(opts.data + '/valid.csv', 'label', VOCAB)
    #     np.save(saved_valid_path, VALID)
else:
    raise Exception('Error: model arch id not found:', opts.model_arch)

if opts.model_arch == 0 or opts.model_arch == 4:
    model = SelfTeaching(opts.prime_enc_dims, opts.aux_enc_dims, opts.cls_enc_dims, opts.drate)
elif opts.model_arch == 1:
    model = JointSelfTeaching(eval(opts.encoder_dims), eval(opts.decoder_dims),
                    opts.prime_enc_dims, opts.cls_enc_dims, opts.drate)
elif opts.model_arch == 2 or opts.model_arch == 5:
    model = SelfTeaching(opts.prime_enc_dims, opts.aux_enc_dims, opts.cls_enc_dims, opts.drate)
    autoenc_model_states = torch.load(opts.autoenc_model)
    autoenc_enc_dims = [(300, 800), (800, 600)]
    autoenc_dec_dims = [(600, 800), (800, 300)]
    autoenc_model = AutoEncoder(autoenc_enc_dims, autoenc_dec_dims, drop=0.05)
    autoenc_model.load_state_dict(autoenc_model_states)
    autoenc_model.eval()
    for param in autoenc_model.parameters():
        param.requires_grad = False
    if opts.gpu:
        autoenc_model = autoenc_model.cuda()
elif opts.model_arch == 3:
    model = SelfTeaching(opts.prime_enc_dims, opts.aux_enc_dims, opts.cls_enc_dims, opts.drate)
    autoenc_model_states = torch.load(opts.autoenc_model)
    autoenc_enc_dims = [(300, 800), (800, 600)]
    autoenc_dec_dims = [(600, 800), (800, 300)]
    autoenc_model = AutoEncoder(autoenc_enc_dims, autoenc_dec_dims, drop=0.05)
    autoenc_model.load_state_dict(autoenc_model_states)
    autoenc_model.train()
    for param in autoenc_model.decoder.parameters():
        param.requires_grad = False
    if opts.gpu:
        autoenc_model = autoenc_model.cuda()
else:
    raise Exception('Error: model arch id not found:', opts.model_arch)

print('preprocessing time:', time() - train_start)

if opts.gpu:
    model = model.cuda()
print(model)

def train(model, data, opts):
    Utils.CreateDirectory(opts.model_path)
    Utils.OutputTrainingConfig(vars(opts), opts.model_path + '/config.txt')
    model.train()

    min_loss = float('inf')
    min_epoch = -1

    optimizer = optim.Adam(model.parameters(), lr=opts.lrate, weight_decay=opts.weight_decay)
    if opts.model_arch == 3:
        optimizer = optim.Adam(list(model.parameters()) + list(autoenc_model.encoder.parameters()),
            lr=opts.lrate, weight_decay=opts.weight_decay)
    lr_decay = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opts.lrdecay)

    if opts.model_arch == 0 or opts.model_arch == 2:
        pos_weight = torch.tensor([opts.pos_weight])
        if opts.gpu:
            pos_weight = pos_weight.cuda()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif opts.model_arch == 1:
        pos_weight = torch.tensor([opts.pos_weight])
        if opts.gpu:
            pos_weight = pos_weight.cuda()
        crit1 = nn.MSELoss()
        crit2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif opts.model_arch == 4 or opts.model_arch == 5:
        multi_class_weight = torch.tensor(eval(opts.multi_class_weight)).float()
        if opts.gpu:
            multi_class_weight = multi_class_weight.cuda()
        criterion = nn.CrossEntropyLoss(weight=multi_class_weight)
    else:
        pass
    n_epochs = opts.num_epochs

    train_order = [i for i in range(len(data))]
    print('total number of examples:', len(train_order))

    for epoch in range(n_epochs):
        print('[MAIN] Start epoch:', epoch)
        optimizer.zero_grad()
        lr_decay.step()

        random.shuffle(train_order)
        batch_size = opts.batch_size
        batch_cnt = int(len(train_order) / batch_size)
        if len(train_order) % batch_size != 0:
            batch_cnt += 1

        start_time = time()
        accu_loss = 0
        total = 0
        correct = 0
        tpSum = 0
        fpSum = 0
        fnSum = 0
        for i in range(batch_cnt):
            optimizer.zero_grad()
            cur_batch_idx = train_order[i * batch_size : (i + 1) * batch_size]
            cur_prime_batch, cur_aux_batch, cur_label_batch = [], [], []
            for idx in cur_batch_idx:
                cur_prime_batch.append(data[idx][0])
                cur_aux_batch.append(data[idx][1])
                cur_label_batch.append(data[idx][2])
            cur_prime_tensor = torch.from_numpy(np.asarray(cur_prime_batch, dtype=np.float32)).contiguous()
            cur_aux_tensor = torch.from_numpy(np.asarray(cur_aux_batch, dtype=np.float32)).contiguous()
            cur_label_tensor = torch.from_numpy(np.asarray(cur_label_batch, dtype=np.float32)).contiguous()

            if opts.gpu:
                cur_prime_tensor = cur_prime_tensor.cuda()
                cur_aux_tensor = cur_aux_tensor.cuda()
                cur_label_tensor = cur_label_tensor.cuda()

            if opts.model_arch == 0:
                pred_tensor = model(cur_prime_tensor, cur_aux_tensor)
                loss = criterion(pred_tensor.squeeze(), cur_label_tensor)
            elif opts.model_arch == 1:
                auto_recover_prime_tensor, auto_recover_aux_tensor, pred_tensor = model(cur_prime_tensor, cur_aux_tensor)
                loss1 = crit1(auto_recover_prime_tensor, cur_prime_tensor)
                loss2 = crit2(pred_tensor.squeeze(), cur_label_tensor)
                loss3 = crit1(auto_recover_aux_tensor, cur_aux_tensor)
                loss = loss1 + loss3 + loss2
            elif opts.model_arch == 2:
                cur_prime_tensor_autoenc = autoenc_model.encode(cur_prime_tensor)
                cur_aux_tensor_autoenc = autoenc_model.encode(cur_aux_tensor)
                pred_tensor = model(cur_prime_tensor_autoenc, cur_aux_tensor_autoenc)
                # pred_tensor = model(cur_prime_tensor, cur_aux_tensor)
                loss = criterion(pred_tensor.squeeze(), cur_label_tensor)
            elif opts.model_arch == 4:
                pred_tensor = model(cur_prime_tensor, cur_aux_tensor)
                loss = criterion(pred_tensor, cur_label_tensor.long())
            elif opts.model_arch == 5:
                cur_prime_tensor_autoenc = autoenc_model.encode(cur_prime_tensor)
                cur_aux_tensor_autoenc = autoenc_model.encode(cur_aux_tensor)
                pred_tensor = model(cur_prime_tensor_autoenc, cur_aux_tensor_autoenc)
                # pred_tensor = model(cur_prime_tensor, cur_aux_tensor)
                loss = criterion(pred_tensor, cur_label_tensor.long())
            else:
                pass
            loss.backward()

            grad_norm = 0.
            para_norm = 0.
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    grad_norm += m.weight.grad.data.norm() ** 2
                    para_norm += m.weight.data.norm() ** 2
                    if m.bias is not None:
                        grad_norm += m.bias.grad.data.norm() ** 2
                        para_norm += m.bias.data.norm() ** 2
            if opts.model_arch == 3:
                for m in autoenc_model.encoder.modules():
                    if isinstance(m, nn.Linear):
                        grad_norm += m.weight.grad.data.norm() ** 2
                        para_norm += m.weight.data.norm() ** 2
                        if m.bias is not None:
                            grad_norm += m.bias.grad.data.norm() ** 2
                            para_norm += m.bias.data.norm() ** 2

            grad_norm ** 0.5
            para_norm ** 0.5

            shrinkage = opts.max_grad_norm / grad_norm
            if shrinkage < 1 :
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.grad.data = m.weight.grad.data * shrinkage
                        m.bias.grad.data = m.bias.grad.data * shrinkage
                if opts.model_arch == 3:
                    for m in autoenc_model.encoder.modules():
                        if isinstance(m, nn.Linear):
                            m.weight.grad.data = m.weight.grad.data * shrinkage
                            m.bias.grad.data = m.bias.grad.data * shrinkage

            optimizer.step()
            accu_loss += loss.item()

            if opts.model_arch < 4:
                total += batch_size
                pred_class = pred_tensor.squeeze() > 0.5
                correct += torch.sum(pred_class == cur_label_tensor.byte())
                tp = torch.dot(pred_class.float(), cur_label_tensor)
                fn = cur_label_tensor.sum() - tp
                fp = pred_class.sum().float() - tp
                tpSum += tp
                fpSum += fp
                fnSum += fn

                if opts.verbose and i > 0 and i % 10 == 0:
                    prec_score = tpSum / (tpSum + fpSum)
                    recall_score = tpSum / (tpSum + fnSum)
                    print(('Batch: {batch:4d} | Loss: {loss:.4f} | Prec.: {prec:.4f} | Rec.: {recall:.4f} | p-norm: {para_norm:.3f} | g-norm {grad_norm:.3f} | Time elapsed: {time:7.2f}').format(batch=i,
                        loss=accu_loss / (i + 1), prec=prec_score, recall=recall_score, para_norm=para_norm, grad_norm=grad_norm, time=(time() - start_time)))
            elif opts.model_arch == 4 or opts.model_arch == 5:
                if opts.verbose and i > 0 and i % 10 == 0:
                    print(('Batch: {batch:4d} | Loss: {loss:.4f} | p-norm: {para_norm:.3f} | g-norm {grad_norm:.3f} | Time elapsed: {time:7.2f}').format(batch=i,
                        loss=accu_loss / (i + 1), para_norm=para_norm, grad_norm=grad_norm, time=(time() - start_time)))

        if accu_loss < min_loss:
            min_loss = accu_loss
            min_epoch = epoch
            torch.save(model.state_dict(), opts.model_path + '/model.bin')
            if opts.model_arch == 3:
                torch.save(autoenc_model.state_dict(), opts.model_path + '/autoenc_model.bin')
    print('Min loss:', min_loss, 'epoch:', min_epoch)

if opts.train:
    train(model, TRAIN, opts)

train_end = time()
print('Total training time:', train_end - train_start)
