#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import time
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataSet
from model import HNSD

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(dataset, data_num, out_root):
    print(dataset, data_num)
    setup_seed(987654)

    # parameters setting
    hidden_dim = 512
    d = 16
    T = 251
    fc_num = 128

    learning_rate = 0.001
    decay = 0.000001
    Epoch = 30
    BatchSize = 32

    """load feature files"""
    if dataset == 'IEMOCAP':
        file = r'/home/liuluyao/HNSD/data/IEMOCAP_leave_{}_data.pkl'.format(str(data_num+1))
    elif dataset == 'MSP':
        file = r'/home/liuluyao/HNSD/data/MSP_leave_{}_data.pkl'.format(str(data_num+1))
    elif dataset == 'MELD':
        file = r'/home/liuluyao/HNSD/data/MELD_data_{}.pkl'.format(str(data_num))
    else:
        print('Wrong dataset name!')

    with open(file, 'rb') as f:
        features = pickle.load(f)

    if dataset == 'MELD':
        val_mfcc = np.array(features['valid_mfcc'])
        val_delta = np.array(features['valid_delta'])
        val_delta_delta = np.array(features['valid_delta_delta'])
        val_y = np.array(features['valid_emo'])
        val_sex = np.array(features['valid_sex'])

        train_mfcc = np.array(features['train_x'])
        train_delta = np.array(features['train_delta'])
        train_delta_delta = np.array(features['train_delta_delta'])
        train_y = np.array(features['train_emo'])
        train_sex = np.array(features['train_sex'])
    else:
        val_mfcc = np.array(features['valid_mfcc'])
        val_delta = np.array(features['valid_delta'])
        val_delta_delta = np.array(features['valid_delta_delta'])
        val_y = np.array(features['valid_emo'])
        val_sex = np.array(features['valid_sex'])

        train_mfcc = np.array(features['train_x'])
        train_delta = np.array(features['train_delta'])
        train_delta_delta = np.array(features['train_delta_delta'])
        train_y = np.array(features['train_emo'])
        train_sex = np.array(features['train_sex'])

    print(train_mfcc.shape)

    """training processing"""
    print('start training...')

    # load training data
    train_data = DataSet(train_mfcc, train_delta, train_delta_delta, train_y, train_sex)
    train_loader = DataLoader(train_data, batch_size=BatchSize, shuffle=True)

    # load model
    model = HNSD(in_dim=60, d=d, hidden_dim=hidden_dim,  T=T, fc_num=fc_num)

    if torch.cuda.is_available():
        model = model.cuda()

    # criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

    # result saving
    maxWA = 0
    maxUA = 0
    totalrunningtime = 0

    # training epochs
    for i in range(Epoch):
        start_time = time.time()
        tq = tqdm(len(train_y))

        model.train()
        train_loss = 0
        for _, data in enumerate(train_loader):
            mfcc, delta, delta_delta, y, sex = data
            if torch.cuda.is_available():
                mfcc = mfcc.cuda()
                delta = delta.cuda()
                delta_delta = delta_delta.cuda()

                y = y.cuda()

            out = model(mfcc, delta, delta_delta)
            loss = criterion(out, y.squeeze(1))

            train_loss += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tq.update(BatchSize)
        tq.close()

        # # learning rate adjustment
        # if i>0 and i%10 == 0:
        #     learning_rate = learning_rate/10
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = learning_rate

        """validation process"""
        end_time = time.time()
        totalrunningtime += end_time-start_time
        print('total_running_time: ', totalrunningtime)

        model.eval()
        UA = [0, 0, 0, 0]
        num_correct = 0
        class_total = [0, 0, 0, 0]
        matrix = np.mat(np.zeros((4, 4)), dtype=int)

        for i in range(len(val_y)):
            mfcc = torch.from_numpy(val_mfcc[i]).float()
            delta = torch.from_numpy(val_delta[i]).float()
            delta_delta = torch.from_numpy(val_delta_delta[i]).float()

            y = torch.from_numpy(np.array(val_y[i])).long()

            if torch.cuda.is_available():
                mfcc = mfcc.cuda()
                delta = delta.cuda()
                delta_delta = delta_delta.cuda()

                y = y.cuda()


            out = model(mfcc.unsqueeze(0), delta.unsqueeze(0), delta_delta.unsqueeze(0))

            pred = torch.max(out, 1)[1]

            if pred[0] == y.item():
                num_correct += 1
            matrix[int(y.item()), int(pred[0])] += 1

        for i in range(4):
            for j in range(4):
                class_total[i] += matrix[i,j]
            UA[i] = round(matrix[i, i] / class_total[i], 3)
        WA = num_correct/ len(val_y)
        if (maxWA<WA):
            maxWA=round(WA, 5)
            best_matrix = matrix

        if (maxUA < sum(UA) / 4):
            maxUA = round(sum(UA) / 4, 5)

        print('Acc: {:.6f}\nUA:{},{}\nmaxWA:{},maxUA{}'.format(WA, UA, sum(UA) / 4, maxWA, maxUA))


    result_file = os.path.join( out_root,'{}_{}_results.pkl'.format(dataset, data_num))
    results = {
        'WA':maxWA,
        'UA':maxUA,
        'Matrix':best_matrix
    }

    with open(result_file, 'wb') as f:
        pickle.dump(results, f)




    return maxWA, maxUA, best_matrix


