#! /usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import pickle
import logging
import time
import random
import warnings
warnings.filterwarnings('ignore')

from data_loader_mfcc import DataSet
from transformer import Encoder
# from cnn_tcn_atten import AttenCNN


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



# setup_seed(111111)
# setup_seed(123456)
# setup_seed(0)
# setup_seed(999999)
def train(data_num):

    setup_seed(987654)

    # logger setting
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关

    # parameters setting

    learning_rate = 0.00001
    lr_cent = 0.15
    Epoch = 50
    BatchSize = 32
    # MODEL_NAME='MULITMANET_with_gender'

    #MODEL_PATH = './model_result/IEMOCAP.pth'.format(str(case),element, file_num)

    # load features file
    print('load features data ...')
    logging.info('load features data...')
    # file = r'J:/code/MMATA/data/MSP_leave_{}_data_16000.pkl'.format(data_num)
    file = r'J:/code/MMATA/data/MELD_data_{}.pkl'.format(data_num)

    with open(file, 'rb') as f:
        features = pickle.load(f)

    val_X = np.array(features['valid_x'])
    val_y = np.array(features['valid_emo'])
    val_sex = np.array(features['valid_sex'])

    train_X = np.array(features['train_x'])
    train_y = np.array(features['train_emo'])
    train_sex = np.array(features['train_sex'])
    print(train_X.shape)


    '''training processing'''
    print('start training...')
    logging.info('start training....')
    # load data
    #  train_trans, train_trans_len,
    train_data = DataSet(train_X, train_y, train_sex)
    train_loader = DataLoader(train_data, batch_size=BatchSize, shuffle=True)

    # load model
    # ahead_text = 7, ahidden_text = 96

    model = Encoder(max_seq_len=260, f_dim=60, dim_per_head=60, num_heads=8, ffn_dim=2048, dropout_rate=0.1, num_layers=6)

    if torch.cuda.is_available():
        model = model.cuda()

    # criterion
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    # result saving
    maxWA = 0
    maxUA = 0
    totalrunningtime = 0

    for i in range(Epoch):
        start_time = time.time()
        tq = tqdm(len(train_y))

        model.train()
        print_loss=0
        j = 0
        for _,data in enumerate(train_loader):
            x, y, sex= data
            if torch.cuda.is_available():
                x = x.cuda()

                y = y.cuda()



            out_emotion = model(x.unsqueeze(1))

            loss_emotion = criterion(out_emotion, y.squeeze(1))


            # loss_gender = criterion(out_gender, sex.squeeze(1))
            # center_loss_gender = center_loss_sex(out_gender_center, sex.squeeze(1))

            # loss = alpha*(loss_emotion + center_rate*center_loss_emotion)+beta * (loss_gender+center_rate*center_loss_gender)
            loss = loss_emotion
            print_loss += loss.data.item()*BatchSize
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            tq.update(BatchSize)
        tq.close()
        print('epoch: {}, loss: {:.4}'.format(i, print_loss/len(train_y)))
        logging.info('epoch: {}, loss: {:.4}'.format(i, print_loss))
        if i>0 and i%10 == 0:
            learning_rate = learning_rate/10
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        '''validation process'''
        end_time = time.time()
        totalrunningtime += end_time-start_time
        print('total_running_time:', totalrunningtime)
        model.eval()
        UA = [0, 0, 0, 0]
        num_correct = 0
        class_total = [0, 0, 0, 0]
        matrix = np.mat(np.zeros((4,4)), dtype=int)


        for i in range(len(val_y)):
            x = torch.from_numpy(val_X[i]).float()

            y = torch.from_numpy(np.array(val_y[i])).long()

            if torch.cuda.is_available():
                x = x.cuda()

                y = y.cuda()
                sex = sex.cuda()


            out_emotion = model(x.unsqueeze(0).unsqueeze(0))
            pred_emotion = torch.max(out_emotion, 1)[1]
            # pred_gender = torch.max(out_gender, 1)[1]

            if pred_emotion[0] == y.item():
                num_correct +=1
            matrix[int(y.item()), int(pred_emotion[0])] +=1

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






    return maxWA, maxUA,best_matrix


if __name__ == '__main__':
    train_id = ['0', '1', '2', '3', '4']
    results = {}
    for i in train_id:
        max_WA, max_UA, best_matrix = train(i)
        results[i] = {
            'WA':max_WA,
            'UA':max_UA,
            'Matrix':best_matrix
        }

    results_file = r'J:\code\MMATA\comparison_results\2020-Head_Attention\MELD.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(results)
