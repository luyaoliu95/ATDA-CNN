import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

from dyrelu import DyReLUB
from TAMs import TAM

class TAM_CNN(nn.Module):
    # TAM(8,8, atten=True, atten_method='add', c_scale_method='conv',
    #                 f_scale_method='maxpool', t_scale_method='maxpool', atten_view=('c', 't', 'f'),
    #                  f_scale=3, t_scale=3, c_scale=3)
    def __init__(self, TAM_inc=8, TAM_outc=8, atten=False, atten_method='matmul', atten_view=('t'),
                 c_scale_method='conv', t_scale_method='conv', f_scale_method='conv',
                 f_scale=3, t_scale=3, c_scale=3, center=True, gender=True):
        super(TAM_CNN, self).__init__()

        self.in_features = 156
        self.center = center
        self.gender = gender
        self.conv2_in_c = 8*len(atten_view)

        self.conv1 = nn.Conv2d(kernel_size=(3, 3), in_channels=1, out_channels=8)
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=self.conv2_in_c, out_channels=16, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=96, padding=1)
        self.conv6 = nn.Conv2d(kernel_size=(3, 3), in_channels=96, out_channels=128, padding=1)
        self.conv7 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=156, padding=1)

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(96)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(156)

        self.relu1 = DyReLUB(8, conv_type='2d')
        self.relu2 = DyReLUB(16, conv_type='2d')
        self.relu3 = DyReLUB(32, conv_type='2d')
        self.relu4 = DyReLUB(64, conv_type='2d')
        self.relu5 = DyReLUB(96, conv_type='2d')
        self.relu6 = DyReLUB(128, conv_type='2d')
        self.relu7 = DyReLUB(156, conv_type='2d')

        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))


        self.integral_fc_emotion = nn.Linear(in_features=self.in_features, out_features=4)
        if center:
            self.integral_fc_emotion_center = nn.Linear(in_features=self.in_features, out_features=4)
        if gender:
            self.integral_fc_sex = nn.Linear(in_features=self.in_features, out_features=2)
            if center:
                self.integral_fc_sex_center = nn.Linear(in_features=self.in_features, out_features=2)

        self.dropout = nn.Dropout(0.1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # TAM
        self.TAM = TAM(TAM_inc, TAM_outc, atten=atten, atten_method=atten_method, atten_view=atten_view,
                       c_scale_method=c_scale_method, f_scale_method=f_scale_method, t_scale_method=t_scale_method,
                       f_scale=f_scale, t_scale=t_scale, c_scale=c_scale)
        #




    def forward(self, *input):
        input = input[0]
        x1 = self.conv1(input)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        #
        x1 = self.TAM(x1, input)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x2 = self.relu2(x2)

        x2 = self.maxp(x2)
        x3 = self.conv3(x2)
        x3 = self.bn3(x3)

        x3 = self.relu3(x3)



        x3 = self.maxp(x3)
        x4 = self.conv4(x3)
        x4 = self.bn4(x4)

        x4 = self.relu4(x4)


        x4 = self.maxp(x4)
        x5 = self.conv5(x4)
        x5 = self.bn5(x5)

        x5 = self.relu5(x5)

        #
        x5 = self.maxp(x5)
        x6 = self.conv6(x5)
        x6 = self.bn6(x6)

        x6 = self.relu6(x6)

        x6 = self.maxp(x6)
        x7 = self.conv6(x6)
        x7 = self.bn6(x7)

        x7 = self.relu6(x7)


        x = self.dropout(x6)
        out_x = self.pool(x)


        out_x = torch.reshape(out_x, (out_x.shape[0], out_x.shape[1]))

        # print('out_x: ', out_x.shape)
        out_emotion = self.integral_fc_emotion(out_x)
        out_emotion_center = None
        out_sex = None
        out_sex_center = None
        if self.center:
            out_emotion_center = self.integral_fc_emotion_center(out_x)
        if self.gender:
            out_sex = self.integral_fc_sex(out_x)
            if self.center:
                out_sex_center = self.integral_fc_sex_center(out_x)
        # print('out: ', out.shape)

        return out_x, out_emotion, out_sex, out_emotion_center, out_sex_center
        # 3 out_sex,  5 , out_sex_center