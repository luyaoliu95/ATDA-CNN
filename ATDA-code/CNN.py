import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

from dyrelu import DyReLUB


class CNN(nn.Module):
    def __init__(self, DyRELU=True, Center=True, Gender=True):
        super(CNN, self).__init__()

        self.DyRELU = DyRELU
        self.Center = Center
        self.Gender = Gender

        self.in_features = 128
        self.conv1 = nn.Conv2d(kernel_size=(3, 3), in_channels=1, out_channels=8)
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=8, out_channels=16, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=96, padding=1)
        self.conv6 = nn.Conv2d(kernel_size=(3, 3), in_channels=96, out_channels=128, padding=1)


        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(96)
        self.bn6 = nn.BatchNorm2d(128)

        if self.DyRELU:
            self.relu1 = DyReLUB(8, conv_type='2d')
            self.relu2 = DyReLUB(16, conv_type='2d')
            self.relu3 = DyReLUB(32, conv_type='2d')
            self.relu4 = DyReLUB(64, conv_type='2d')
            self.relu5 = DyReLUB(96, conv_type='2d')
            self.relu6 = DyReLUB(128, conv_type='2d')
        else:
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.relu4 = nn.ReLU()
            self.relu5 = nn.ReLU()
            self.relu6 = nn.ReLU()


        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))

        self.integral_fc_emotion = nn.Linear(in_features=self.in_features, out_features=4)
        if self.Center:
            self.integral_fc_emotion_center = nn.Linear(in_features=self.in_features, out_features=4)
        if self.Gender:
            self.integral_fc_sex = nn.Linear(in_features=self.in_features, out_features=2)
        if self.Center and self.Gender:
            self.integral_fc_sex_center = nn.Linear(in_features=self.in_features, out_features=2)

        self.dropout = nn.Dropout(0.1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, *input):
        x1 = self.conv1(input[0])
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x2 = self.relu2(x2)

        x3 = self.maxp(x2)
        x3 = self.conv3(x3)
        x3 = self.bn3(x3)

        x3 = self.relu3(x3)

        x4 = self.maxp(x3)
        x4 = self.conv4(x4)
        x4 = self.bn4(x4)

        x4 = self.relu4(x4)

        x5 = self.maxp(x4)
        x5 = self.conv5(x5)
        x5 = self.bn5(x5)

        x5 = self.relu5(x5)

        #
        x6 = self.maxp(x5)
        x6 = self.conv6(x6)
        x6 = self.bn6(x6)

        x6 = self.relu6(x6)


        x = self.dropout(x6)

        # print('attn: ', attn.shape)
        out_x = self.pool(x)

        out_x = torch.reshape(out_x, (out_x.shape[0], out_x.shape[1]))

        # print('out_x: ', out_x.shape)
        out_emotion = self.integral_fc_emotion(out_x)

        if self.Gender and self.Center:
            out_sex = self.integral_fc_sex(out_x)
            out_sex_center = self.integral_fc_sex_center(out_x)
            out_emotion_center = self.integral_fc_emotion_center(out_x)
            return out_x, out_emotion, out_sex, out_emotion_center, out_sex_center
        if self.Gender:
            out_sex = self.integral_fc_sex(out_x)
            return out_x, out_emotion, out_sex
        if self.Center:
            out_emotion_center = self.integral_fc_emotion_center(out_x)
            return out_x, out_emotion, out_emotion_center
        else:
            return out_x, out_emotion
        # print('out: ', out.shape)


    #
