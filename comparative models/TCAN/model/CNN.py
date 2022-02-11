import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random


class CNN_only(nn.Module):
    def __init__(self):
        super(CNN_only, self).__init__()

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

        self.relu = nn.ReLU()



        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # self.integral_fc_emotion = nn.Linear(in_features=self.in_features, out_features=4)
        #
        # self.integral_fc_emotion_center = nn.Linear(in_features=self.in_features, out_features=4)
        #
        # self.integral_fc_sex = nn.Linear(in_features=self.in_features, out_features=2)
        #
        # self.integral_fc_sex_center = nn.Linear(in_features=self.in_features, out_features=2)

        self.dropout = nn.Dropout(0.1)
        # self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, *input):
        x1 = self.conv1(input[0].unsqueeze(1))

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x2 = self.relu(x2)

        # x3 = self.maxp(x2)
        x3 = self.conv3(x2)
        x3 = self.bn3(x3)

        x3 = self.relu(x3)

        x4 = self.maxp(x3)
        x4 = self.conv4(x4)
        x4 = self.bn4(x4)

        x4 = self.relu(x4)

        x5 = self.maxp(x4)
        x5 = self.conv5(x5)
        x5 = self.bn5(x5)

        x5 = self.relu(x5)

        #
        x6 = self.maxp(x5)
        x6 = self.conv6(x6)
        x6 = self.bn6(x6)

        x6 = self.relu(x6)

        x = self.dropout(x6)



        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))



        return x
        # return out_emotion, out_emotion_center, out_sex, out_sex_center
        # print('out: ', out.shape)

if __name__ == "__main__":
    x = torch.randn(32,60, 251)
    model = CNN_only()
    model(x)