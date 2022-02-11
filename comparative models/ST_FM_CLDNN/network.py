

import torch
import torch.nn as nn
import torch.nn.functional as F

class ST_FM_CLDNN(nn.Module):
    def __init__(self):
        super(ST_FM_CLDNN, self).__init__()

        # 3D conv layers (ST-CNN)
        self.STconv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(1, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2), stride=2)
        )

        self.STconv2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(1,3,3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 1), stride=2)
        )

        # FM-CNN
        self.FMconv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(15, 1), stride=1, padding=1),

            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.FMconv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(15, 2), stride=1, padding=1),

            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.FMconv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(15, 4), stride=1, padding=1),

            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.FMconv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(15, 8), stride=1, padding=1),

            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.FMPool = nn.AdaptiveMaxPool2d(1)

        # BLSTM
        self.blstm = nn.LSTM(batch_first=True, input_size=4*96, hidden_size=128, bidirectional=True, num_layers=2)
        self.dropout = nn.Dropout(0.2)

        # FC layers
        self.fc1_emotion = nn.Sequential(
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU()
        )
        self.fc1_emotion_center = nn.Sequential(
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU()
        )
        self.fc1_gender = nn.Sequential(
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU()
        )
        self.fc1_gender_center = nn.Sequential(
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU()
        )

        self.fc2_emotion = nn.Sequential(
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU()
        )
        self.fc2_emotion_center = nn.Sequential(
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU()
        )
        self.fc2_gender = nn.Sequential(
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU()
        )
        self.fc2_gender_center = nn.Sequential(
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU()
        )


        self.fc3_emotion = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU()
        )
        self.fc3_emotion_center = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU()
        )
        self.fc3_gender = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU()
        )
        self.fc3_gender_center = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU()
        )

        # classifier
        self.fc4_emotion = nn.Linear(in_features=32, out_features=4)
        self.fc4_emotion_center = nn.Linear(in_features=32, out_features=4)
        self.fc4_gender = nn.Linear(in_features=32, out_features=2)
        self.fc4_gender_center = nn.Linear(in_features=32, out_features=2)



    def seperate_block(self, x):
        # the shape of x is [batch_size, channel, depth, features, frames]
        batch_size, channel, depth, features, frames = x.shape

        x = x[:, :, :, :, :240].reshape(-1, channel, depth, features, 16)

        return x, x.shape[0] // batch_size

    def forward(self, x):
        # print(x.shape)
        batch_size = x.shape[0]
        x, num_blocks = self.seperate_block(x)
        # print(x.shape)
        # 3D conv layers ST-CNN
        x = self.STconv1(x)
        # print('conv1: ', x.shape)
        x = self.STconv2(x)
        # print('conv2: ', x.shape)

        x = x.reshape(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*x.shape[4])
        # print(x.shape)
        # FM-CNN

        x1 = self.FMPool(self.FMconv1(x)).squeeze(-1).squeeze(-1)
        x2 = self.FMPool(self.FMconv2(x)).squeeze(-1).squeeze(-1)
        x3 = self.FMPool(self.FMconv3(x)).squeeze(-1).squeeze(-1)
        x4 = self.FMPool(self.FMconv4(x)).squeeze(-1).squeeze(-1)

        # print(x1.shape, x2.shape, x3.shape, x4.shape)

        x = torch.cat((x1, x2, x3, x4), dim=-1)

        # print(x.shape, num_blocks)

        x = x.reshape(batch_size, -1, x.shape[-1])

        # print(x.shape)

        # lstm
        x_lstm, hn = self.blstm(x)

        # print(hn[0].shape)
        # print(x_lstm.shape)



        avg = torch.mean(x_lstm, dim=1)
        std = torch.std(x_lstm, dim=1)
        min = torch.min(x_lstm, dim=1).values
        max = torch.max(x_lstm, dim=1).values

        # print(min.shape)

        x = torch.cat((avg, std, min, max), dim=-1)
        # print(x.shape)

        x = self.dropout(x)

        # fc layers
        x1 = self.fc1_emotion(x)
        x1 = self.fc2_emotion(x1)
        x1 = self.fc3_emotion(x1)
        emotion = self.fc4_emotion(x1)

        x2 = self.fc1_gender(x)
        x2 = self.fc2_gender(x2)
        x2 = self.fc3_gender(x2)
        gender = self.fc4_gender(x2)
        #
        x3 = self.fc1_emotion_center(x)
        x3 = self.fc2_emotion_center(x3)
        x3 = self.fc3_emotion_center(x3)
        emotion_center = self.fc4_emotion_center(x3)
        #
        x4 = self.fc1_gender_center(x)
        x4 = self.fc2_gender_center(x4)
        x4 = self.fc3_gender_center(x4)
        gender_center = self.fc4_gender_center(x4)


        # emotion = self.fc_emotion(x)
        # gender = self.fc_gender(x)
        # emotion_center = self.fc_emotion_center(x)
        # gender_center = self.fc_gender_center(x)

        # , gender, emotion_center, gender_center
        return emotion, gender, emotion_center, gender_center



if __name__ == "__main__":

    # input size: (Batch_size, Channel, Deep, High, Width), Deep维度包含时序信息
    from torchsummary import summary

    x = torch.randn(32, 1, 3, 60, 251)

    model = ST_FM_CLDNN()

    model(x)

