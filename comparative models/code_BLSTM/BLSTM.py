

import torch
import torch.nn as nn
import torch.nn.functional as F

class BLSTM(nn.Module):
    def __init__(self, hidden_dim=60):
        super(BLSTM, self).__init__()

        # input_size: [batch_size, seq_len, input_size]
        self.blstm = nn.LSTM(batch_first=True, input_size=60, hidden_size=hidden_dim, bidirectional=True, num_layers=2)

        self.fc1 = nn.Linear(in_features=30120, out_features=3000)
        self.fc2 = nn.Linear(in_features=3000, out_features=64)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        # self.fc_emotion = nn.Linear(in_features=64, out_features=4)
        # self.fc_emotion_center = nn.Linear(in_features=64, out_features=4)
        # self.fc_gender = nn.Linear(in_features=64, out_features=2)
        # self.fc_gender_center = nn.Linear(in_features=64, out_features=2)



    def forward(self, x):

        # – output(seq_len, batch, num_directions * hidden_size)
        # – h_n(num_layers * num_directions, batch, hidden_size)
        # – c_n(num_layers * num_directions, batch, hidden_size)
        # x = x.transpose(1,2)

        batch_size = x.shape[0]


        output, hn = self.blstm(x)

        # print(output.shape)
        # print(hn[0].shape)
        # print(hn[1].shape)

        x = output.reshape(batch_size, -1)
        #
        # hn = torch.reshape(batch_size,)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        # emotion = self.fc_emotion(x)
        # emotion_center = self.fc_emotion_center(x)
        # gender = self.fc_gender(x)
        # gender_center = self.fc_gender_center(x)
        # emotion, gender, emotion_center,  gender_center

        return x

if __name__ == '__main__':
    x = torch.randn(32,  60, 251)
    model = BLSTM()
    model(x)