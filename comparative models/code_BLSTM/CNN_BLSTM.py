

import torch
import torch.nn as nn
import torch.nn.functional as F

from BLSTM import BLSTM
from CNN import CNN

class CNN_BLSTM(nn.Module):
    def __init__(self):
        super(CNN_BLSTM, self).__init__()

        self.BLSTM = BLSTM()
        self.CNN = CNN()

        # self.fc = nn.Linear(in_features=64+64, out_features=4)

        self.fc_emotion = nn.Linear(in_features=64*2, out_features=4)
        self.fc_emotion_center = nn.Linear(in_features=64*2, out_features=4)
        self.fc_gender = nn.Linear(in_features=64*2, out_features=2)
        self.fc_gender_center = nn.Linear(in_features=64*2, out_features=2)
    def forward(self, x):
        x = x.transpose(1,2)
        x_lstm = x
        x_cnn = x.unsqueeze(1)
        print(x_lstm.shape)
        x_lstm = self.BLSTM(x_lstm)
        print(x_lstm.shape)
        x_cnn = self.CNN(x_cnn)
        print(x_cnn.shape)

        x = torch.cat((x_cnn, x_lstm), dim=-1)

        # x = self.fc(x)
        emotion = self.fc_emotion(x)
        emotion_center = self.fc_emotion_center(x)
        gender = self.fc_gender(x)
        gender_center = self.fc_gender_center(x)

        return emotion, gender, emotion_center,  gender_center

if __name__ == '__main__':

    from torchsummary import summary
    x = torch.randn(32, 60, 251)
    model = CNN_BLSTM()
    # summary(model, ( 60, 251))
    x = model(x)
