

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(50, 1), stride=(10, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3))
        )


        self.conv_2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 3))
        )

        self.fc1 = nn.Linear(in_features=1728, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.fc_classifier = nn.Linear(in_features=64, out_features=4)

    def forward(self, x):

        batch_size = x.shape[0]
        # x = x.transpose(-1, -2)
        # print(x.shape)

        x = self.conv_1(x)
        # print(x.shape)


        x = self.conv_2(x)
        # print(x.shape)

        x = x.view(batch_size, -1)
        # print(x.shape)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))

        x = self.fc_classifier(x)

        return x


if __name__ == '__main__':
    x = torch.randn(32, 1, 60, 251)
    model = CNN()
    model(x)
