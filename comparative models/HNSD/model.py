import torch
import torch.nn as nn
import torch.functional as F

class HNSD(nn.Module):
    def __init__(self, in_dim=60, d=16, hidden_dim=512, T=251, fc_num=128):
        super(HNSD, self).__init__()
        self.in_dim = in_dim
        self.d = d
        self.hidden_dim = hidden_dim
        self.T = T
        self.fc_num = fc_num

        # LSTM
        self.LSTM_s = nn.LSTM(input_size=self.in_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.LSTM_d = nn.LSTM(input_size=2*self.in_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)

        # GMU (following by activation function, tanh-hs and hd, sigmoid-z)
        self.Ws = nn.Parameter(torch.randn(1, self.hidden_dim, self.hidden_dim), requires_grad=True)
        self.Wd = nn.Parameter(torch.randn(1, self.hidden_dim, self.hidden_dim), requires_grad=True)

        self.Wc = nn.Parameter(torch.randn(1, 1, 2 * self.hidden_dim), requires_grad=True)

        # Attention
        self.W = nn.Parameter(torch.randn(1, self.d, self.hidden_dim), requires_grad=True)
        self.b = nn.Parameter(torch.randn(1, self.d, self.T), requires_grad=True)
        self.V = nn.Parameter(torch.randn(1, self.d, 1), requires_grad=True)

        # fully connection layer
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=self.fc_num)

        self.bn = nn.BatchNorm1d(num_features=self.fc_num)
        self.relu = nn.ReLU()

        # self.fc2 = nn.Linear(in_features=self.fc_num, out_features=4)

        self.integral_fc_emotion = nn.Linear(in_features=self.fc_num, out_features=4)
        self.integral_fc_emotion_center = nn.Linear(in_features=self.fc_num, out_features=4)
        self.integral_fc_sex = nn.Linear(in_features=self.fc_num, out_features=2)
        self.integral_fc_sex_center = nn.Linear(in_features=self.fc_num, out_features=2)

    def forward(self, x_s, delta, delta_delta):

        x_d = torch.cat((delta, delta_delta), dim=-2)

        # LSTM
        outs, (hs, cs) = self.LSTM_s(x_s.permute(0,2,1))
        outd, (hd, cd) = self.LSTM_d(x_d.permute(0,2,1))

        fs = outs   # [batch_size, T, hidden_dim]
        fd = outd   # [batch_size, T, hidden_dim]


        # GMU
        h_s = torch.tanh(torch.matmul(self.Ws, fs.permute(0,2,1)))    # Ws[batch_size, hidden_dim, hidden_dim], h_s[batch_size, hidden_dim, T]
        h_d = torch.tanh(torch.matmul(self.Wd, fd.permute(0,2,1)))    # Wd[batch_size, hidden_dim, hidden_dim], h_d[batch_size, hidden_dim, T]

        z = torch.sigmoid(torch.matmul(self.Wc, torch.cat((fs, fd), dim=2).permute(0,2,1)))  # Wc[batch_size, 1, 2*hidden_dim], z[batch_size, 1, T]

        h = torch.mul(h_s, z) + torch.mul(h_d, (1-z))  # h[batch_size, hidden_dim, T]

        # Attention
        s = torch.matmul(self.V.permute(0,2,1), torch.tanh(torch.matmul(self.W, h) + self.b))         # s[batch_size, 1, T]
        alpha = torch.softmax(s, dim=-1)        # s[batch_size, 1, T]
        mu = torch.sum(torch.mul(alpha, h), dim=-1)  # mu[batch_size, hidden_dim]

        mu = torch.squeeze(mu, dim=1)

        f = self.relu(self.bn(self.fc1(mu)))     # f[batch_size, fc_num]
        # f = self.fc1(mu)
        # out = self.fc2(f)           # out[batch_size, 4]
        emotion_x = self.integral_fc_emotion(f )
        emotion_center_x = self.integral_fc_emotion_center(f)
        gender_x = self.integral_fc_sex(f)
        gender_center_x = self.integral_fc_sex_center(f )
        
        return emotion_x, gender_x, emotion_center_x, gender_center_x


if __name__ == '__main__':
    x_s = torch.randn(2, 26, 251)
    delta = torch.randn(2, 26, 251)
    delta_delta = torch.randn(2, 26, 251)

    model = HNSD()
    model.eval()
    out = model(x_s, delta, delta_delta)
    print(out.shape)







