import torch
import torch.nn as nn

class STC_Attention(nn.Module):
    def __init__(self, t=251, f=60, flat=110208):
        super(STC_Attention, self).__init__()
        self.t = t
        self.f = f
        self.flat = flat

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)

        self.conv_t = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(self.f, 1))
        self.conv_f = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, self.t))
        self.adap_maxpool = nn.AdaptiveMaxPool2d((self.f, 1))
        self.adap_avgpool = nn.AdaptiveAvgPool2d((self.f, 1))
        self.global_avg = nn.AdaptiveAvgPool2d(1)

        # mlp
        self.mlp1 = nn.Linear(in_features=32, out_features=int(32/2))
        self.mlp2 = nn.Linear(in_features=int(32/2), out_features=32)

        # classifier
        self.class_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(in_features=self.flat, out_features=1024)
        # self.fc2 = nn.Linear(in_features=1024, out_features=4)

        self.integral_fc_emotion = nn.Linear(in_features=1024, out_features=4)
        self.integral_fc_emotion_center = nn.Linear(in_features=1024, out_features=4)
        self.integral_fc_sex = nn.Linear(in_features=1024, out_features=2)
        self.integral_fc_sex_center = nn.Linear(in_features=1024, out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)

        # spectro-temporal attention
        x_maxpool = self.adap_maxpool(x.permute(0,3,2,1)).permute(0,3,2,1)
        x_avgpool = self.adap_avgpool(x.permute(0,3,2,1)).permute(0,3,2,1)
        # print(x_avgpool.shape)
        # print(x_maxpool.shape)

        x_concat = torch.cat((x_avgpool, x_maxpool), dim=1)
        # print('x_concat: ', x_concat.shape)

        x_t = self.conv_t(x_concat)
        x_f = self.conv_f(x_concat)
        # print('x_t: ', x_t.shape)
        # print('x_f: ', x_f.shape)

        ST_w = torch.matmul(x_f, x_t)

        # channel attention
        x_c = self.global_avg(x).reshape(x.shape[0], x.shape[1])
        x_mlp1 = self.mlp1(x_c)
        x_mlp2 = self.mlp2(x_mlp1)
        x_mlp = x_mlp2.view(x_mlp2.shape[0], x_mlp2.shape[1], 1)
        # print('x_mlp: ', x_mlp.shape)
        # print('ST_w: ', ST_w.shape)

        STC_w = torch.matmul(x_mlp, ST_w.reshape(ST_w.shape[0], 1, ST_w.shape[2]*ST_w.shape[3]))
        STC_w = STC_w.reshape(STC_w.shape[0], STC_w.shape[1], self.f, self.t)
        # print(STC_w.shape)

        # res
        x_attention = torch.mul(torch.softmax(STC_w, dim=1), x) + x

        # classifier
        x_c = self.class_conv(x_attention)
        x_pool = self.pool(x_c)
        x = x_pool.reshape(x_pool.shape[0], x_pool.shape[1]*x_pool.shape[2]*x_pool.shape[3])
        # print('x_flatten: ', x_flatten.shape)

        x = self.fc1(x)

        emotion_x = self.integral_fc_emotion(x)
        emotion_center_x = self.integral_fc_emotion_center(x)
        gender_x = self.integral_fc_sex(x)
        gender_center_x = self.integral_fc_sex_center(x)

        return emotion_x, gender_x, emotion_center_x, gender_center_x




if __name__ == '__main__':
    x = torch.randn(32, 1, 60, 251)
    model = STC_Attention()
    x_out = model(x)



