import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dyrelu import DyReLUB

class TAM(nn.Module):
    def __init__(self, in_c, out_c, atten=False, atten_method='matmul', c_scale_method='conv',
                 f_scale_method='conv', t_scale_method='conv', atten_view=('t'),
                 f_scale=1, t_scale=1, c_scale=1):
        super(TAM, self).__init__()

        self.atten = atten

        self.atten_view = atten_view
        self.atten_method = atten_method
        self.c_scale_method = c_scale_method
        self.f_scale_method = f_scale_method
        self.t_scale_method = t_scale_method
        self.f_scale = f_scale
        self.t_scale = t_scale
        self.c_scale = c_scale
        self.res_out_c = out_c * len(atten_view)
        self.f = 58
        self.t = 249
        self.c = 8

        self.differ_conv = nn.Conv2d(kernel_size=1, in_channels=in_c, out_channels=out_c)
        self.res_conv = nn.Conv2d(kernel_size=(3, 3), in_channels=1, out_channels=self.res_out_c)
        self.res_bn = nn.BatchNorm2d(self.res_out_c)
        self.res_relu = DyReLUB(self.res_out_c, conv_type='2d')

        # attention
        if self.atten:
            self.atten_query = nn.Conv2d(in_channels=8, out_channels=8,kernel_size=1)
            self.atten_key = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1)
            # self.atten_value = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1)

        if self.f_scale >1:
            self.f_query, self.f_key, self.f_value = self.multi_scales(self.f, self.f_scale_method, self.f_scale)
        if self.t_scale >1:
            self.t_query, self.t_key, self.t_value = self.multi_scales(self.t, self.t_scale_method, self.t_scale)
        if self.c_scale > 1:
            self.c_query, self.c_key, self.c_value = self.multi_scales(self.c, self.c_scale_method, self.c_scale)


    def forward(self, x, input):
        x_t_fea, _ = x.split([x.shape[-1] - 1, 1], dim=-1)
        first_t, _ = x.split([1, x.shape[-1] - 1], dim=-1)

        x_conv = self.differ_conv(x)
        _, x_t_plusone_q = x_conv.split([1, x_conv.shape[-1] - 1], dim=-1)
        x_query = x_t_plusone_q - x_t_fea

        differ = torch.cat((first_t, x_query), dim=-1)

        x_res = self.res_conv(input)
        x_res = self.res_bn(x_res)
        x_res = self.res_relu(x_res)

        if self.atten:
            return self.attention(differ) + x_res
        else:
            return differ + x_res


    def multi_scales(self, in_c, scale_method, num_scale):
        c = in_c

        query = nn.ModuleList()
        key = nn.ModuleList()
        value = nn.ModuleList()

        if scale_method == 'conv':
            for i in range(num_scale - 1):
                query.append(
                    nn.Conv1d(in_channels=c, out_channels=c, kernel_size=i + 1, stride=i+1))
                key.append(nn.Conv1d(in_channels=c, out_channels=c, kernel_size=i + 1, stride=i+1))
                if self.atten_method not in ('matmul', 'cos'):
                    value.append(nn.Conv1d(in_channels=c, out_channels=c, kernel_size=i + 1, stride=i+1))

        elif scale_method == 'avgpool':
            for i in range(num_scale - 1):
                query.append(nn.AvgPool1d(kernel_size=i + 1, stride=i+1))
                key.append(nn.AvgPool1d(kernel_size=i + 1, stride=i+1))
                if self.atten_method not in ('matmul', 'cos'):
                    value.append(nn.AvgPool1d(kernel_size=i + 1, stride=i+1))

        elif scale_method == 'maxpool':
            for i in range(num_scale - 1):
                query.append(nn.MaxPool1d(kernel_size=i + 1, stride=i+1))
                key.append(nn.MaxPool1d(kernel_size=i + 1, stride=i+1))
                if self.atten_method not in ('matmul', 'cos'):
                    value.append(nn.MaxPool1d(kernel_size=i + 1, stride=i+1))
        else:
            raise Exception('scale_method must is one of strings in {"conv", "avgpool", "maxpool"}!')

        return query, key, value

    def attention(self, differ):

        Q = self.atten_query(differ)
        K = self.atten_key(differ)
        V = differ # self.atten_value(differ)

        atten_differ = None

        bs = Q.shape[0]
        c = Q.shape[1]
        f = Q.shape[2]
        t = Q.shape[3]
        # feature attention
        if "f" in self.atten_view:

            Q_f = torch.reshape(Q, (bs, f, c * t))
            K_f = torch.reshape(K, (bs, f, c * t))
            V_f = torch.reshape(V, (bs, f, c * t))

            atten_f = self.get_atten_differ(Q_f, K_f, V_f).view(bs, c, f, t)
            if self.f_scale > 1:

                for i in range(self.f_scale - 1):
                    if self.atten_method not in ('matmul', 'cos'):
                        atten_f_scaled = self.get_atten_differ(self.f_query[i](Q_f),
                                                               self.f_key[i](K_f),
                                                               self.f_value[i](V_f))

                        atten_f_scaled = F.interpolate(atten_f_scaled, size=c * t, mode='nearest')
                        atten_f_scaled = atten_f_scaled.view(bs, c, f, t)
                    else:
                        atten_f_scaled = self.get_atten_differ(self.f_query[i](Q_f),
                                                               self.f_key[i](K_f),
                                                               V_f).view(bs, c, f, t)
                    atten_f = atten_f + atten_f_scaled

            if atten_differ == None:
                atten_differ = atten_f
            else:
                atten_differ = torch.cat((atten_differ, atten_f), dim=1)

        # channel attention
        if 'c' in self.atten_view:

            Q_c = torch.reshape(Q, (bs, c, t * f))
            K_c = torch.reshape(K, (bs, c, t * f))
            V_c = torch.reshape(V, (bs, c, t * f))

            atten_c = self.get_atten_differ(Q_c, K_c, V_c).view(bs, c, f, t)
            if self.c_scale > 1:
                for i in range(self.c_scale - 1):
                    if self.atten_method not in ('matmul', 'cos'):
                        atten_c_scaled = self.get_atten_differ(self.c_query[i](Q_c),
                                                               self.c_key[i](K_c),
                                                               self.c_value[i](V_c))
                        atten_c_scaled = F.interpolate(atten_c_scaled, size=t * f, mode='nearest')
                        atten_c_scaled = atten_c_scaled.view(bs, c, f, t)
                    else:
                        atten_c_scaled = self.get_atten_differ(self.c_query[i](Q_c),
                                                               self.c_key[i](K_c),
                                                               V_c).view(bs, c, f, t)
                    atten_c = atten_c + atten_c_scaled

            if atten_differ == None:
                atten_differ = atten_c
            else:
                atten_differ = torch.cat((atten_differ, atten_c), dim=1)

        # time attention

        if "t" in self.atten_view:

            Q_t = torch.reshape(Q, (bs, t, c * f))
            K_t = torch.reshape(K, (bs, t, c * f))
            V_t = torch.reshape(V, (bs, t, c * f))

            atten_t = self.get_atten_differ(Q_t, K_t, V_t).view(bs, c, f, t)

            if self.t_scale > 1:
                for i in range(self.t_scale - 1):
                    if self.atten_method not in ('matmul', 'cos'):
                        atten_t_scaled = self.get_atten_differ(self.t_query[i](Q_t),
                                                               self.t_key[i](K_t),
                                                               self.t_value[i](V_t))
                        atten_t_scaled = F.interpolate(atten_t_scaled, size=c * f, mode='nearest')
                        atten_t_scaled = atten_t_scaled.view(bs, c, f, t)
                    else:
                        atten_t_scaled = self.get_atten_differ(self.t_query[i](Q_t),
                                                               self.t_key[i](K_t),
                                                               V_t).view(bs, c, f, t)

                    atten_t = atten_t + atten_t_scaled

            if atten_differ == None:
                atten_differ = atten_t
            else:
                atten_differ = torch.cat((atten_differ, atten_t), dim=1)

        return atten_differ

    def get_atten_differ(self, Q, K, V):

        # the shape of Q,K,V is (batchsize, atten_view, dimension)
        if self.atten_method == 'matmul':
            atten_alpha = F.softmax(torch.matmul(Q, K.permute(0,2,1))/np.power(Q.shape[-1], 0.5), dim=1)

            atten_differ = torch.matmul(atten_alpha, V)  # the shape of atten_differ is same as Q, K, V

        elif self.atten_method == 'mul':
            atten_alpha = F.softmax(torch.mul(Q, K) / np.power(Q.shape[-1], 0.5), dim=1)
            atten_differ = torch.mul(atten_alpha, V)
        elif self.atten_method == 'add':
            atten_alpha = F.softmax(torch.add(Q, K), dim=1)
            atten_differ = torch.mul(atten_alpha, V)
        elif self.atten_method == 'cos':

            atten_alpha = F.softmax(torch.cosine_similarity(Q, K, dim=2), dim=1)

            atten_alpha = atten_alpha.unsqueeze(-1).expand(atten_alpha.shape[0], atten_alpha.shape[1], atten_alpha.shape[1])
            atten_differ = torch.matmul(atten_alpha, V)

        elif self.atten_method == 'sub':
            atten_alpha = F.softmax((Q-K) , dim=1)
            atten_differ = torch.mul(atten_alpha, V)
        else:
            raise Exception('scale_method must is one of strings in {"matmul", "mul", "add", "cos", "sub"}!')

        return atten_differ


if __name__ == '__main__':
    x1 = torch.randn(32, 8, 58, 249)
    x = torch.randn(32, 1, 60, 251)
    model = TAM(8,8, atten=True, atten_method='cos', c_scale_method='conv',
                f_scale_method='maxpool', t_scale_method='conv', atten_view=('t'),
                 f_scale=1, t_scale=2, c_scale=1)
    out = model(x1, x)
