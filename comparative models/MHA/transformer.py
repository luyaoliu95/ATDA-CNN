import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from model.CNN import CNN_only

import numpy as np


class PositionEncoder(nn.Module):
    def __init__(self, f_dim, max_seq_len, dropout_rate=0.5):
        super(PositionEncoder, self).__init__()

        self.max_seq_len = max_seq_len
        self.f_dim = f_dim
        self.dropout = nn.Dropout(dropout_rate)
        # self.LayerNorm = nn.LayerNorm([max_seq_len, f_dim])

        self.position_embeddings = nn.Embedding(max_seq_len, f_dim)
        # initialize_weight(self.position_embeddings)

    def forward(self, x):
        position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=x.device)

        position_ids = position_ids.unsqueeze(0).expand(x.shape[0], self.max_seq_len)

        position_embeddings = self.position_embeddings(position_ids)

        x = x + position_embeddings
        return self.dropout(x)


def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)

class GEGLU(nn.Module):
    # https://arxiv.org/pdf/2002.05202.pdf
    # 有关于GEGLU函数的定义：GEGLU(x, W, V, b, c) = GELU(xW + b) ⊗ (xV + c)
    # GLU = gated linear units

    def forward(self, x):
        # print('in geglu x={}'.format(x.shape)) # x=[1, 256, 4096]
        x, gates = x.chunk(2, dim = -1)
        # 按照最后一列切割成两个tensor
        # print('in middle of geglu x={}, gates={}'.format(x.shape, gates.shape))
        # x=[1, 256, 2048], gates=[1, 256, 2048]

        out = x * F.gelu(gates)
        # print('out geglu out={}'.format(out.shape))
        # out=[1, 256, 2048]
        return out


def fourier_encode(x, max_freq, num_bands = 4, base = 2):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(1., np.log(max_freq / 2) / np.log(base), num_bands,
                            base = base, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * np.pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1)
    return x



class MultiHeadAtten(nn.Module):
    """
    The multi-head attention module in transformer.

    f_dim: the dimension of input data;
    atten_layer_type: the type of the layers to generate Q, K, V. (1D-Conv, Linear);
    n_heads: the number of heads in the multi-head attention module;
    mask: the mask added to the attentional score;
    """
    def __init__(self, f_dim, atten_layer_type, dim_per_head, n_heads=8, need_mask=False, dropout_rate=0.5):
        super(MultiHeadAtten, self).__init__()
        self.f_dim = f_dim
        self.n_heads = n_heads
        self.dim_per_head = dim_per_head

        self.atten_layer_type = atten_layer_type
        self.need_mask = need_mask
        self.dropout = nn.Dropout(dropout_rate)

        self.K_layer = self.kqv_generate_layer(atten_layer_type, f_dim, dim_per_head*n_heads)
        # initialize_weight(self.K_layer)
        self.Q_layer = self.kqv_generate_layer(atten_layer_type, f_dim, dim_per_head*n_heads)
        # initialize_weight(self.Q_layer)
        self.V_layer = self.kqv_generate_layer(atten_layer_type, f_dim, dim_per_head*n_heads)
        # initialize_weight(self.V_layer)
        self.combine_heads = nn.Linear(in_features=self.n_heads*self.dim_per_head, out_features=f_dim)
        # initialize_weight(self.combine_heads)



    def forward(self, x):
        batch_size = x.size(0)
        # the shape of x: [batch_size, num_segments, f_dim]
        # separated shape: [batch_size, n_heads, number_segments, f_dim/n_heads]
        K = self.K_layer(x)
        K = self.seperate_heads(K, self.n_heads, self.dim_per_head)
        Q = self.Q_layer(x)
        Q = self.seperate_heads(Q, self.n_heads, self.dim_per_head)
        V = self.V_layer(x)
        V = self.seperate_heads(V, self.n_heads, self.dim_per_head)

        attn_x = self.attention(K, Q, V, self.need_mask)
        attn_x = attn_x.transpose(1,2).contiguous().view(batch_size, -1, self.n_heads*self.dim_per_head)

        attn_x = self.combine_heads(attn_x)

        return attn_x

    def kqv_generate_layer(self, atten_layer_type, f_dim,  out_dim):
        if atten_layer_type == "Linear":
            return nn.Linear(in_features=f_dim, out_features=out_dim)
        elif atten_layer_type == '1D-Conv':
            return None
        else:
            print("The 'atten_layer_type' must be the one of {'Linear', '1D-Conv'}!")
            return None

    def seperate_heads(self, x, num_heads, dim_per_head):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, num_heads, dim_per_head)
        return x.permute(0, 2, 1, 3)

    def generate_mask(self, mask):
        pass


    def attention(self, K, Q, V, need_mask):
        if need_mask:
            pass
        else:
            weight = F.softmax(torch.matmul(Q, K.permute(0, 1, 3, 2))/np.power(K.shape[-1], 0.5), dim=-1)
            output = torch.matmul(weight, V)

            return output


class FeedForwardLayer(nn.Module):
    def __init__(self, f_dim, ffn_dim, dropout_rate=0.5):
        super(FeedForwardLayer, self).__init__()
        self.f_dim = f_dim
        self.ffn1 = nn.Linear(f_dim, ffn_dim)
        # initialize_weight(self.ffn1)
        self.dropout = nn.Dropout(dropout_rate)
        # self.ffn2 = nn.Linear(ffn_dim, ffn_dim)
        # # initialize_weight(self.ffn2)
        # self.ffn3 = nn.Linear(ffn_dim, ffn_dim)
        # initialize_weight(self.ffn3)

        self.ffn6 = nn.Linear(ffn_dim, f_dim)
        # initialize_weight(self.ffn6)

        self.geglu = GEGLU()


    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size*seq_len, self.f_dim)

        x = F.relu(self.dropout(self.ffn1(x)))
        # x = F.relu(self.dropout(self.ffn2(x)))
        # x = F.relu(self.dropout(self.ffn3(x)))

        output = F.relu(self.dropout(self.ffn6(x)))
        output = output.view(batch_size, seq_len, self.f_dim)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, max_seq_len, f_dim, dim_per_head, num_heads, ffn_dim, dropout_rate):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAtten(f_dim=f_dim, atten_layer_type='Linear',
                                        dim_per_head=dim_per_head, n_heads=num_heads)
        self.LayerNorm = nn.LayerNorm([max_seq_len, f_dim])
        self.FeedForwardLayer = FeedForwardLayer(f_dim, ffn_dim=ffn_dim, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        atten_x = self.attention(x)
        resnorm_x = self.dropout(self.LayerNorm(atten_x))
        ffn_x = self.dropout(self.FeedForwardLayer(resnorm_x)) + x

        return ffn_x

class Encoder(nn.Module):
    """
    add the encoder linear to control the hidden dimension in transformer
    remove the LayerNorm operation before the classifier
    investigate: using the combine_head layer / remove the combine_layer
    """
    def __init__(self, max_seq_len=251, f_dim=60, dim_per_head=64, num_heads=8, ffn_dim=256, dropout_rate=0.1, num_layers=6):
        super(Encoder, self).__init__()

        # self.PositionEcoder = PositionEncoder(f_dim=f_dim, max_seq_len=max_seq_len, dropout_rate=dropout_rate)


        self.EncodingLayer = EncoderLayer(max_seq_len=max_seq_len,
                                          f_dim=f_dim, dim_per_head=dim_per_head, num_heads=num_heads,
                                          ffn_dim=ffn_dim, dropout_rate=dropout_rate)
        self.LayerNorm = nn.LayerNorm([max_seq_len, f_dim])
        self.Encoders = nn.ModuleList([self.EncodingLayer for _ in range(num_layers)])

        # ffn for classification of emotion and gender

        self.emotion_ffn = nn.Linear(in_features=max_seq_len*f_dim, out_features=4)
        self.emotion_ffn_center = nn.Linear(in_features=max_seq_len*f_dim, out_features=4)
        self.gender_ffn = nn.Linear(in_features=max_seq_len*f_dim, out_features=2)
        self.gender_ffn_center = nn.Linear(in_features=max_seq_len*f_dim, out_features=2)

    def forward(self, x):
        # x_local = self.CNN(x)

        x = x.transpose(2,1)

        # weights = []
        # x = self.encoder_linear(x)
        # pos_x = self.PositionEcoder(x)
        # x = pos_x
        for encoder in self.Encoders:
            x = encoder(x)
            # weights.append(weight.cpu().detach().numpy())

        x = self.LayerNorm(x).view(x.shape[0], -1)


        # x = torch.cat((x_local, x), dim=-1)
        # classifier
        emotion = self.emotion_ffn(x)
        emotion_center = self.emotion_ffn_center(x)
        gender = self.gender_ffn(x)
        gender_center = self.gender_ffn_center(x)

        #

        return emotion, gender, emotion_center, gender_center

class cleanEncoder(nn.Module):
    def __init__(self, max_seq_len, f_dim, dim_per_head, num_heads, ffn_dim, dropout_rate, num_layers):
        super(cleanEncoder, self).__init__()

        # self.PositionEcoder = PositionEncoder(f_dim=f_dim, max_seq_len=max_seq_len, dropout_rate=dropout_rate)
        self.EncodingLayer = EncoderLayer(max_seq_len=max_seq_len,
                                          f_dim=f_dim, dim_per_head=dim_per_head, num_heads=num_heads,
                                          ffn_dim=ffn_dim, dropout_rate=dropout_rate)

        self.Encoders = nn.ModuleList([self.EncodingLayer for _ in range(num_layers)])

    def forward(self, x):
        x = x.transpose(2,1)
        # pos_x = self.PositionEcoder(x)
        # x = pos_x
        # weights = []

        for encoder in self.Encoders:
            x = encoder(x)

            # weights.append(weight.cpu().detach().numpy())

        return x


if __name__ == "__main__":
    x = torch.randn(32, 60, 260)
    # model = Encoder(max_seq_len=260, f_dim=60, dim_per_head=60, num_heads=8, ffn_dim=2048, dropout_rate=0.1, num_layers=6)
    # emotion,  emotion_center, gender, gender_center = model(x)
    x = fourier_encode(x, 15)
    print(x.shape)
    b, *axis, _, device = x.shape

    axis_pos = list(map(lambda size: torch.linspace(-1., 1.,
                                                    steps=size), (60)))
    print(len(axis_pos))
    pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
    print('pos={}'.format(pos.shape))  # pos=torch.size([224, 224, 2])
