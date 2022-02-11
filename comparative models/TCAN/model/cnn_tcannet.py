
import torch
import torch.nn as nn
import torch.nn.functional as F
from CNN import CNN_only
from tcanet import TCANet

class CNN_TCAN(nn.Module):

    def __init__(self, emb_size, input_output_size, num_channels, seq_len, num_sub_blocks, temp_attn, nheads, en_res,
                 conv, key_size, kernel_size=2, dropout=0.3, wdrop=0.0, emb_dropout=0.1, tied_weights=False,
                 dataset_name=None, visual=True):
        super(CNN_TCAN, self).__init__()

        self.cnn = CNN_only()

        self.tcan = TCANet(emb_size, input_output_size, num_channels, seq_len, num_sub_blocks, temp_attn, nheads, en_res,
                 conv, key_size, kernel_size, dropout, wdrop, emb_dropout, tied_weights,
                 dataset_name, visual)

        self.pool = nn.AdaptiveAvgPool2d((1, input_output_size))

    def forward(self, x):
        x_cnn = self.cnn(x)
        x = self.tcan(x_cnn)

        x = x[0]

        x = self.pool(x).view(x.shape[0], -1)

        return x

class CNN_TCAN_Joint_Loss(nn.Module):

    def __init__(self, emb_size, input_output_size, num_channels, seq_len, num_sub_blocks, temp_attn, nheads, en_res,
                 conv, key_size, kernel_size=2, dropout=0.3, wdrop=0.0, emb_dropout=0.1, tied_weights=False,
                 dataset_name=None, visual=True):
        super(CNN_TCAN_Joint_Loss, self).__init__()

        self.in_features = input_output_size

        self.cnn = CNN_only()

        self.tcan = TCANet(emb_size, input_output_size, num_channels, seq_len, num_sub_blocks, temp_attn, nheads, en_res,
                 conv, key_size, kernel_size, dropout, wdrop, emb_dropout, tied_weights,
                 dataset_name, visual)

        self.emotion_ffn = nn.Linear(in_features=self.in_features, out_features=4)

        self.emotion_ffn_center = nn.Linear(in_features=self.in_features, out_features=4)

        self.gender_ffn = nn.Linear(in_features=self.in_features, out_features=2)

        self.gender_ffn_center = nn.Linear(in_features=self.in_features, out_features=2)

        # self.tcan_emotion_center = TCANet(emb_size, input_output_size, num_channels, seq_len, num_sub_blocks, temp_attn,
        #                            nheads, en_res,
        #                            conv, key_size, kernel_size, dropout, wdrop, emb_dropout, tied_weights,
        #                            dataset_name, visual)
        #
        # self.tcan_gender = TCANet(emb_size, 2, num_channels, seq_len, num_sub_blocks, temp_attn, nheads, en_res,
        #          conv, key_size, kernel_size, dropout, wdrop, emb_dropout, tied_weights,
        #          dataset_name, visual)
        #
        # self.tcan_gender_center = TCANet(emb_size, 2, num_channels, seq_len, num_sub_blocks, temp_attn, nheads, en_res,
        #                           conv, key_size, kernel_size, dropout, wdrop, emb_dropout, tied_weights,
        #                           dataset_name, visual)

        self.pool = nn.AdaptiveAvgPool2d((1, input_output_size))
        self.pool_gender = nn.AdaptiveAvgPool2d((1, 2))

    def forward(self, x):
        x_cnn = self.cnn(x)
        x_emotion = self.tcan(x_cnn)

        x_emotion = x_emotion[0]

        x = self.pool(x_emotion).view(x_emotion.shape[0], -1)

        # classifier
        emotion = self.emotion_ffn(x)
        emotion_center = self.emotion_ffn_center(x)
        gender = self.gender_ffn(x)
        gender_center = self.gender_ffn_center(x)


        return emotion, gender, emotion_center, gender_center

if __name__ == '__main__':
    from config import Config

    args_list = [
        Config(optim='Adam', key_size=300, lr=1e-4, epochs=1, gpu_id=2, num_subblocks=0, levels=4, en_res=False,
              temp_attn=True, seq_len=96, valid_len=40, conv=False, visual=False,
              log="tcanet_num_subblocks-1_levels-4_conv-False") ]

    x = torch.randn(32, 60, 251)

    args = args_list[0]


    model = CNN_TCAN_Joint_Loss(args.emsize, 512, [args.nhid] * args.levels, args.valid_len, args.num_subblocks, temp_attn=args.temp_attn,
                   nheads=args.nheads,
                   en_res=args.en_res, conv=args.conv, dropout=args.dropout, emb_dropout=args.emb_dropout,
                   key_size=args.key_size,
                   kernel_size=args.ksize, tied_weights=args.tied, dataset_name=args.dataset_name, visual=args.visual)

    out = model(x)

