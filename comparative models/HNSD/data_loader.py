import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

dict = {
    0: torch.Tensor([0]),
    1: torch.Tensor([1]),
    2: torch.Tensor([2]),
    3: torch.Tensor([3]),
}


class DataSet(Dataset):
    def __init__(self, mfcc, delta, delta_delta,  Y, sex):
        self.mfcc = mfcc
        self.delta = delta
        self.delta_delta = delta_delta

        self.Y = Y
        self.sex = sex

    def __getitem__(self, index):
        mfcc = self.mfcc[index]

        mfcc = torch.from_numpy(mfcc.astype(np.float32))
        mfcc = mfcc.float()

        delta = self.delta[index]

        delta = torch.from_numpy(delta.astype(np.float32))
        delta = delta.float()

        delta_delta = self.delta_delta[index]

        delta_delta = torch.from_numpy(delta_delta.astype(np.float32))
        delta_delta = delta_delta.float()




        y = self.Y[index]
        y = dict[y]
        y = y.long()

        sex = self.sex[index]
        sex = dict[sex]
        sex = sex.long()


        return mfcc, delta, delta_delta, y, sex

    def __len__(self):
        return len(self.mfcc)

# if __name__=='__main__':
#     import pickle
#     file = r'../processing/features_mfcc_all.pkl'
#     with open(file, 'rb') as f:
#         features = pickle.load(f)
#
#     val_X_f = features['val_X_f']
#     val_X_t = features['val_X_t']
#     val_y = features['val_y']
#     val_sex = features['val_sex']
#
#     val_data = DataSet(val_X_f, val_X_t, val_y, val_sex)
#     val_loader = DataLoader(val_data, batch_size=10, shuffle=True)
#     for i ,data in enumerate(val_loader):
#         x_f, x_t, y, sex = data
#         print(y)
#         print(sex)
#         print(val_X_f.shape)
#         print(val_X_t.shape)
#         break

