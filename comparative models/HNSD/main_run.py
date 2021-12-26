
import numpy as np
import os
import pickle
from train_Joint_Loss import train

datasets = ['MELD']

def train_HNSD():
    for dataset in datasets:
        out_root = r'/home/liuluyao/comparison_results/'
        UA = []
        WA = []
        matrixs = []
        for i in range(5):
            maxWA, maxUA, best_matrix = train(dataset=dataset, data_num=i+1)
            WA.append(maxWA)
            UA.append(maxUA)
            matrixs.append(best_matrix)

        results = {
                'WA': WA,
                'UA': UA,
                'matrixs':matrixs,
                'mean_WA': np.mean(np.array(WA)),
                'mean_UA': np.mean(np.array(UA))
            }
        with open(os.path.join(out_root, dataset + '.pkl'), 'wb') as f:
            pickle.dump(results, f)
        print('UA: ', UA)
        print('mean_UA: ', np.mean(np.array(UA)))
        print('WA: ', WA)
        print('mean_WA: ', np.mean(np.array(WA)))

if __name__ == '__main__':
    train_HNSD()