#

import glob
from tqdm import tqdm
import os

import random

from silence_delete import delete_silence
from feature import feature_extractor

import numpy as np
import librosa as rosa
import pickle



def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def random_split(path, test_cluster, data_aug):
    """split train and test datasets randomly"""
    wav_files = glob.glob(os.path.join(path, r'wav\*.wav'), recursive=True)

    n = len(wav_files)
    cluster1 = list(np.random.choice(range(n), int(n * 0.2), replace=False))
    rest = list(set(range(n)) - set(cluster1))
    cluster2 = list(np.random.choice(rest, int(n * 0.2), replace=False))
    rest = list(set(rest) - set(cluster2))
    cluster3 = list(np.random.choice(rest, int(n * 0.2), replace=False))
    rest = list(set(rest) - set(cluster3))
    cluster4 = list(np.random.choice(rest, int(n * 0.2), replace=False))
    rest = list(set(rest) - set(cluster4))
    cluster5 = rest

    indices_dict = {
        'cluster1': cluster1,
        'cluster2': cluster2,
        'cluster3': cluster3,
        'cluster4': cluster4,
        'cluster5': cluster5,
    }
    train_data_dict = {}
    valid_data_dict = {}
    for n in range(test_cluster):
        valid_indices = []
        train_indices = []
        train_files = []
        valid_files = []
        train_wav_name = []
        for k in indices_dict.keys():
            if k[-1] == str(n+1):
                valid_indices.extend(indices_dict[k])
            else:
                train_indices.extend(indices_dict[k])

        for i in train_indices:
            train_files.append(wav_files[i])
            train_wav_name.append(os.path.basename(wav_files[i]))
        for i in valid_indices:
            valid_files.append(wav_files[i])

        # load the aug_files
        if data_aug == True:
            aug_wav_files = glob.glob(os.path.join(path, r'aug_wav\*.wav'), recursive=True)

            for file in aug_wav_files:
                if os.path.basename(file)[2:] in train_wav_name:
                    train_files.append(file)

        train_data_dict[str(n+1)] = train_files
        valid_data_dict[str(n+1)] = valid_files

    print(train_data_dict.keys())

    return train_data_dict, valid_data_dict

def speaker_split(path, test_cluster,  data_aug):
    """split train and test datasets according to speakers"""
    wav_files = glob.glob(os.path.join(path, r'wav\*.wav'), recursive=True)

    train_files = []
    train_wav_name = []
    valid_files = []

    for file in wav_files:
        if os.path.basename(file)[4] == str(test_cluster):
            valid_files.append(file)
        else:
            train_files.append(file)
            train_wav_name.append(os.path.basename(file))

    if data_aug == True:
        aug_wav_files = glob.glob(os.path.join(path, r'aug_wav\*\*.wav'), recursive=True)

        for file in aug_wav_files:
            if os.path.basename(file)[2:] in train_wav_name:
                train_files.append(file)

    return train_files, valid_files

def get_num_label(emo_label_str, sex_label_str, emo_num, sex_num):

    # get the number of label of emotion
    emo_str2text = {
        'N': 'neutral',
        'S': 'sad',
        'A': 'angry',
        'H': 'happy',
    }

    emo_text2num = {
        'neutral': 0,
        'sad': 1,
        'angry': 2,
        'happy': 3
    }
    emo = emo_str2text[emo_label_str]
    emo_label = emo_text2num[emo]
    emo_num[emo] += 1

    # get the number of label of sex
    sex_dict = {
        'F': 0,
        'M': 1
    }
    sex_label = sex_dict[sex_label_str]
    sex_num[sex_label_str] += 1

    return emo_label, sex_label


def get_data(files, fixed_t, silence_delete, feature_to_use):
    """
    get the processed wav data from the original wav data
    such as: silence delete; fixed length processing etc...
    """
    emo_num = {
        'neutral':0,
        'happy':0,
        'sad':0,
        'angry':0
    }
    sex_num = {
        'F':0,
        'M':0
    }

    mfccs = []
    delta = []
    delta_delta = []
    emo_labels = []
    sex_labels = []
    for i, wav_file in enumerate(tqdm(files)):

        # delete the un-renamed wav_file
        if len(os.path.basename(wav_file).split('-')) == 1:
            continue

        # get the label
        emo_str = str(os.path.basename(wav_file).split('-')[2][-1])


        sex_str = os.path.basename(wav_file).split('-')[3][0]

        emo_label, sex_label = get_num_label(emo_str, sex_str, emo_num, sex_num)

        # read wav data
        wav_data, sr = rosa.load(wav_file, sr=16000)

        # delete the silence parts if 'silence_delete == True'
        if silence_delete == True:
            wav_data, _, _ = delete_silence(wav_data)


        if (fixed_t * sr >= len(wav_data)):
            wav_data = list(wav_data)
            wav_data.extend(np.zeros(int(fixed_t * sr - len(wav_data))))
            wav_data = np.array(wav_data)
            mfcc = feature_extractor(wav_data, sr=sr, feature_to_use=feature_to_use, n_mels=60)

            if mfcc.shape!=(60, 251):
                continue
            mfccs.append(mfcc)
            delta.append(rosa.feature.delta(mfcc))
            delta_delta.append(rosa.feature.delta(rosa.feature.delta(mfcc)))
        else:
            wav_data = list(wav_data[:int(fixed_t * sr)])
            wav_data = np.array(wav_data)
            mfcc = feature_extractor(wav_data, sr=sr, feature_to_use=feature_to_use, n_mels=60)
            if mfcc.shape!=(60, 251):
                continue
            mfccs.append(mfcc)
            delta.append(rosa.feature.delta(mfcc))
            delta_delta.append(rosa.feature.delta(rosa.feature.delta(mfcc)))

        emo_labels.append(emo_label)
        sex_labels.append(sex_label)


    mfccs = np.array(mfccs)
    delta = np.array(delta)
    delta_delta = np.array(delta_delta)
    emo_labels = np.array(emo_labels)
    sex_labels = np.array(sex_labels)
    return mfccs, delta, delta_delta, emo_labels, sex_labels

def process_IEMOCAP(path, seed=987654, fixed_t=6, test_cluster=1, split_method='random', silence_delete=True, data_aug=False, feature_to_use='mfcc'):

    setup_seed(seed)

    # split wav_files into train_files and valid_files
    if split_method == 'random':
        train_data_dict, valid_data_dict = random_split(path, test_cluster=test_cluster, data_aug=data_aug)

    # elif split_method == 'speaker':
    #     train_files, valid_files = speaker_split(path, test_cluster=test_cluster, data_aug=data_aug)
    else:
        print("'split_method' = 'random' or 'speaker'!")
        raise ValueError

    for i in range(test_cluster):
        # get the train and valid data
        train_mfcc, train_delta, train_delta_delta, train_emo, train_sex = get_data(files=train_data_dict[str(i+1)], fixed_t=fixed_t, silence_delete=silence_delete, feature_to_use=feature_to_use)
        valid_mfcc, valid_delta, valid_delta_delta, valid_emo, valid_sex = get_data(files=valid_data_dict[str(i+1)], fixed_t=fixed_t, silence_delete=silence_delete, feature_to_use=feature_to_use)

        data = {
            'train_x': train_mfcc,
            'train_delta': train_delta,
            'train_delta_delta': train_delta_delta,
            'train_emo': train_emo,
            'train_sex': train_sex,

            'valid_mfcc': valid_mfcc,
            'valid_delta': valid_delta,
            'valid_delta_delta': valid_delta_delta,
            'valid_emo': valid_emo,
            'valid_sex': valid_sex
        }

        # save the data

        save_path = '.\data\\MSP_leave_' + str(i+1) + '_data.pkl'

        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

        print('train_mfcc:', train_mfcc.shape)
        print('train_delta:', train_delta.shape)
        print('train_delta_delta:', train_delta_delta.shape)
        print('train_emo: ', train_emo.shape)
        print('train_sex: ', train_sex.shape)

        print('valid_mfcc:', valid_mfcc.shape)
        print('valid_delta:', valid_delta.shape)
        print('valid_delta_delta:', valid_delta_delta.shape)
        print('valid_emo: ', valid_emo.shape)
        print('valid_sex: ', valid_sex.shape)

if __name__ == '__main__':

    process_IEMOCAP(r'K:\MSP_IMPROV', test_cluster=5, split_method='random', silence_delete=True, data_aug=False)














