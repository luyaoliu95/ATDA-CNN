
from moviepy.editor import *
import numpy as np
import os
from glob import glob
import pandas as pd
import pickle
import librosa as rosa
from tqdm import tqdm
from silence_delete import delete_silence
from feature import feature_extractor
import random

"""
test:  
{'Chandler': 379, 'Joey': 411, 'Rachel': 356, 'Monica': 346, 'Phoebe': 291, 'Ross': 373} 
{'neutral': 994, 'sadness': 173, 'anger': 303, 'joy': 337, 'surprise': 238, 'fear': 46, 'disgust': 65}

{'neutral': 994, 'joy': 337, 'sadness': 173, 'anger': 303}
{'F': 825, 'M': 982}

dev:  
{'Chandler': 101, 'Joey': 149, 'Rachel': 164, 'Monica': 137, 'Phoebe': 185, 'Ross': 217}
{'neutral': 396, 'sadness': 95, 'anger': 140, 'joy': 134, 'surprise': 129, 'fear': 38, 'disgust': 21}

{'neutral': 395, 'joy': 134, 'sadness': 95, 'anger': 140}
{'F': 384, 'M': 380}

train:  
{'Chandler': 1283, 'Joey': 1510, 'Rachel': 1435, 'Monica': 1299, 'Phoebe': 1321, 'Ross': 1458}
{'neutral': 3820, 'sadness': 587, 'anger': 954, 'joy': 1449, 'surprise': 1034, 'fear': 226, 'disgust': 236}

{'neutral': 1501, 'joy': 1449, 'sadness': 587, 'anger': 954}
{'F': 2176, 'M': 2315}
"""

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Convert movie into audio
def Mp4toWav(mp4_dir, wav_dir):
    MP4s = glob(mp4_dir, recursive=True)
    i = 0
    for MP4 in MP4s:

        mp4_name = os.path.basename(MP4)[:-4]

        video = VideoFileClip(MP4)
        audio = video.audio
        audio.write_audiofile(os.path.join(wav_dir, mp4_name+'.wav'))
        i +=1
    print(i)

speaker_gender_dict = {
    'Chandler':'M',
    'Joey':'M',
    'Rachel':'F',
    'Monica':'F',
    'Phoebe':'F',
    'Ross':'M'
}

emo_dict = {
    'neutral': 0,
    'sadness': 1,
    'anger': 2,
    'joy': 3
}

gender_label = {
    'M':0,
    'F':1
}

emo_num = {
    'neutral': 0,
    'sadness': 0,
    'anger': 0,
    'joy': 0,
    'surprise':0,
    'fear':0,
    'disgust':0
}

speaker_num = {
    'Chandler':0,
    'Joey':0,
    'Rachel':0,
    'Monica':0,
    'Phoebe':0,
    'Ross':0
}

def calculator(label_file):
    # calculate the number of data
    speaker_num_dict = speaker_num
    emo_num_dict = emo_num
    data = pd.read_csv(label_file)
    speakers = data['Speaker']
    emotions = data['Emotion']

    for i, s in enumerate(speakers):
        if s in speaker_num_dict.keys():
            speaker_num_dict[s] +=1
            emo_num_dict[emotions[i]] +=1

    print('speaker_number: ', speaker_num_dict)
    print('emotion_number: ', emo_num_dict)

    return speaker_num_dict, emo_num_dict


# get the labels of emotion and gender
def process_label_file(label_file):
    label_data = pd.read_csv(label_file)[['Speaker', 'Emotion', 'Dialogue_ID', 'Utterance_ID']]
    label_dict = {}
    for index, row in label_data.iterrows():
        if row['Speaker'] not in speaker_gender_dict.keys():
            continue
        if row['Emotion'] not in emo_dict.keys():
            continue
        meta={}
        meta['speaker'] = row['Speaker']
        meta['gender_label'] = speaker_gender_dict[row['Speaker']]
        meta['emotion'] = row['Emotion']
        meta['emo_label'] = emo_dict[row['Emotion']]
        label_dict['dia'+str(row['Dialogue_ID'])+'_utt'+str(row['Utterance_ID'])] = meta

    return label_dict


def get_data(wav_files, fixed_t, silence_delete, feature_to_use, label_file, file_type):
    """
    get the processed wav data from the original wav data
    such as: silence delete; fixed length processing etc...
    """

    # read the label dict
    labels_dict = process_label_file(label_file)

    emo_num = {
        'neutral':0,
        'joy':0,
        'sadness':0,
        'anger':0
    }
    sex_num = {
        'F':0,
        'M':0
    }

    mfccs = []
    emo_labels = []
    sex_labels = []
    for i, wav_file in enumerate(tqdm(wav_files)):

        wav_name = os.path.basename(wav_file)[:-4]
        if wav_name not in labels_dict.keys():
            continue

        # labels
        emo_label = labels_dict[wav_name]['emo_label']
        sex_label = gender_label[labels_dict[wav_name]['gender_label']]

        if emo_num['neutral'] > 1500 and labels_dict[wav_name]['emotion']=='neutral' and file_type=='train':
            continue

        # read wav data
        wav_data, sr = rosa.load(wav_file, sr=16000)

        # delete the silence parts if 'silence_delete == True'
        if silence_delete == True:
            wav_data, _, _ = delete_silence(wav_data)
            if wav_data is None:
                continue

        # get the wav data with a fixed length
        if emo_label == 1:
            for i in range(1):
                if (fixed_t*sr >= len(wav_data)):
                    wav_data = list(wav_data)
                    wav_data.extend(np.zeros(int(fixed_t*sr-len(wav_data))))
                    wav_data = np.array(wav_data)
                    mfcc = feature_extractor(wav_data, sr=sr, feature_to_use=feature_to_use, n_mfcc=60)
                    mfccs.append(mfcc)
                else:
                    wav_data = list(wav_data[:int(fixed_t*sr)])
                    wav_data = np.array(wav_data)
                    mfcc = feature_extractor(wav_data, sr=sr, feature_to_use=feature_to_use, n_mfcc=60)
                    mfccs.append(mfcc)


                emo_labels.append(emo_label)
                sex_labels.append(sex_label)

                emo_num[labels_dict[wav_name]['emotion']] +=1
                sex_num[labels_dict[wav_name]['gender_label']] +=1
        else:
            if (fixed_t * sr >= len(wav_data)):
                wav_data = list(wav_data)
                wav_data.extend(np.zeros(int(fixed_t * sr - len(wav_data))))
                wav_data = np.array(wav_data)
                mfcc = feature_extractor(wav_data, sr=sr, feature_to_use=feature_to_use, n_mfcc=60)
                mfccs.append(mfcc)

            else:
                wav_data = list(wav_data[:int(fixed_t * sr)])
                wav_data = np.array(wav_data)
                mfcc = feature_extractor(wav_data, sr=sr, feature_to_use=feature_to_use, n_mfcc=60)
                mfccs.append(mfcc)

            emo_labels.append(emo_label)
            sex_labels.append(sex_label)

            emo_num[labels_dict[wav_name]['emotion']] += 1
            sex_num[labels_dict[wav_name]['gender_label']] += 1

    print(emo_num)
    print(sex_num)
    mfccs = np.array(mfccs)
    emo_labels = np.array(emo_labels)
    sex_labels = np.array(sex_labels)
    return mfccs, emo_labels, sex_labels

def process_MELD(root_path, seed=987654, fixed_t=6, silence_delete=True, feature_to_use='mfcc', test_cluster=5):

    setup_seed(seed)

    train_wav_files = glob(os.path.join(root_path, 'train_speech\*.wav'), recursive=True)
    train_label_file = os.path.join(root_path, 'train_sent_emo.csv')
    dev_wav_files = glob(os.path.join(root_path, 'dev_speech\*.wav'), recursive=True)
    dev_label_file = os.path.join(root_path, 'dev_sent_emo.csv')
    test_wav_files = glob(os.path.join(root_path, 'test_speech\*.wav'), recursive=True)
    test_label_file = os.path.join(root_path, 'test_sent_emo.csv')


    train_x, train_emo, train_sex = get_data(train_wav_files, fixed_t, silence_delete, feature_to_use, train_label_file, 'train')
    dev_x, dev_emo, dev_sex = get_data(dev_wav_files, fixed_t, silence_delete, feature_to_use, dev_label_file, 'dev')
    valid_x, valid_emo, valid_sex = get_data(test_wav_files, fixed_t, silence_delete, feature_to_use, test_label_file, 'test')

    all_x = np.concatenate((train_x, dev_x, valid_x), axis=0)
    all_emo = np.concatenate((train_emo, dev_emo, valid_emo), axis=0)
    all_sex= np.concatenate((train_sex, dev_sex, valid_sex), axis=0)

    n = len(all_x)
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

    for n in range(test_cluster):
        valid_indices = []
        train_indices = []
        train_x = []
        valid_x = []

        train_emo = []
        valid_emo = []

        train_sex = []
        valid_sex = []
        for k in indices_dict.keys():
            if k[-1] == str(n + 1):
                valid_indices.extend(indices_dict[k])
            else:
                train_indices.extend(indices_dict[k])

        for i in train_indices:
            train_x.append(all_x[i])
            train_emo.append(all_emo[i])
            train_sex.append(all_sex[i])

        for i in valid_indices:
            valid_x.append(all_x[i])
            valid_emo.append(all_emo[i])
            valid_sex.append(all_sex[i])

        data = {
            'train_x': train_x,
            'train_emo': train_emo,
            'train_sex': train_sex,

            'valid_x': valid_x,
            'valid_emo': valid_emo,
            'valid_sex': valid_sex
        }

        # save the data

        save_path = '..\data\without_VAD\MELD_data_{}.pkl'.format(n)

        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    # train_x = np.concatenate((train_x, dev_x), axis=0)
    # train_emo = np.concatenate((train_emo, dev_emo), axis=0)
    # train_sex = np.concatenate((train_sex, dev_sex), axis=0)
    #
    #
    # print('train_x: ', train_x.shape)
    # print('train_emo: ', train_emo.shape)
    # print('train_sex: ', train_sex.shape)
    #
    # print('dev_x: ', dev_x.shape)
    # print('dev_emo: ', dev_emo.shape)
    # print('dev_sex: ', dev_sex.shape)
    #
    # print('test_x: ', valid_x.shape)
    # print('test_emo: ', valid_emo.shape)
    # print('test_sex: ', valid_sex.shape)
    #
    #
    #




if __name__ == '__main__':
    # mp4_dir = r'K:\MELD.Raw\MELD.Raw\train\train_splits\*.mp4'
    # wav_dir = r'K:\MELD.Raw\MELD.Raw\train_speech'
    # Mp4toWav(mp4_dir, wav_dir)
    root_dir = r'K:\MELD.Raw\MELD.Raw'
    process_MELD(root_dir, seed=987654, fixed_t=6, silence_delete=False, feature_to_use='mfcc')
    # train_label_file = os.path.join(root_dir, 'train_sent_emo.csv')
    # dev_label_file = os.path.join(root_dir, 'dev_sent_emo.csv')
    # test_label_file = os.path.join(root_dir, 'test_sent_emo.csv')
    #
    # train_speaker_dict, train_emo_dict = calculator(train_label_file)
    # # dev_speaker_dict, dev_emo_dict = calculator(dev_label_file)
    # # test_speaker_dict, test_emo_dict = calculator(test_label_file)
    #
    # print('train: ', train_speaker_dict, train_emo_dict)
    # # print('dev: ', dev_speaker_dict, dev_emo_dict)
    # # print('test: ', test_speaker_dict, test_emo_dict)
