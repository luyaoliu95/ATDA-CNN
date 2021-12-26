
import librosa as rosa
import numpy as np

def feature_extractor(x, sr, feature_to_use, win_length=768, hop_length=384, **kwargs):

    if feature_to_use == 'mfcc':
        if 'n_mfcc' in kwargs.keys():
            n_mfcc = kwargs['n_mfcc']
        else:
            n_mfcc = 60
        return rosa.feature.mfcc(x, sr=sr, n_mfcc=n_mfcc, win_length=win_length,
                                             hop_length=hop_length)

    elif feature_to_use == 'melspectrogram':
        if 'n_mels' in kwargs.keys():
            n_mels = kwargs['n_mels']
        else:
            n_mels = 128

        n_fft = win_length
        return rosa.feature.melspectrogram(x, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    else:
        print("'feature_to_use' = 'mfcc' or 'melspectrogram'! ")
        raise ValueError

