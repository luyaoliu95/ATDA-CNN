import numpy as np
import librosa as rosa
import matplotlib.pyplot as plt
'''
max_time: 25.888
min_time: 0.274
mean_time: 3.2899402033087504
'''

def fixed_length(wav, sr, fixed_time=2, stride_time=1):
    '''
    将每个utterance裁剪至2s的固定长度的utterance;
    小于2s在后补零
    '''
    time = len(wav)/sr
    print('time:', time)
    if time <= fixed_time:
        new_wav = wav.extend([0]*int(sr*(fixed_time-time)))
    else:
        start = 0
        new_wav = []
        n = int((time-fixed_time)/stride_time)+1
        print(n)
        for i in range(n):
            new_wav.append(wav[start:int(fixed_time*sr)])
            start = start+stride_time

    return new_wav

def sqrt_hann(y):
    """
    :param y:
    :return:signal after cover hanning window
    """
    return np.sqrt(np.hanning(y))


def enframe(y, frame_length=2048, hop_length=512, winc='hanning',  center=True):
    """
    :param y: time sequence
    :param frame_lenth: the length of a frame
    :param hop_length: the length of windows move
    :param center: if set center as True, padding time sequence to a proper length
    :return: framed sequence
    """
    if center==True:
        y = np.hstack((np.zeros(int(frame_length / 2)), y, np.zeros(int(frame_length / 2))))
    length_y = len(y)
    n_f = int(np.ceil((1.0 * length_y - frame_length) / hop_length))
    pad_length = int((n_f-1)*hop_length + frame_length)  #平铺后的长度

    indices = np.tile(np.arange(0, frame_length), (n_f, 1)) + np.tile(np.arange(0, n_f*hop_length, hop_length), (frame_length, 1)).T  # 每帧的索引
    indices = np.array(indices, dtype=np.int32)

    frames = y[indices]  # 得到帧信号, 用索引拿数据
    if winc == 'hanning':
        win = sqrt_hann(frame_length)
        return win*frames
    else:
        return frames




def delete_silence(y, frame_length=2048, hop_length=512, center=True):
    # 计算短时过零率
    zcr = rosa.feature.zero_crossing_rate(y,frame_length=frame_length, hop_length=hop_length)
    zcr = zcr[0]   #all zcrs of frames of the utterance
    # 计算短时能量
    frames = enframe(y,frame_length=frame_length, hop_length=hop_length)
    amp = np.array(list(abs(frames).sum(axis=1)))

    #设置过零率的上下限

    zcr_high = max(max(zcr)*0.8, min(zcr)*2)   #过零率高门限

    #设置短时能量门限
    amp_low = min([min(amp)*10, max(amp)*0.05+min(amp)*5.0, max(amp)*0.1])  #短时能量低门限

    # 判断各帧的是否为静音
    silence_label=np.zeros(zcr.shape[0])


    if zcr.shape[0] != amp.shape[0]:
        print('zcr!=amp')

        if zcr.shape[0] > amp[0]:
            zcr = zcr[:amp.shape[0]]

        else:
            amp = amp[:zcr.shape[0]]

        if amp.shape[0]!=zcr.shape[0]:
            return None, None, None


        print('zcr', zcr.shape)
        print('amp', amp.shape)


    for i in range(zcr.shape[0]):
        if zcr[i]>=zcr_high or amp[i]<=amp_low:
            silence_label[i]=1

    #print(silence_label)

    speech_start = []
    speech_end = []
    for i in range(len(silence_label)-1):
        if i==0 and silence_label[i] == 0:
            speech_start.append(i)
        if i==len(silence_label)-1 and silence_label[i] == 0:
            speech_end.append(i)
        if silence_label[i] == 1 and silence_label[i+1] == 0:
            speech_start.append(i)
        if silence_label[i] == 0 and silence_label[i+1] == 1:
            speech_end.append(i)

    speech_start = (speech_start-np.ones(len(speech_start)))*hop_length
    speech_end = (speech_end-np.ones(len(speech_end)))*hop_length + np.ones(len(speech_end))*frame_length

    # delete silence
    y_without_silence = []
    if len(speech_start) == 0 or len(speech_end) == 0:
        return y, speech_start, speech_end
    elif len(speech_end) != len(speech_start):
        speech_end = np.append(speech_end, (len(y)-1))


    if len(speech_start) == 1:
        if speech_start[0] == -512:
            speech_start[0] = 0
        if speech_end[0] > len(y):
            speech_end[0] = len(y) - 1
        y_without_silence.extend(y[int(speech_start[0]):int(speech_end[0])])


    for i in range(len(speech_start)-1):
        if speech_start[i] == -512:
            speech_start[i] = 0
        if speech_end[i] > len(y):
            speech_end[i] = len(y) - 1
        if speech_end[i] >= speech_start[i+1]:

            speech_start[i+1] = speech_start[i]
            continue
        else:

            y_without_silence.extend(y[int(speech_start[i]):int(speech_end[i])])


    y_without_silence.extend(y[int(speech_start[-1]):int(speech_end[-1])])



    return y_without_silence,speech_start, speech_end


# file = '/home/liuluyao/IEMOCAP/wav/Ses01F-impro01-01-F-F005.wav'
# y, sr = rosa.load(file,sr=None)
#
# y_without_silence, speech_start, speech_end = delete_silence(y)
#
# print('silence_label:', len(y_without_silence)/sr, len(y)/sr)
# print('speech_start:', speech_start)
# print('speech_end:', speech_end)
#
# plt.plot(y)
# plt.vlines(speech_start, -0.6,0.6,'r','--',label='start')
# plt.vlines(speech_end,-0.5,0.5,'b','-',label='end')
# plt.show()
#
# print('time:', len(y)/sr)
# # #mfccs = np.array([rosa.feature.mfcc(ys, sr,n_mfcc=128).T for ys in np.array(y_without_silence)])
# #         #print(mfccs.shape)
# # print(np.array(y))
# mfccs = rosa.feature.mfcc(y, sr, n_mfcc=13).T
#
# delete_index = np.zeros(mfccs.shape[0], dtype=bool)
#
# for i in range(mfccs.shape[0]):
#     if silence_label[i] != 1:
#         delete_index[i] = True
# mfccs = mfccs[delete_index]
# """ 2s长音频对应mfccs[0]=63"""
# print('without:', mfccs.shape)
# n = int(mfccs.shape[0]/63)
# print(mfccs[0:63].reshape([1, 63*13]))
#
#
# mfccs_2s = np.zeros([1, 63*13])
# start = 0
# for i in range(n):
#     mfccs_2s = np.concatenate((mfccs_2s, mfccs[start:int(start + 63)].reshape([1, 63*13])))
#     start = int(start + 63)
# mfccs_2s = mfccs_2s[1:]
# print(mfccs_2s.shape)
#
# print(silence_label.shape)
# print(np.array(y_without_silence).shape)


if __name__=='__main__':
    import glob
    path = r"K:\IEMOCAP\aug_wav"

    wav_files = glob.glob(path + '\*.wav')
    time = []
    for i in wav_files:
        print(i)
        wav_data,sr = rosa.load(i, sr=None)
        without_silence_wav_data, _,__ = delete_silence(wav_data)
        time.append(round(len(without_silence_wav_data)/sr,3))
    print('time:',time)
    print('max_time:', np.max(time))
    print('min_time:', np.min(time))
    print('mean_time:', np.mean(time))