import librosa
import os
import numpy as np
import pickle
from python_speech_features import fbank

import config as c
from DB_reader import read_noisy_wav_structure


def convert_wav_to_feature(filename, feat_dir, mode):
    '''
    Converts the wav path to feat path
    ex) Input : ../011_16k/011c0201.wav
        Output : ../011_16k/011c0201.pkl
    '''

    filename_only = filename.split('/')[-1].replace('.wav', '.pkl')

    if mode == 'train' or mode == 'valid':
        output_folder_name = feat_dir  # c.TRAIN_FEAT_DIR or c.VALID_FEAT_DIR
    elif mode == 'test':
        output_folder_name = os.path.join(feat_dir, filename.split('/')[-2])  # Contain the Noise and SNR ex) airport_SNR-5

    output_file_name = os.path.join(output_folder_name, filename_only)
    # ex) ../Feature/MFB/train or valid or test(/440_16k)/440c0201.pkl

    return output_folder_name, output_file_name


def normalize_frame(m, scale=True):
    if scale:
        return (m-np.mean(m, axis=0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return m - np.mean(m, axis=0)


def extract_mfb(filename, feat_dir, mode, count):
    audio, sr = librosa.load(filename, sr=c.SR, mono=True)
    features, energies = fbank(signal=audio, samplerate=c.SR, nfilt=c.FILTER_BANK, winlen=0.025)

    if c.USE_LOGSCALE:
        features = 20 * np.log10(np.maximum(features, 1e-5))

    features = normalize_frame(features, scale=c.USE_SCALE)
    print(features.shape)  # features_shape : (# of frames, nfilt)

    output_folder_name, output_file_name = convert_wav_to_feature(filename, feat_dir, mode=mode)

    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    if os.path.isfile(output_file_name):
        print('\'' + '/'.join(output_file_name.split('/')[-3:]) + '\'' + 'file already extracted!')
    else:
        with open(output_file_name, 'wb') as fp:
            pickle.dump(features, fp)
            print('[%s]feature extraction (%s DB). step : %d, file : \'%s\''
                  % ('MFB', mode, count, '/'.join(filename.split('/')[-3:])))


class ModeError(Exception):
    def __str__(self):
        return "Wrong Mode (Type : 'train' or 'valid' or 'test')"


def feature_extraction(mode):
    if (mode != 'train') and (mode != 'test') and (mode != 'valid'):
        raise ModeError

    count = 1

    # _1.0 means that 1second length silence is padded on training and validation sets.
    # _0.0 means no manipulation is conducted on test sets
    if mode == 'train':
        wav_dir, feat_dir = c.TRAIN_WAV_DIR, c.TRAIN_FEAT_DIR + '_1.0'
    elif mode == 'valid':
        wav_dir, feat_dir = c.VALID_WAV_DIR, c.VALID_FEAT_DIR + '_1.0'
    else:
        wav_dir, feat_dir = c.TEST_WAV_DIR, c.TEST_FEAT_DIR + '_0.0'

    DB = read_noisy_wav_structure(wav_dir, mode)

    for i in range(len(DB)):
        filename = DB['filename'][i]
        extract_mfb(filename, feat_dir, mode, count)
        count = count + 1

    print('-'*20 + 'Feature Extraction Done' + '-'*20)


if __name__ == '__main__':
    feature_extraction(mode='train')
    feature_extraction(mode='valid')
    feature_extraction(mode='test')
