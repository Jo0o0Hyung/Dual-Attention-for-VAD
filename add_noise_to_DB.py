import os
import librosa
import random
import math
import itertools
import numpy as np

import config as c
from add_noise import add_noise
from DB_reader import read_wav_structure, read_noise_wav_structure

random.seed(200)


def train_mix(train_DB, noise_DB, noisy_wav_dir):
    SNRs = [-10, -5, 0, 5, 10, 15]
    total_hours = c.TRAIN_TOTAL_HOURS
    total_frames = int(total_hours * 3600 / 0.01)
    length_frame = 0
    count = 1

    while length_frame < total_frames:
        SNR = random.choice(SNRs)
        '''
        sample returns a random sample of items.
        ex) 1584 /home/user/DB/VAD_DB/Aurora4_16k_wav/train_folder/016_16k/015c0216.wav
        '''
        filename = train_DB.sample(n=1)['filename'].reset_index(drop=True)[0]
        noise_path = noise_DB.sample(n=1)['filename'].reset_index(drop=True)[0]

        speech, sr = librosa.load(filename, sr=c.SR, mono=True)
        noise, sr = librosa.load(noise_path, sr=c.SR, mono=True)

        ''' Silence Padding : Padding Silence 1 seconds at both ends of speech'''
        silence = speech[:int(0.2 * sr)]
        silence = np.tile(silence, 5)
        padded_speech = np.hstack((silence, speech, silence))

        num_rep = math.ceil(len(padded_speech) / len(noise))
        repeated_noise = np.tile(noise, num_rep)

        mixed_wav, _ = add_noise(padded_speech, repeated_noise, sr, SNR)
        noisy_wav_path = os.path.join(noisy_wav_dir, 'train_folder', str(count) + '_' + filename.split('/')[-1])
        librosa.output.write_wav(noisy_wav_path, mixed_wav, sr)

        print('[train] count:%d, speech:%s, noise type:%s, SNR:%ddB' % (
        count, filename.split('/')[-1].replace('.wav', ''), noise_path.split('/')[-1].replace('.wav', ''), SNR))

        length_frame = length_frame + math.ceil(len(speech) / 160)
        count = count + 1

    print('Training Data Mix is Completed')


def valid_mix(valid_DB, noise_DB, noisy_wav_dir):
    SNRs = [-10, -5, 0, 5, 10]

    total_hours = c.VALID_TOTAL_HOURS
    total_frames = int(total_hours * 3600 / 0.01)
    length_frame = 0
    count = 1

    while length_frame < total_frames:
        SNR = random.choice(SNRs)
        filename = valid_DB.sample(n=1)['filename'].reset_index(drop=True)[0]
        noise_path = noise_DB.sample(n=1)['filename'].reset_index(drop=True)[0]

        speech, sr = librosa.load(filename, sr=c.SR, mono=True)
        noise, sr = librosa.load(noise_path, sr=c.SR, mono=True)

        ''' Silence Padding : Padding Silence 1 seconds at both ends of speech'''
        silence = speech[:int(0.2 * sr)]
        silence = np.tile(silence, 5)
        padded_speech = np.hstack((silence, speech, silence))

        num_rep = math.ceil(len(padded_speech) / len(noise))
        repeated_noise = np.tile(noise, num_rep)

        mixed_wav, _ = add_noise(padded_speech, repeated_noise, sr, SNR)
        noisy_wav_path = os.path.join(noisy_wav_dir, 'valid_folder', str(count) + '_' + filename.split('/')[-1])
        librosa.output.write_wav(noisy_wav_path, mixed_wav, sr)

        print('[valid] count:%d, speech:%s, noise type:%s, SNR:%ddB, n_frames:%d' % (
        count, filename.split('/')[-1].replace('.wav', ''), noise_path.split('/')[-1].replace('.wav', ''), SNR,
        length_frame))

        length_frame = length_frame + math.ceil(len(speech) / 160)
        count = count + 1

    print('Validation Data Mix is Completed')


def test_mix(test_DB, noise_DB, noisy_wav_dir):
    SNRs = [-5, 0, 5, 10]
    noise_types = ['airport', 'babble', 'car', 'destroyerengine', 'F16_cockpit', 'factory', 'machinegun', 'street',
                   'train', 'volvo']
    count = 1

    # itertools.product make the combination of noise_types and SNRs
    for noise_type, SNR in list(itertools.product(noise_types, SNRs)):
        noise_path = noise_DB[noise_DB['filename'].str.contains(noise_type)]['filename'].reset_index(drop=True)[0]
        noisy_wav_sub_dir = os.path.join(noisy_wav_dir, 'test_folder', noise_type + '_SNR' + str(SNR))

        if not os.path.exists(noisy_wav_sub_dir):
            os.makedirs(noisy_wav_sub_dir)

        for i in range(len(test_DB)):
            speech_path = test_DB['filename'][i]
            noisy_wav_path = os.path.join(noisy_wav_sub_dir, speech_path.split('/')[-1])

            if os.path.isfile(noisy_wav_path):
                print('[%d] %s is already exists!' % (count, noisy_wav_path))
                count = count + 1
                continue
            speech, sr = librosa.load(speech_path, sr=c.SR, mono=True)
            noise, sr = librosa.load(noise_path, sr=c.SR, mono=True)

            mixed_wav, _ = add_noise(speech, noise, sr, SNR)
            librosa.output.write_wav(noisy_wav_path, mixed_wav, sr)

            print('[test] count:%d, speech:%s, noise type:%s, SNR:%ddB' % (
            count, speech_path.split('/')[-1].replace('.wav', ''), noise_path.split('/')[-1].replace('.wav', ''),
            SNR))

            count = count + 1

    print('Test Data Mix is Completed')


def mix_and_save(wav_DB, noise_DB, noisy_wav_dir, mode):
    if mode == 'train':
        train_DB = wav_DB[wav_DB['dataset_id'] == 'train_folder'].reset_index(drop=True)
        if len(os.listdir(os.path.join(noisy_wav_dir, 'train_folder'))) == 0:
            train_mix(train_DB, noise_DB, noisy_wav_dir)
        else:
            print('train folder is not empty!')

    elif mode == 'valid':
        valid_DB = wav_DB[wav_DB['dataset_id'] == 'valid_folder'].reset_index(drop=True)
        if len(os.listdir(os.path.join(noisy_wav_dir, 'valid_folder'))) == 0:
            valid_mix(valid_DB, noise_DB, noisy_wav_dir)
        else:
            print('valid folder is not empy!')

    elif mode == 'test':
        test_DB = wav_DB[wav_DB['dataset_id'] == 'test_folder'].reset_index(drop=True)
        test_mix(test_DB, noise_DB, noisy_wav_dir)


def main():
    mode = 'valid'

    if mode == 'train':
        noise_dir = c.TRAIN_NOISE_DIR
    else:
        noise_dir = c.TEST_NOISE_DIR
    noise_DB = read_noise_wav_structure(noise_dir)

    wav_dir = c.WAV_DIR
    wav_DB = read_wav_structure(wav_dir)

    noisy_wav_dir = c.NOISY_WAV_DIR

    mix_and_save(wav_DB, noise_DB, noisy_wav_dir, mode)
