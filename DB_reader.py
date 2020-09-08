import logging
import os
from glob import glob
import pandas as pd

import config as c

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)


# Recursively finds all feature files matching the pattern.
def find_wav_files1(directory, pattern='*.wav'):
    return glob(os.path.join(directory, pattern))


def find_wav_files2(directory, pattern='**/*.wav'):
    return glob(os.path.join(directory, pattern))


def find_wav_files3(directory, pattern='**/**/*.wav'):
    return glob(os.path.join(directory, pattern))


def find_feat_files1(directory, pattern='*.pkl'):
    return glob(os.path.join(directory, pattern))


def find_feat_files2(directory, pattern='**/*.pkl'):
    return glob(os.path.join(directory, pattern))


def convert_wav_path_to_noisy_wav_path(filename):
    noisy_wav_dir = c.NOISY_WAV_DIR
    noisy_wav_path = os.path.join(noisy_wav_dir, filename.split('/')[-3]) # train / valid / test

    if not os.path.exists(noisy_wav_path):
        os.makedirs(noisy_wav_path)

    return noisy_wav_path


def convert_file_name_to_label_path(filename, mode):
    if mode == 'train' or mode == 'valid':
        # get padding time from filename
        padding_time = filename.split('/')[-3].split('_')[-1]
        label_dir = c.LABEL_DIR + '_' + padding_time
        sub_dir = filename.split('/')[-2]  # 'train_foler'
        pkl_name = filename.split('/')[-1]  # '12345_xxxxx.pkl' -> 12345 is count number
        # if padding time is 1.0, label has been saved as mat type
        if padding_time == '1.0':
            mod_name = pkl_name.split('_')[-1].replace('.pkl', '.mat')
        # if padding time is not 1.0, label has been saved as pkl type
        else:
            mod_name = pkl_name.split('_')[-1]
    if mode == 'test':
        sub_dir = filename.split('/')[-3]  # 'test_folder'
        pkl_name = filename.split('/')[-1]
        mod_name = pkl_name.replace('.pkl', '.mat')
        label_dir = c.LABEL_DIR

    sub_path = os.path.join(sub_dir, mod_name)
    label_path = os.path.join(label_dir, sub_path)

    return label_path


def read_wav_structure(directory):
    DB = pd.DataFrame()
    DB['filename'] = find_wav_files3(directory)
    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/'))
    DB['speaker_id'] = DB['filename'].apply(lambda x: x.split('/')[-2])  # speaker folder name
    DB['dataset_id'] = DB['filename'].apply(lambda x: x.split('/')[-3]) # train / valid / test
    DB['noisy_wav_dir'] = DB['filename'].apply(lambda x: convert_wav_path_to_noisy_wav_path(x)) # noisy wav path
    logging.info(DB.head(10))

    return DB


def read_noise_wav_structure(directory):
    DB = pd.DataFrame()
    DB['filename'] = find_wav_files1(directory)
    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/'))
    DB['foldername'] = DB['filename'].apply(lambda x: x.split('/')[-2])  # Nonspeech_16k or Aurora_noise

    return DB


def read_noisy_wav_structure(directory, mode):
    DB = pd.DataFrame()
    if mode == 'train' or mode == 'valid':
        DB['filename'] = find_wav_files1(directory)
        DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/'))
        DB['dataset_id'] = DB['filename'].apply(lambda x: x.split('/')[-2])
    elif mode == 'test':
        DB['filename'] = find_wav_files2(directory)
        DB['filename'].apply(lambda x: x.replace('\\', '/'))
        DB['dataset_id'] = DB['filename'].apply(lambda x: x.split('/')[-3])
    logging.info(DB.head(10))

    return DB


def read_DB_structure(directory, mode):
    DB = pd.DataFrame()
    if mode == 'train' or mode == 'valid':
        DB['filename'] = find_feat_files1(directory)
    elif mode == 'test':
        DB['filename'] = find_feat_files2(directory)
        DB['noise_type'] = DB['filename'].apply(lambda x: x.split('/')[-2])

    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/'))
    DB['label_path'] = DB['filename'].apply(lambda x: convert_file_name_to_label_path(x, mode))
    logging.info(DB.head(10))

    return DB
