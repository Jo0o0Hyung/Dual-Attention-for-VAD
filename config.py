import os

SR = 16000

DB_DIR = '../VAD_DB'
TRAIN_NOISE_DIR = os.path.join(DB_DIR, 'Nonspeech_16k')
TEST_NOISE_DIR = os.path.join(DB_DIR, 'Aurora_noise')
WAV_DIR = os.path.join(DB_DIR, 'Aurora4_16k_wav')
NOISY_WAV_DIR = os.path.join(DB_DIR, 'Aurora4_16k_noisy_wav')

TRAIN_WAV_DIR = os.path.join(NOISY_WAV_DIR, 'train_folder')
VALID_WAV_DIR = os.path.join(NOISY_WAV_DIR, 'valid_folder')
TEST_WAV_DIR = os.path.join(NOISY_WAV_DIR, 'test_folder')

LABEL_DIR = os.path.join(DB_DIR, 'Au4_label_silence_padded')

FEAT_DIR = os.path.join(DB_DIR, 'Feature')
MFB_DIR = os.path.join(FEAT_DIR, 'MFB')

TRAIN_FEAT_DIR = os.path.join(MFB_DIR, 'train_folder')
VALID_FEAT_DIR = os.path.join(MFB_DIR, 'valid_folder')
TEST_FEAT_DIR = os.path.join(MFB_DIR, 'test_folder')

# About Feature
USE_LOGSCALE = True
USE_SCALE = False
USE_GLOBAL_NORM = True
FILTER_BANK = 40

TRAIN_TOTAL_HOURS = 60
VALID_TOTAL_HOURS = 6

P_DNN_HIDDEN_SIZE = 32
