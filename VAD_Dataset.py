import torch
import random
import math
import numpy as np
import torch.utils.data as data
import os

import config as c
from utils import calc_global_mean_std, global_feature_normalize


class LSTMInputTrain(object):
    def __init__(self, sequence_length, input_per_file=1):
        super(LSTMInputTrain, self).__init__()
        self.sequence_length = sequence_length
        self.input_per_file = input_per_file

    def __call__(self, frames_features, labels):
        num_frames = len(frames_features)
        network_inputs, network_targets = [], []

        for i in range(self.input_per_file):
            j = random.randrange(1, num_frames - self.sequence_length)

            for k in range(self.sequence_length):
                network_inputs.append(frames_features[j + k])
                network_targets.append(labels[j + k])

        return np.array(network_inputs), np.array(network_targets)


class LSTMInputTest(object):
    def __init__(self, input_per_file=1):
        super(LSTMInputTest, self).__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features, labels):

        network_inputs = frames_features
        network_inputs = np.expand_dims(network_inputs, axis=0)
        network_targets = labels
        network_targets = np.squeeze(network_targets)

        return np.array(network_inputs), np.array(network_targets)


class ToTensorInput(object):
    def __call__(self, np_features, label):
        if isinstance(np_features, np.ndarray) and isinstance(label, np.ndarray):
            tensor_features = torch.from_numpy(np_features).float()
            label = torch.from_numpy(label).long()

            return tensor_features, label


class VAD_Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, feat, label):
        for t in self.transforms:
            feat, label = t(feat, label)

        return feat, label

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'

        return format_string


class VAD_Dataset(data.Dataset):
    def __init__(self, DB, loader, transform=None):
        self.DB = DB
        [_, padding_time] = self.DB['filename'][0].split('/')[-3].split('_')
        self.len = len(DB)
        self.transform = transform
        self.loader = loader
        feat_dir = c.MFB_DIR

        if c.USE_GLOBAL_NORM:
            MS_path = os.path.join(feat_dir + '_' + str(padding_time), 'Train_Mean_Var')
            if not os.path.exists(MS_path):
                os.makedirs(MS_path)
            mean_path = os.path.join(MS_path, 'train_mean.txt')
            std_path = os.path.join(MS_path, 'train_std.txt')
            self.train_mean, self.train_std = calc_global_mean_std(mean_path, std_path, DB)

    def __getitem__(self, index):
        feat_path = self.DB['filename'][index]
        label_path = self.DB['label_path'][index]
        feature, label = self.loader(feat_path, label_path)
        if c.USE_GLOBAL_NORM:
            feature = global_feature_normalize(feature, self.train_mean, self.train_std)

        if self.transform:
            feature, label = self.transform(feature, label)

        return feature, label

    def __len__(self):
        return self.len
