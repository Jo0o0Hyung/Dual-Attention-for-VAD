import numpy as np
import os
import pickle

from utils import read_feature
from DB_reader import read_DB_structure
import config as c


def padding(DB, feature_directory, label_directory, padding_time):
    num_of_DB = len(DB)
    for i in range(num_of_DB):
        feat_path = DB['filename'][i]
        output_file = feat_path.split('/')[-1]
        output_file_path = os.path.join(feature_directory, output_file)

        label_path = DB['label_path'][i]
        output_label = label_path.split('/')[-1]
        output_label = output_label.split('.')[0] + '.pkl'
        output_label_path = os.path.join(label_directory, output_label)

        feature, label = read_feature(feat_path, label_path)

        start_seg_feat, end_seg_feat = feature[:100], feature[-100:]
        start_seg_label, end_seg_label = label[:100], label[-100:]

        # Data have already been padded with 1 seconds silence at at both ends of speech.
        if padding_time == 0.0:
            final_feature = feature[100:-100]
            final_label = label[100:-100]
        elif padding_time == 2.0:
            final_feature = np.concatenate((start_seg_feat, feature, end_seg_feat))
            final_label = np.concatenate((start_seg_label, label, end_seg_label))
        elif padding_time == 3.0:
            final_feature = np.concatenate((start_seg_feat, start_seg_feat, feature, end_seg_feat, end_seg_feat))
            final_label = np.concatenate((start_seg_label, start_seg_label, label, end_seg_label, end_seg_label))
        else:
            raise ValueError

        if os.path.isfile(output_file_path):
            print('\'' + output_file + '\'' + 'feature already extracted!')
        else:
            with open(output_file_path, 'wb') as fp:
                pickle.dump(final_feature, fp)
                print('[Padding] Feature : %s is done!' % output_file)

        if os.path.isfile(output_label_path):
            print('\'' + output_label + '\'' + 'label already extracted!')
        else:
            with open(output_label_path, 'wb') as fp:
                pickle.dump(final_label, fp)
                print('[Padding] Label : %s is done!' % output_label)


# Trimming the data based on extracted label
def Aurora_EPD(DB, feature_directory, label_directory):
    num_of_DB = len(DB)
    for i in range(num_of_DB):
        feat_path = DB['filename'][i]
        output_file = feat_path.split('/')[-1]
        output_file_path = os.path.join(feature_directory, output_file)

        label_path = DB['label_path'][i]
        output_label = label_path.split('/')[-1]
        output_label = output_label.split('.')[0] + '.pkl'
        output_label_path = os.path.join(label_directory, output_label)

        feature, label = read_feature(feat_path, label_path)

        # Getting start and end points in utterance
        start_point = np.where(label == 1)[0][0]
        end_point = np.where(label == 1)[0][-1]

        epd_feature = feature[start_point:end_point]
        epd_label = label[start_point:end_point]

        if os.path.isfile(output_file_path):
            print('\'' + output_file + '\'' + 'feature already extracted!')
        else:
            with open(output_file_path, 'wb') as fp:
                pickle.dump(epd_feature, fp)
                print('[EPD] Feature : %s is done!' % output_file)

        if os.path.isfile(output_label_path):
            print('\'' + output_label + '\'' + 'label already extracted!')
        else:
            with open(output_label_path, 'wb') as fp:
                pickle.dump(epd_label, fp)
                print('[EPD] Label : %s is done!' % output_label)


def main():
    padding_time = 0.0
    train_feat_dir = os.path.join(c.MFB_DIR + '_' + str(1.0), 'train_folder')
    output_feat_dir = os.path.join(c.MFB_DIR + '_' + str(padding_time), 'train_folder')
    output_label_dir = os.path.join(c.LABEL_DIR + '_' + str(padding_time), 'train_folder')

    if not os.path.exists(output_feat_dir):
        os.makedirs(output_feat_dir)

    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    train_DB = read_DB_structure(train_feat_dir, 'train')

    # -1.0 means EPD
    if padding_time == '-1.0':
        Aurora_EPD(train_DB, output_feat_dir, output_label_dir)
    else:
        padding(train_DB, output_feat_dir, output_label_dir, padding_time)


if __name__ == '__main__':
    main()
