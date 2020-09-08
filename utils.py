import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import scipy.io
import numpy as np
from sklearn.metrics import roc_curve

import config as c
matplotlib.use('Agg')


def read_feature(feat_path, label_path):
    with open(feat_path, 'rb') as f:
        feature = pickle.load(f)

    label_extension = label_path.split('/')[-1].split('.')[-1]
    if label_extension == 'mat':
        label = scipy.io.loadmat(label_path)
        label = label['final_label']
    elif label_extension == 'pkl':
        with open(label_path, 'rb') as f:
            label = pickle.load(f)

    if len(feature) != len(label):
        feature = feature[0:len(label)]

    return feature, label


def eer(label, pred):
    '''
    EER : Equal Error Rate
    FAR (FPR) : False Accpet(Positive) Rate (FP / (TN + FP))
    TPR : True Positive Rate (TP / (TP + FN))
    '''
    FAR, TPR, threshold = roc_curve(label, pred, pos_label=1)
    MR = 1 - TPR
    EER = FAR[np.nanargmin(np.absolute(MR - FAR))]

    return FAR, MR, EER


def global_feature_normalize(feature, train_mean, train_std):
    mu = train_mean
    sigma = train_std

    return (feature - mu) / sigma


def train_mean_std(train_DB):
    print('Start to Calculate the Global Mean and Standard Deviation of train DB')
    n_files = len(train_DB)
    train_mean, train_std, n_frames = 0., 0., 0.

    # calculate the global mean of train DB
    for i in range(n_files):
        filename = train_DB['filename'][i]
        labelname = train_DB['label_path'][i]
        inputs, _ = read_feature(filename, labelname)
        temp_n_frames = len(inputs)
        train_mean += np.sum(inputs, axis=0, keepdims=True)
        n_frames += temp_n_frames
    train_mean = train_mean / n_frames

    # calculate the global std of train DB
    for i in range(n_files):
        filename = train_DB['filename'][i]
        labelname = train_DB['label_path'][i]
        inputs, _ = read_feature(filename, labelname)
        deviation = np.sum((inputs - train_mean) ** 2, axis=0, keepdims=True)
        train_std += deviation
    train_std = train_std / (n_frames - 1)
    train_std = np.sqrt(train_std)

    return train_mean, train_std


def calc_global_mean_std(mean_path, std_path, train_DB):
    try:
        mean = np.loadtxt(mean_path, delimiter='\n')
        mean = np.expand_dims(mean, 0)
        std = np.loadtxt(std_path, delimiter='\n')
        std = np.expand_dims(std, 0)
        # print("The global mean and std of train DB are loaded from saved files")
        return mean, std

    except:
        mean, std = train_mean_std(train_DB)
        np.savetxt(mean_path, mean, delimiter='\n')
        np.savetxt(std_path, std, delimiter='\n')
        print("The global mean and std of train DB are saved")
        return mean, std


def get_global_mean_std(train_DB, padding_time):
    MS_path = os.path.join(c.MFB_DIR + '_' + str(float(padding_time)), 'Train_Mean_Var')
    mean_path = os.path.join(MS_path, 'train_mean.txt')
    std_path = os.path.join(MS_path, 'train_std.txt')
    if not os.path.exists(MS_path):
        os.makedirs(MS_path)
    train_mean, train_std = calc_global_mean_std(mean_path, std_path, train_DB)
    return train_mean, train_std


def save_lr_and_losses(log_dir, epoch, lr, train_loss, valid_loss, valid_AUC):
    directory = os.path.join(log_dir, 'generated_outputs')
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = "epoch" + str(epoch).zfill(2) + "_lr_and_loss.txt"

    lr_and_loss = {}
    lr_and_loss["lr"] = lr
    lr_and_loss["train_loss"] = train_loss
    lr_and_loss["valid_loss"] = valid_loss
    lr_and_loss["valid_AUC"] = valid_AUC

    f = open(os.path.join(directory, filename), 'w')
    for k, v in lr_and_loss.items():
        data = k + " " + str(v) + "\n"
        f.write(data)
    f.close()


def visualize_the_loss(log_dir):
    directory = os.path.join(log_dir, 'generated_outputs')
    if not os.path.exists(directory):
        raise Exception("generated_outputs are not exist")
    # filenames = sorted(os.listdir(directory))
    filenames = [f for f in os.listdir(directory) if f.endswith('.txt')]  # ex) ['epoch25_lr_and_loss.txt',...]
    index_list = [int(f.split('_')[0][-2:]) for f in filenames]  # ex) [25, ...]
    index_list, filenames = zip(*sorted(zip(index_list, filenames)))

    train_loss = []
    valid_loss = []
    valid_AUC = []

    for filename in filenames:
        if filename.endswith(".txt"):
            full_filename = os.path.join(directory, filename)
            f = open(full_filename, 'r')
            lines = f.readlines()
            train_loss_tmp = float(lines[1].split()[-1])
            valid_loss_tmp = float(lines[2].split()[-1])
            valid_AUC_tmp = float(lines[3].split()[-1])
            train_loss.append(train_loss_tmp)
            valid_loss.append(valid_loss_tmp)
            valid_AUC.append(valid_AUC_tmp)
            f.close()

    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    print('Lowest training loss at epoch %d' % minposs)

    maxAUCposs = valid_AUC.index(max(valid_AUC)) + 1
    print('Highest validation AUC at epoch %d' %maxAUCposs)

    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    dest = os.path.join(directory, 'loss_plot.png')
    fig.savefig(dest, bbox_inches='tight')

    return minposs, maxAUCposs


def visualize_the_learning_rate(log_dir):
    directory = os.path.join(log_dir, 'generated_outputs')
    if not os.path.exists(directory):
        raise Exception("generated_outputs are not exist")

    filenames = [f for f in os.listdir(directory) if f.endswith('.txt')]  # ex) ['epoch25_lr_and_loss.txt',...]
    index_list = [int(f.split('_')[0][-2:]) for f in filenames]  # ex) [25, ...]
    index_list, filenames = zip(*sorted(zip(index_list, filenames)))

    lr = []

    for filename in filenames:
        full_filename = os.path.join(directory, filename)
        f = open(full_filename, 'r')
        lines = f.readlines()
        lr_tmp = float(lines[0].split()[-1])
        lr.append(lr_tmp)
        f.close()

    fig = plt.figure(figsize=(10, 8)).gca()
    plt.plot(range(1, len(lr) + 1), lr, label='Learning rate')

    plt.xlabel('epochs')
    plt.ylabel('learning_rate')
    plt.ylim(0, max(lr))  # consistent scale
    plt.xlim(1, len(lr) + 1)  # consistent scale
    plt.legend()
    plt.tight_layout()
    # plt.show()
    fig.xaxis.set_major_locator(MaxNLocator(integer=True))
    dest = os.path.join(directory, 'lr_plot.png')
    fig.figure.savefig(dest, bbox_inches='tight')
