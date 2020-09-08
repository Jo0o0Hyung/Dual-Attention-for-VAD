import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tabulate import tabulate
from sklearn.metrics import roc_auc_score

import config as c
from DB_reader import read_DB_structure
from VAD_Dataset import LSTMInputTest, ToTensorInput
from Model import Model
from utils import read_feature, calc_global_mean_std, global_feature_normalize, eer

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'

parser = argparse.ArgumentParser(description='PyTorch VAD')
parser.add_argument('--log-dir', default='./checkpoints_', help='folder to output model checkpoints')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--workers', default=4, type=int, metavar='W',
                    help='number of data loading workers')
parser.add_argument('--hidden-size', default=64, type=int, metavar='HS',
                    help='number of hidden units')
parser.add_argument('--RNN-model', default='BasicRNN', type=str, metavar='RM',
                    help='choose the RNN Model(BasicRNN or AttentionRNN)')
parser.add_argument('--attention-type', default='Combined', type=str, metavar='AT',
                    help='choose the attention type')
parser.add_argument('--num-layers', default=3, type=int, metavar='NL',
                    help='number of hidden layers')
parser.add_argument('--seq-len', default=50, type=int, metavar='SL',
                    help='length of RNN''s input sequence')
parser.add_argument('--batch-size', default=128, type=int, metavar='BS',
                    help='input batch size for training')
parser.add_argument('--test-batch-size', default=1, type=int, metavar='TBS',
                    help='input batch size for validating(testing)')
parser.add_argument('--gamma', default=0.1, type=float, metavar='G',
                    help='hyperparameter for focal loss')
parser.add_argument('--cp-num', type=int, default=1, metavar='ES',
                    help='which check point to load')
parser.add_argument('--seed', default=2019, type=int, metavar='S',
                    help='random seed for initializing training')
parser.add_argument('--padding-time', default=1.0, type=float, metavar='PT',
                    help='padding time in train(valid) data')
parser.add_argument('--loss', default='FocalLoss', type=str, metavar='L',
                    help='choose the loss function. CrossEntropy or FocalLoss')
parser.add_argument('--lr', default=1e-1, type=float, metavar='LR',
                    help='starting learning rate')
parser.add_argument('--lr-decay', default=1e-4, type=float, metavar='LRD',
                    help='learning rate decay ratio')
parser.add_argument('--weight-decay', default=0.0, type=float, metavar='WD',
                    help='weight decay')
parser.add_argument('--optimizer', default='sgd', type=str, metavar='OPT',
                    help='the optimizer to use')
parser.add_argument('--no-cuda', default=False, action='store_true',
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--log-interval', default=22, metavar='LI',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()


def test_input_load(feat_path, label_path):
    inputs, targets = read_feature(feat_path, label_path)

    if args.Backbone_model == 'baseLSTM' or args.Backbone_model == 'CLDNN':
        train_DB = read_DB_structure(os.path.join(c.MFB_DIR + '_' + str(args.padding_time), 'train_folder'), 'train')
        MS_path = os.path.join(c.MFB_DIR + '_' + str(args.padding_time), 'Train_Mean_Var')

    elif args.Backbone_model == '2DCRNN':
        train_DB = read_DB_structure(os.path.join(c.STFT_DIR + '_1.0', 'train_folder'), 'train')
        MS_path = os.path.join(c.STFT_DIR + '_1.0', 'Train_Mean_Var')

    if c.USE_GLOBAL_NORM:
        mean_path = os.path.join(MS_path, 'train_mean.txt')
        std_path = os.path.join(MS_path, 'train_std.txt')
        train_mean, train_std = calc_global_mean_std(mean_path, std_path, train_DB)
        inputs = global_feature_normalize(inputs, train_mean, train_std)

    TI = LSTMInputTest()
    TT = ToTensorInput()

    inputs, targets = TI(inputs, targets)
    inputs, targets = TT(inputs, targets)

    with torch.no_grad():
        inputs = Variable(inputs)
        targets = Variable(targets)

    return inputs, targets


def select_test_DB(test_DB_all, DB_list):
    '''
    Return the test_DB list which is sorted according to DB_list's sequence
    '''
    test_DB = pd.DataFrame()
    test_DB = test_DB.append(test_DB_all[test_DB_all['noise_type'] == DB_list], ignore_index=True)

    return test_DB


def test(model, DB, criterion):
    n_files = len(DB)
    n_frames, n_correct, n_total = 0, 0, 0
    mean_cost, mean_accuracy, mean_AUC, mean_EER = 0, 0, 0, 0
    temp_AUC = 0

    for i in range(n_files):
        feat_path = DB['filename'][i]
        label_path = DB['label_path'][i]
        inputs, targets = test_input_load(feat_path, label_path)

        device_num = 'cuda:' + args.gpu_id
        device = torch.device(device_num)

        if args.cuda:
            inputs, targets = inputs.to(device), targets.to(device)

        linear_out, sigmoid_out, _ = model(x=inputs)
        linear_out, sigmoid_out = linear_out.squeeze(0), sigmoid_out.squeeze(0)
        linear_out[linear_out != linear_out] = 0
        sigmoid_out[sigmoid_out != sigmoid_out] = 0

        temp_cost = criterion(linear_out, targets.float()).data.cpu().numpy().item()
        pred = sigmoid_out >= 0.5

        n_correct += (pred.long() == targets.long()).sum().item()
        n_frames += len(targets)
        np_targets = targets.data.cpu().numpy()
        np_sigmoid_out = sigmoid_out.data.cpu().numpy()
        np_sigmoid_out = np.nan_to_num(np_sigmoid_out)
        ROC_AUC = roc_auc_score(np_targets, np_sigmoid_out)
        _, _, temp_eer = eer(np_targets.flatten(), np_sigmoid_out.flatten())
        temp_AUC += ROC_AUC
        mean_cost += temp_cost / n_files
        mean_EER += temp_eer / n_files

    mean_accuracy = 100. * n_correct / n_frames
    mean_AUC = temp_AUC / n_files

    print(tabulate([['Averaged cost', mean_cost], ['Averaged AUC (%)', mean_AUC*100],
                    ['Averaged ACC (%)', mean_accuracy], ['Averaged EER (%)', mean_EER*100]],
                   tablefmt='rst'))

    return mean_accuracy, mean_AUC, mean_EER, mean_cost, temp_AUC, n_files


def main():
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        cudnn.benchmark = True

    noise_list = ['airport_', 'babble_', 'car_', 'destroyerengine_', 'F16_cockpit_', 'factory_', 'machinegun_',
                  'street_', 'train_', 'volvo_']
    SNR_list = ['SNR-5', 'SNR0', 'SNR5', 'SNR10']
    DB_list = []

    for i in range(len(noise_list)):
        for j in range(len(SNR_list)):
            DB_list.append(noise_list[i] + SNR_list[j])

    LOG_DIR = args.log_dir + str(
        args.seed) + '/Padding-{}/Atype-{}_Loss-{}_gamma-{}'.format(args.padding_time, args.attention_type, args.loss, args.gamma)
    print(LOG_DIR)

    input_size = c.FILTER_BANK
    model = Model(rnn_model=args.RNN_model, input_size=input_size, rnn_hidden_size=args.hidden_size,
                  num_layers=args.num_layers, dnn_hidden_size=c.P_DNN_HIDDEN_SIZE, seq_len=args.seq_len,
                  attention_type=args.attention_type)

    test_DB = read_DB_structure(os.path.join(c.MFB_DIR + '_' + str(1.0), 'test_folder'), 'test')

    device_num = 'cuda:' + args.gpu_id
    device = torch.device(device_num)

    if args.cuda:
        model.to(device)

    print('=> loading checkpoint: CP_NUM = ' + str(args.cp_num))
    checkpoint = torch.load(LOG_DIR + '/checkpoint ' + str(args.cp_num) + '.pth')

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    snr_files = np.zeros(4)
    snr_AUC = np.zeros(4)
    five_files = np.zeros(4)
    five_noises_auc = np.zeros(4)

    for i in range(len(DB_list)):
        selected_DB = select_test_DB(test_DB, DB_list[i])
        print(DB_list[i])
        m_Acc, m_AUC, m_EER, m_cost, temp_AUC, n_files = test(model, selected_DB, criterion)
        snr = DB_list[i].split('_')[-1][3:]

        if snr == '-5':
            snr_files[0] = snr_files[0] + n_files
            snr_AUC[0] = snr_AUC[0] + temp_AUC
            if DB_list[i].split('_')[0] == 'babble' or DB_list[i].split('_')[0] == 'destroyerengine' or \
                    DB_list[i].split('_')[0] == 'F16_cockpit' or DB_list[i].split('_')[0] == 'factory' or DB_list[i].split('_')[0] == 'street':
                five_noises_auc[0] = five_noises_auc[0] + temp_AUC
                five_files[0] = five_files[0] + n_files

        elif snr == '0':
            snr_files[1] = snr_files[1] + n_files
            snr_AUC[1] = snr_AUC[1] + temp_AUC
            if DB_list[i].split('_')[0] == 'babble' or DB_list[i].split('_')[0] == 'destroyerengine' or \
                    DB_list[i].split('_')[0] == 'F16_cockpit' or DB_list[i].split('_')[0] == 'factory' or DB_list[i].split('_')[0] == 'street':
                five_noises_auc[1] = five_noises_auc[1] + temp_AUC
                five_files[1] = five_files[1] + n_files

        elif snr == '5':
            snr_files[2] = snr_files[2] + n_files
            snr_AUC[2] = snr_AUC[2] + temp_AUC
            if DB_list[i].split('_')[0] == 'babble' or DB_list[i].split('_')[0] == 'destroyerengine' or \
                    DB_list[i].split('_')[0] == 'F16_cockpit' or DB_list[i].split('_')[0] == 'factory' or DB_list[i].split('_')[0] == 'street':
                five_noises_auc[2] = five_noises_auc[2] + temp_AUC
                five_files[2] = five_files[2] + n_files

        elif snr == '10':
            snr_files[3] = snr_files[3] + n_files
            snr_AUC[3] = snr_AUC[3] + temp_AUC
            if DB_list[i].split('_')[0] == 'babble' or DB_list[i].split('_')[0] == 'destroyerengine' or \
                    DB_list[i].split('_')[0] == 'F16_cockpit' or DB_list[i].split('_')[0] == 'factory' or DB_list[i].split('_')[0] == 'street':
                five_noises_auc[3] = five_noises_auc[3] + temp_AUC
                five_files[3] = five_files[3] + n_files

    print('-'*7 + 'All Noises' + '-'*7)
    print(tabulate([['-5dB AUC', 100*(snr_AUC[0] / snr_files[0])], [' 0dB AUC', 100*(snr_AUC[1] / snr_files[1])],
                    [' 5dB AUC', 100*(snr_AUC[2] / snr_files[2])], ['10dB AUC', 100*(snr_AUC[3] / snr_files[3])],
                    ['-5,0dB AVG', 100*((snr_AUC[0]/snr_files[0] + snr_AUC[1]/snr_files[1])/2)],
                    ['Total AVG', 100*((snr_AUC[0]/snr_files[0] + snr_AUC[1]/snr_files[1] + snr_AUC[2]/snr_files[2] +
                                        snr_AUC[3]/snr_files[3])/4)]],
                   tablefmt='grid'))
    print('-' * 7 + '5 Noises' + '-' * 7)
    print(tabulate([['-5dB AUC', 100 * (five_noises_auc[0] / five_files[0])], [' 0dB AUC', 100 * (five_noises_auc[1] / five_files[1])],
                    [' 5dB AUC', 100 * (five_noises_auc[2] / five_files[2])], ['10dB AUC', 100 * (five_noises_auc[3] / five_files[3])],
                    ['-5,0dB AVG', 100*((five_noises_auc[0]/five_files[0] + five_noises_auc[1]/five_files[1])/2)],
                    ['Total AVG', 100*((five_noises_auc[0]/five_files[0] + five_noises_auc[1]/five_files[1] +
                                        five_noises_auc[2]/five_files[2] + five_noises_auc[3]/five_files[3])/4)]],
                   tablefmt='grid'))


if __name__ == '__main__':
    main()
