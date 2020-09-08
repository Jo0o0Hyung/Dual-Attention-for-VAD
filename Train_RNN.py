import os
import argparse
import time
import shutil
import warnings
from logger import Logger
import random
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import config as c
from utils import read_feature, save_lr_and_losses, visualize_the_learning_rate, visualize_the_loss
from DB_reader import read_DB_structure
from Model import Model
from VAD_Dataset import VAD_Dataset, VAD_Compose, LSTMInputTrain, LSTMInputTest, ToTensorInput
from focalloss import focal_loss

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'

parser = argparse.ArgumentParser(description='PyTorch VAD(Voice Activity Detection)')
# Model Options
parser.add_argument('--log-dir', default='./checkpoints_', help='folder to output model checkpoints')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--start-epoch', default=1, type=int, metavar='SE',
                    help='manual epoch number')
parser.add_argument('--epochs', default=20, type=int, metavar='E',
                    help='number of epochs to train')

parser.add_argument('--RNN-model', default='BasicRNN', type=str, metavar='RM',
                    help='choose the RNN Model(BasicRNN or AttentionRNN)')
parser.add_argument('--attention-type', default='Combined', type=str, metavar='AT',
                    help='choose the attention type (TA / FA / DA1 / DA2)')
parser.add_argument('--hidden-size', default=64, type=int, metavar='HS',
                    help='number of LSTM hidden units')
parser.add_argument('--seq-len', default=50, type=int, metavar='SL',
                    help='number of sequential length')
parser.add_argument('--seed', default=2019, type=int, metavar='S',
                    help='random seed for initializing training')
parser.add_argument('--shuffle', action='store_true', default=True,
                    help='shuffle or not')
parser.add_argument('--num-layers', default=3, type=int, metavar='NL',
                    help='number of hidden layers')
parser.add_argument('--batch-size', default=128, type=int, metavar='BS',
                    help='input batch size for training')
parser.add_argument('--padding-time', default=1.0, type=float, metavar='PT',
                    help='padding time in train(valid) data')
parser.add_argument('--loss', default='FocalLoss', type=str, metavar='L',
                    help='choose the loss function. CrossEntropy or FocalLoss')
parser.add_argument('--valid-batch-size', default=1, type=int, metavar='VBS',
                    help='input batch size for validating(testing)')
parser.add_argument('--gamma', default=0.1, type=float, metavar='G',
                    help='hyper parameter for focal loss')
parser.add_argument('--lr', default=1e-1, type=float, metavar='LR',
                    help='starting learning rate')
parser.add_argument('--lr-decay', default=1e-4, type=float, metavar='LRD',
                    help='learning rate decay ratio')
parser.add_argument('--weight-decay', default=0.0, type=float, metavar='WD',
                    help='weight decay')
parser.add_argument('--optimizer', default='sgd', type=str, metavar='OPT',
                    help='the optimizer to use')
#Device
parser.add_argument('--no-cuda', default=False, action='store_true',
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--log-interval', default=22, metavar='LI',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()


def save_checkpoint(state, is_best, filename='chekpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copy(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_optimizer(model, new_lr):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr, momentum=0.9, dampening=0, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr, weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=new_lr, lr_decay=args.lr_decay, weight_decay = args.wd)

    return optimizer


def accuracy(output, target, topk=(1,)):
    '''Computes the accuracy over the k top predictions for the specified values of k'''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]

    return lr


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    n_correct, n_total = 0, 0
    zero_count, one_count = 0, 0
    model.train()
    end = time.time()

    for batch_idx, (data) in enumerate(train_loader):
        inputs, targets = data

        total_element = targets.shape[0] * targets.shape[1]
        one_element = np.count_nonzero(targets)
        zero_element = total_element - one_element
        zero_count += zero_element
        one_count += one_element

        data_time.update(time.time() - end)

        inputs = Variable(inputs)
        targets = Variable(targets)

        device_num = 'cuda:' + args.gpu_id
        device = torch.device(device_num)

        if args.cuda:
            inputs, targets = inputs.to(device), targets.to(device)

        linear_out, sigmoid_out = model(x=inputs)

        pred = sigmoid_out >= 0.5
        targets = targets.squeeze(-1)
        n_correct += (pred.long() == targets.long()).sum().item()
        n_total += args.seq_len * args.batch_size
        train_acc = 100. * n_correct / n_total
        loss = criterion(linear_out, targets.float())
        losses.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {train_acc:.4f}'.format(
                    epoch, batch_idx * len(inputs), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, train_acc=train_acc)
            )
    zero_proportion = (zero_count / (zero_count + one_count)) * 100
    one_proportion = (one_count / (zero_count + one_count)) * 100
    print('zero_element : {} zero_proportion : {}\n'
          'one_element : {} one_proportion : {}\n'.format(zero_count, zero_proportion, one_count, one_proportion))

    return losses.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()

    mean_AUC = 0.
    n_correct, n_total = 0., 0.
    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, data in enumerate(val_loader):
            inputs, targets = data
            inputs = inputs.squeeze(0)
            targets = targets.squeeze(0)

            device_num = 'cuda:' + args.gpu_id
            device = torch.device(device_num)

            if args.cuda:
                inputs, targets = inputs.to(device), targets.to(device)

            linear_out, sigmoid_out = model(x=inputs)
            linear_out[linear_out != linear_out] = 0
            sigmoid_out[sigmoid_out != sigmoid_out] = 0

            linear_out, sigmoid_out = linear_out.squeeze(0), sigmoid_out.squeeze(0)
            loss = criterion(linear_out, targets.float())

            np_targets = targets.data.cpu().numpy()
            np_sigmoid_out = sigmoid_out.data.cpu().numpy()
            np_sigmoid_out = np.nan_to_num(np_sigmoid_out)

            temp_AUC = roc_auc_score(np_targets, np_sigmoid_out)
            pred = sigmoid_out >= 0.5
            n_correct += (pred.long() == targets.long()).sum().item()
            n_total += len(targets)
            val_acc = 100. * n_correct / n_total

            losses.update(loss.item(), inputs.size(0))
            mean_AUC += temp_AUC
            batch_time.update(time.time() - end)
            end = time.time()

            if i % (args.log_interval * 20) == 0:
                print('Validating the model: ({:8d}/{:8d})'.format(i, len(val_loader.dataset)))

        mean_AUC /= (i + 1)

        print(' * Validation => '
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'AUC {mean_AUC:.3f}\t'
              'Acc {val_acc:.5f}'.format(loss=losses, mean_AUC=mean_AUC * 100, val_acc=val_acc))

    return losses.avg, mean_AUC


def main():
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device_num = 'cuda:' + args.gpu_id
    device = torch.device(device_num)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    LOG_DIR = args.log_dir + str(
        args.seed) + '/Padding-{}/Atype-{}_Loss-{}_gamma-{}'.format(args.padding_time, args.attention_type, args.loss, args.gamma)

    if not os.path.exists(LOG_DIR):
        logger = Logger(LOG_DIR)

    if args.cuda:
        cudnn.benchmark = True

    input_size = c.FILTER_BANK
    model = Model(rnn_model=args.RNN_model, input_size=input_size, rnn_hidden_size=args.hidden_size,
                  num_layers=args.num_layers, dnn_hidden_size=c.P_DNN_HIDDEN_SIZE, seq_len=args.seq_len,
                  attention_type=args.attention_type)
    train_feat_dir = os.path.join(c.MFB_DIR + '_' + str(float(args.padding_time)), 'train_folder')
    valid_feat_dir = os.path.join(c.MFB_DIR + '_' + str(1.0), 'valid_folder')

    train_DB = read_DB_structure(train_feat_dir, 'train')
    valid_DB = read_DB_structure(valid_feat_dir, 'valid')

    transform = VAD_Compose([
        LSTMInputTrain(sequence_length=args.seq_len),
        ToTensorInput()
    ])

    transform_v = VAD_Compose([
        LSTMInputTest(),
        ToTensorInput()
    ])

    file_loader = read_feature

    train_dataset = VAD_Dataset(DB=train_DB, loader=file_loader, transform=transform)
    valid_dataset = VAD_Dataset(DB=valid_DB, loader=file_loader, transform=transform_v)
    print('\nParsed Options:\n{}\n'.format(vars(args)))

    if args.cuda:
        model.to(device)

    start = args.start_epoch
    end = start + args.epochs
    if args.loss == 'CrossEntropy':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'FocalLoss':
        criterion = focal_loss(alpha=1.0, gamma=args.gamma)

    optimizer = create_optimizer(model, args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, min_lr=1e-5, verbose=True)

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                               shuffle=args.shuffle, num_workers=args.workers, pin_memory=args.cuda)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=args.valid_batch_size,
                                               shuffle=False, num_workers=args.workers, pin_memory=args.cuda)

    for epoch in range(start, end):
        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        valid_loss, valid_AUC = validate(valid_loader, model, criterion, args)
        scheduler.step(valid_loss, epoch)

        current_LR = get_learning_rate(optimizer)[0]
        print(' * Learning Rate : %0.4f' % current_LR)

        save_lr_and_losses(LOG_DIR, epoch, current_LR, train_loss, valid_loss, valid_AUC)
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   '{}/checkpoint {}.pth'.format(LOG_DIR, epoch))
    min_loss, max_AUC = visualize_the_loss(LOG_DIR)
    visualize_the_learning_rate(LOG_DIR)
    model_total_params = sum(p.numel() for p in model.parameters())
    print('The number of parameters = %d' % model_total_params)


if __name__ == '__main__':
    main()
