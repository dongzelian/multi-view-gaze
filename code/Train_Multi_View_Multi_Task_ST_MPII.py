"""
@author: Dongze Lian
@contact: liandz@shanghaitech.edu.cn
@software: PyCharm
@file: Train_Multi_View_Multi_Task_ST_MPII.py
@time: 2020/1/12 19:30
"""
import sys
sys.path.append('..')
import argparse
import numpy as np
import time


import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import logging
import os
import scipy.io as sio
from tensorboardX import SummaryWriter


import network.gazenet as gazenet
import network.resnet as resnet
import tools.utils as utils


import pdb


parser = argparse.ArgumentParser(description='Network and training parameters choices')
# Network choices
parser.add_argument('--network', type=str, default='ResNet-34', metavar='backbone')
parser.add_argument('--data_dir', type=str, default='/p300/datasets/Gaze/ShanghaiTechGaze/ShanghaiTechGaze/', metavar='NET',
                    help='dataset dir')
parser.add_argument('--camera', type=str, default='multiview', metavar='camera')
parser.add_argument('--batch-size-mpii', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--batch-size-st', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size-mpii', type=int, default=50, metavar='N')
parser.add_argument('--test-batch-size-st', type=int, default=50, metavar='N',)
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-decay', type=int, default=10, metavar='N',
                    help='lr decay interval with epoch (default: 10)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M',
                    help='Weight decay (default: 5e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--resume', default='', type=str, help='path to the lastest checkpoint (default: none)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--print_freq', type=int, default=20, metavar='N')
parser.add_argument('--ckpt_freq', type=int, default=5, metavar='N',)
parser.add_argument('--gpu', type=int, default=0, metavar='N',
                    help='which gpu device (default: 0)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# log path setting
exp_path = os.path.join('/root/github/multi-view-gaze-copy/exps',
                         time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time())))
if not os.path.exists(exp_path):
    os.makedirs(exp_path)


# tensorboardX setting
writer = SummaryWriter(os.path.join(exp_path, 'runs'))


# log setting
log_file = os.path.join(exp_path, 'exp.log')
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s',
                    filename=log_file,
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# print setting
logging.info(args)


# ST Dataset processing
class GazeImageDataset(Dataset):
    def __init__(self, txt_file, txt_dir, transform=None):
        self.txt_dir = txt_dir
        self.transform = transform

        #TODO: modify interface
        self.data_lists = []
        for txt_file_i in txt_file:
            self.data_lists.append(utils.txt2list(os.path.join(self.txt_dir, txt_file_i)))

    def __len__(self):
        return len(self.data_lists[0])

    def __getitem__(self, idx):
        input_list = []
        for i, data_list in enumerate(self.data_lists):
            if (i + 1) % 3 == 0:
                input_list.append(sio.loadmat(args.data_dir + data_list[idx])['eyelocation'])
            elif i == len(self.data_lists) - 1:
                gt = sio.loadmat(args.data_dir + data_list[idx])['xy_gt']
                gt[0] -= W_screen / 2
                gt[1] -= H_screen / 2
                input_list.append(gt)
            else:
                input_list.append(Image.open(args.data_dir + data_list[idx]))

        if self.transform:
            input_tran_list = []
            for i, input in enumerate(input_list):
                if (i + 1) % 3 == 0:
                    input_tran_list.append(torch.squeeze(torch.FloatTensor(input)))
                elif i == len(input_list) - 1:
                    input_tran_list.append(torch.FloatTensor(input))
                else:
                    input_tran_list.append(self.transform(input))
        else:
            input_tran_list = input_list

        # len(input_tran_list) == 10, data & target
        return input_tran_list



# MPIIGaze Dataset processing
#TODO: modify the MPII dataset interface
class MPIIGazeDataset(Dataset):
    def __init__(self, txt_file, data_dir, transform=None):
        self.root_dir = data_dir
        self.transform = transform
        self.lefteye_name_list = utils.txt2list(os.path.join(self.root_dir, txt_file[0]))
        self.righteye_name_list = utils.txt2list(os.path.join(self.root_dir, txt_file[1]))
        self.gt_left_name_list = utils.txt2list(os.path.join(self.root_dir, txt_file[2]))
        self.gt_right_name_list = utils.txt2list(os.path.join(self.root_dir, txt_file[3]))
        self.lefteye_center_list = utils.txt2list(os.path.join(self.root_dir, txt_file[4]))
        self.righteye_center_list = utils.txt2list(os.path.join(self.root_dir, txt_file[5]))


    def __len__(self):
        return len(self.lefteye_name_list)

    def __getitem__(self, idx):
        lefteye_name = self.lefteye_name_list[idx]
        righteye_name = self.righteye_name_list[idx]
        gt_left_name = self.gt_left_name_list[idx]
        gt_right_name = self.gt_right_name_list[idx]
        lefteye_center_name = self.lefteye_center_list[idx]
        righteye_center_name = self.righteye_center_list[idx]


        lefteye = Image.open(lefteye_name)
        righteye = Image.open(righteye_name)
        gt_left = sio.loadmat(gt_left_name)['gt_left']
        gt_right = sio.loadmat(gt_right_name)['gt_right']
        lefteye_center = [float(number) for number in lefteye_center_name]
        righteye_center = [float(number) for number in righteye_center_name]

        sample = {}

        if self.transform:
            sample['le'] = self.transform(lefteye)
            sample['re'] = self.transform(righteye)
            sample['gt_left'] = torch.FloatTensor(gt_left)
            sample['gt_right'] = torch.FloatTensor(gt_right)
            sample['le_center'] = torch.FloatTensor(lefteye_center)
            sample['re_center'] = torch.FloatTensor(righteye_center)
            sample['gt_co'] = torch.FloatTensor([0])

        return sample



# training
def train(train_loader, train_mpii_loader, model, criterion, optimizer, epoch):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    mpii_losses = utils.AverageMeter()
    point_errors = utils.AverageMeter()
    angle_errors = utils.AverageMeter()

    model.train()
    end = time.time()
    #TODO: add MPIIGaze dataset iteration
    train_mpii_iterator = iter(train_mpii_loader)
    for batch_idx, input in enumerate(train_loader):
        data_time.update(time.time() - end)
        data, target = tuple(input[:len(input)-1]), input[-1]
        if args.cuda:
            data, target = tuple(map(lambda x : x.cuda(), data)), target.cuda()

        output = model(*data)
        target = target.view(target.size(0), -1)
        loss = criterion(output, target)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure point error and record loss
        point_error = compute_error(output, target)
        losses.update(loss.item(), data[0].size(0))
        point_errors.update(point_error, data[0].size(0))


        batch_time.update(time.time() - end)
        end = time.time()

        writer.add_scalar('Train/Loss', losses.avg, batch_idx + len(train_loader) * epoch)
        writer.add_scalar('Train/Error', point_errors.avg, batch_idx + len(train_loader) * epoch)


        # print the intermediate results
        if batch_idx % args.print_freq == 0:
            logging.info('Time({}:{:.0f}), Train Epoch [{}]: [{}/{}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                         'Error {error.val:.3f} ({error.avg:.3f})'.format(
                time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())), time.time() % 60,
                epoch, batch_idx, len(train_loader), batch_time=batch_time, data_time=data_time,
                loss=losses, error=point_errors))

        #TODO: MPIIGaze dataset iteration
        (mpii_input, mpii_target), train_mpii_iterator = utils.infinite_get(train_mpii_iterator, train_mpii_loader)
        if args.cuda:
            mpii_input, mpii_target = tuple(map(lambda x: x.cuda(), mpii_input)), mpii_target.cuda()

        mpii_output = model(*mpii_input)
        mpii_target = mpii_target.view(mpii_target.size(0), -1)

        #TODO: add coplanar loss
        mpii_loss = criterion(mpii_output, mpii_target) + 0.000001 * criterion(distance, mpii_target[2])

        optimizer.zero_grad()
        mpii_loss.backward()
        optimizer.step()

        # Measure angle error and record loss
        angle_error = compute_angle_error(mpii_output, mpii_target)
        mpii_losses.update(mpii_loss.item(), data[0].size(0))
        angle_errors.update(angle_error, mpii_input[0].size(0))

        writer.add_scalar('Train/MPII_Loss', mpii_losses.avg, batch_idx + len(train_loader) * epoch)
        writer.add_scalar('Train/Angle_Error', angle_errors.avg, batch_idx + len(train_loader) * epoch)


        # print the intermediate results
        if batch_idx % args.print_freq == 0:
            logging.info('Time({}:{:.0f}), Train Epoch [{}]: [{}/{}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                         'Error {error.val:.3f} ({error.avg:.3f})'.format(
                time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())), time.time() % 60,
                epoch, batch_idx, len(train_loader), batch_time=batch_time, data_time=data_time,
                loss=losses, error=angle_errors))


# testing
def test(test_loader, model, criterion, epoch, minimal_error):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    error = utils.AverageMeter()

    with torch.no_grad():
        model.eval()
        end = time.time()
        for batch_idx, input in enumerate(test_loader):
            data_time.update(time.time() - end)
            data, target = tuple(input[:len(input) - 1]), input[-1]
            if args.cuda:
                data, target = tuple(map(lambda x: x.cuda(), data)), target.cuda()

            output = model(*data)
            target = target.view(target.size(0), -1)
            loss = criterion(output, target)

            point_error = compute_error(output, target)
            losses.update(loss.item(), data[0].size(0))
            error.update(point_error, data[0].size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            # print the intermediate results
            if batch_idx % args.print_freq == 0:
                logging.info('Time({}:{:.0f}), Test Epoch [{}]: [{}/{}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                             'Error {error.val:.3f} ({error.avg:.3f})'.format(
                    time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())), time.time() % 60,
                    epoch, batch_idx, len(test_loader), batch_time=batch_time, data_time=data_time,
                    loss=losses, error=error))

        writer.add_scalar('Test/Loss', losses.avg, epoch)
        writer.add_scalar('Test/Error', error.avg, epoch)
        logging.info(' * Test Error {error.avg:.3f} Minimal_error {minimal_error:.3f}'
                     .format(error=error, minimal_error=minimal_error))

    return error.avg

def compute_error(output, target):
    """Computes the point error between prediction and gt"""
    with torch.no_grad():
        output = output.cpu().numpy()
        target = target.cpu().numpy()
        delta = (output - target) ** 2
        point_error = np.sqrt(delta.sum(axis=1)).mean()
        return point_error

def compute_angle_error(output, target):
    """Computes the angle error between prediction and gt"""
    #TODO: compute angle error
    # with torch.no_grad():
    #     output = output.cpu().numpy()
    #     target = target.cpu().numpy()
    #     delta = (output - target) ** 2
    #     angle_error = np.sqrt(delta.sum(axis=1)).mean()
        #return angle_error
    return


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def adjust_learning_rate(optimizer, epoch):
    new_lr = args.lr * (0.1 ** (epoch // args.lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return new_lr


# ========================================== network config ===============================================
minimal_error = 100000
W_screen = 59.77   # the width of screen
H_screen = 33.62   # the height of screen




model = gazenet.GazeNet(backbone=args.network, view=args.camera, pretrained=True)
model = nn.DataParallel(model).cuda()


optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
criterion = nn.MSELoss()


# optionally resume from a checkpoint
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        epoch_index = checkpoint['epoch']
        minimal_error = checkpoint['minimal_error']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))



# ======================================= Build ST dataset =======================================
# image transform & normalization
data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_dataset = GazeImageDataset(
    txt_file=['leftcamera/lefteye.txt', 'leftcamera/righteye.txt', 'leftcamera/eyelocation.txt',
              'middlecamera/lefteye.txt', 'middlecamera/righteye.txt', 'middlecamera/eyelocation.txt',
              'rightcamera/lefteye.txt', 'rightcamera/righteye.txt', 'rightcamera/eyelocation.txt',
              'gt.txt'],
    txt_dir=args.data_dir + 'annotations/txtfile/train_txt/',
    transform=data_transforms)
logging.info('The number of training data is: {}'.format(len(train_dataset)))


test_dataset = GazeImageDataset(
    txt_file=['leftcamera/lefteye.txt', 'leftcamera/righteye.txt', 'leftcamera/eyelocation.txt',
              'middlecamera/lefteye.txt', 'middlecamera/righteye.txt', 'middlecamera/eyelocation.txt',
              'rightcamera/lefteye.txt', 'rightcamera/righteye.txt', 'rightcamera/eyelocation.txt',
              'gt.txt'],
    txt_dir=args.data_dir + 'annotations/txtfile/test_txt/',
    transform=data_transforms)
logging.info('The number of testing data is: {}'.format(len(test_dataset)))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)


# ======================================= Build MPIIGaze dataset =======================================

#TODO: add MPIIGaze dataset interface
# train_dataset = MPIIGazeDataset(
#     txt_file=['leftcamera/lefteye.txt', 'leftcamera/righteye.txt', 'leftcamera/eyelocation.txt',
#               'middlecamera/lefteye.txt', 'middlecamera/righteye.txt', 'middlecamera/eyelocation.txt',
#               'rightcamera/lefteye.txt', 'rightcamera/righteye.txt', 'rightcamera/eyelocation.txt',
#               'gt.txt'],
#     txt_dir=args.data_dir + 'annotations/txtfile/train_txt/',
#     transform=data_transforms)
# logging.info('The number of training data is: {}'.format(len(train_dataset)))
#
#
# test_dataset = MPIIGazeDataset(
#     txt_file=['leftcamera/lefteye.txt', 'leftcamera/righteye.txt', 'leftcamera/eyelocation.txt',
#               'middlecamera/lefteye.txt', 'middlecamera/righteye.txt', 'middlecamera/eyelocation.txt',
#               'rightcamera/lefteye.txt', 'rightcamera/righteye.txt', 'rightcamera/eyelocation.txt',
#               'gt.txt'],
#     txt_dir=args.data_dir + 'annotations/txtfile/test_txt/',
#     transform=data_transforms)
# logging.info('The number of testing data is: {}'.format(len(test_dataset)))
#
# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
#                           num_workers=args.num_workers, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
#                          num_workers=args.num_workers, pin_memory=True)


# ======================================== Start Training ===============================================
logging.info('\n............Start training............\n')
#start_time = time.time()

for epoch in range(args.epochs):
    new_lr = adjust_learning_rate(optimizer, epoch)
    logging.info('Current learning rate: {}'.format(new_lr))

    # ============================================ Training ===========================================
    logging.info('============ Train stage ============')
    train(train_loader, train_mpii_loader, model, criterion, optimizer, epoch)

    # =========================================== Evaluation ==========================================
    logging.info('============ Test stage ============')
    test_error = test(test_loader, model, criterion, epoch, minimal_error)

    # record the minimal error and save checkpoint
    is_best = test_error < minimal_error
    minimal_error = min(test_error, minimal_error)

    # save the checkpoint
    ckpt_path = os.path.join(exp_path, 'ckpts')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if is_best:
        logging.info('Minimal error {} in epoch {}'.format(minimal_error, epoch))
        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'minimal_error': minimal_error,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(ckpt_path, 'model_best.pth.tar'))

    if epoch % args.ckpt_freq == 0:
        filename = os.path.join(ckpt_path, 'epoch_%d.pth.tar' % epoch)
        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'minimal_error': minimal_error,
                'optimizer': optimizer.state_dict(),
            }, filename)

