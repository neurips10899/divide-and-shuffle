import argparse
import os
import datetime
import time
import shutil

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler


import torchvision
import torchvision.transforms as transforms

from models import *


parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--world_size', type=int, default=4)
parser.add_argument('--sync', type=str, default="bsp")

args = parser.parse_args()
print(args)

best_prec = 0

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    groups = []
    groups.append(torch.distributed.new_group(ranks=[0,1], backend = "nccl"))
    groups.append(torch.distributed.new_group(ranks=[2,3], backend = "nccl"))
    groups.append(torch.distributed.new_group(ranks=[0,3], backend = "nccl"))
    groups.append(torch.distributed.new_group(ranks=[1,2], backend = "nccl"))
    group_dict = {0:[groups[0],groups[2]], 1:[groups[0],groups[3]], 2:[groups[1],groups[3]], 3:[groups[1],groups[2]]}

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    return group_dict

def cleanup():
    dist.destroy_process_group()

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            dist.all_reduce(m.running_mean.data, op=dist.reduce_op.SUM)
            m.running_mean.data /= size
            dist.all_reduce(m.running_var.data, op=dist.reduce_op.SUM)
            m.running_var.data /= size

def group_average_params(model, group_obj):
    size = float(dist.get_world_size())/2
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.reduce_op.SUM, group=group_obj)
        param.data /= size
    
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            dist.all_reduce(m.running_mean.data, op=dist.reduce_op.SUM, group=group_obj)
            m.running_mean.data /= size
            dist.all_reduce(m.running_var.data, op=dist.reduce_op.SUM, group=group_obj)
            m.running_var.data /= size


def main(rank, world_size):
    global args, best_prec
    use_gpu = torch.cuda.is_available()

    group_dict = setup(rank, world_size)
    print('rank:', rank, ';local rank:', args.local_rank, '; world_size:', world_size)
    n = torch.cuda.device_count() // world_size
    n = 1
    total_iter = 0
    device_ids = list(range(rank * n, (rank + 1) * n))
    print('rank:', rank, '; n:', n, '; device_ids', device_ids)

    # Model building
    print('=> Building model...')
    if use_gpu:

        model = wide_resnet_cifar(depth=26, width=10, num_classes=100)

        # mkdir a new folder to store the checkpoint and best model
        if not os.path.exists('result'):
            os.makedirs('result')
        fdir = 'result/wsr26_10_cifar100_bs'+str(args.batch_size)
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        # adjust the lr according to the model type
        if isinstance(model, (ResNet_Cifar, PreAct_ResNet_Cifar)):
            model_type = 1
        elif isinstance(model, Wide_ResNet_Cifar):
            model_type = 2
        elif isinstance(model, (ResNeXt_Cifar, DenseNet_Cifar)):
            model_type = 3
        else:
            print('model type unrecognized...')
            return

        model = model.to(args.local_rank)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        cudnn.benchmark = True
    else:
        print('Cuda is not available!')
        return

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint "{}"'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading and preprocessing
    # CIFAR10
    if args.cifar_type == 10:
        print('=> loading cifar10 data...')
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        sampler = DistributedSampler(train_dataset)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), num_workers=2, sampler=sampler)

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    # CIFAR100
    else:
        print('=> loading cifar100 data...')
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

        train_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        sampler = DistributedSampler(train_dataset)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), num_workers=2, sampler=sampler)

        test_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    if args.evaluate:
        validate(testloader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, model_type)

        # train for one epoch
        train(trainloader, model, criterion, optimizer, group_dict, total_iter, epoch)

        # evaluate on test set
        prec = validate(testloader, model, criterion)

        # remember best precision and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec,best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, fdir)

    cleanup()

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


def train(trainloader, model, criterion, optimizer, group_dict, total_iter, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.sync == "ds_sync":
            group_obj = group_dict[dist.get_rank()][total_iter % 2]
            group_average_params(model, group_obj)
        total_iter = total_iter + 1

        input, target = input.to(args.local_rank), target.to(args.local_rank)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.sync == "bsp":
            average_gradients(model)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   epoch, i, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(args.local_rank), target.to(args.local_rank)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, model_type):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if model_type == 1:
        if epoch < 80:
            lr = args.lr
        elif epoch < 120:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    elif model_type == 2:
        if epoch < 60:
            lr = args.lr
        elif epoch < 120:
            lr = args.lr * 0.2
        elif epoch < 160:
            lr = args.lr * 0.04
        else:
            lr = args.lr * 0.008
    elif model_type == 3:
        if epoch < 150:
            lr = args.lr
        elif epoch < 225:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__=='__main__':
    start_time = datetime.datetime.now()

    dist_rank = int(os.environ['RANK'])
    main(dist_rank, args.world_size)

    end_time = datetime.datetime.now()
    print(end_time - start_time)

