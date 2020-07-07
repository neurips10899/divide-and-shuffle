import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from sklearn.metrics import roc_auc_score, accuracy_score
import csv
import argparse
import datetime
import time

from models import tools
from data.Datasets import get_Dataloader


criterion = torch.nn.CrossEntropyLoss()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='WIDE_RESNET50', help='support --simple-net, alexnet, resnet18,34,50,101,152')
parser.add_argument('--dataset', type=str, default = 'IMAGENET', help='name of dataset')
parser.add_argument('--train_split', type=float, default=0.9)
parser.add_argument('--world_size', type=int, default=4)
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--log_interval', type=int, default=1)
parser.add_argument('--data_path', type=str, default='./data/')

parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--data_scale', type=str, default='small')
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--base_batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--sync', type=str, default="bsp")

args = parser.parse_args()
print(args)


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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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

def train(rank, world_size):
    group_dict = setup(rank, world_size)
    print('rank:', rank, ';local rank:', args.local_rank, '; world_size:', world_size)
    n = torch.cuda.device_count() // world_size
    n = 1
    device_ids = list(range(rank * n, (rank + 1) * n))
    print('rank:', rank, '; n:', n, '; device_ids', device_ids)

    # load data
    loader_train, loader_val = get_Dataloader(args)

    # build models
    mp_model = tools.get_model(args.model, num_classes=1000)
    mp_model = mp_model.to(args.local_rank)

    optimizer = torch.optim.SGD(
         mp_model.parameters(),
         lr=args.learning_rate,
         weight_decay=args.weight_decay)
    scheduler_steplr = StepLR(optimizer, step_size=30, gamma=0.1)

    # train
    results_record = {'acc1': [],  'acc5': [], 'logloss': []}
    total_iter = 0
    for epoch in range(args.num_epochs):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(loader_train),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        #scheduler_warmup.step(epoch)
        end = time.time()
        for iteration, (fields, target) in enumerate(loader_train):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.sync == "ds_sync":
                group_obj = group_dict[dist.get_rank()][total_iter%2]
                group_average_params(mp_model, group_obj)
            total_iter = total_iter + 1

            mp_model.train()
            fields = fields.to(args.local_rank)
            target = target.to(args.local_rank)
            y = mp_model(fields)
            loss = criterion(y, target)
            optimizer.zero_grad()
            loss.backward()
            if args.sync == "bsp":
                average_gradients(mp_model)
            optimizer.step()
            
            # measure batch time including computing and communication
            batch_time.update(time.time() - end)

            acc1, acc5 = accuracy(y, target, topk=(1, 5))
            losses.update(loss.item(), fields.size(0))
            top1.update(acc1[0], fields.size(0))
            top5.update(acc5[0], fields.size(0))

            end = time.time()

            if iteration % args.log_interval == 0 and rank == 0:
                progress.display(iteration)

        scheduler_steplr.step()
        if rank == 0:
            losses = AverageMeter('Loss', ':.3e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')

            mp_model.eval()
            with torch.no_grad():
                for fields, target in loader_val:
                    fields = fields.to(args.local_rank)
                    target = target.to(args.local_rank)

                    y = mp_model(fields)
                    loss = criterion(y, target)

                    acc1, acc5 = accuracy(y, target, topk=(1, 5))
                    top1.update(acc1[0], fields.size(0))
                    top5.update(acc5[0], fields.size(0))
                    losses.update(loss.item(), fields.size(0))

                results_record['acc1'].append(top1.avg)
                results_record['acc5'].append(top5.avg)
                results_record['logloss'].append(losses.avg)
                print('Validation: Top1 Accuracy:', top1.avg, 'Top5 Accuracy:', top5.avg ,'; LogLoss:', losses.avg)
                model_path = './checkpoints/rsp_'+ str(args.num_epochs) + '_' + str(args.batch_size)
                torch.save(mp_model.state_dict(), model_path)
    if rank == 0:
        file_path = './result/rsp_' + str(args.num_epochs) + '_' + args.data_scale \
            + '_' + str(args.learning_rate) + '_' + str(args.weight_decay) \
            + '_' + str(args.batch_size) + '.csv'
        # print(results_record)
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(results_record['acc1'])
            writer.writerow(results_record['acc5'])
            writer.writerow(results_record['logloss'])
        
    cleanup()


if __name__ == "__main__":

    start_time = datetime.datetime.now()

    dist_rank = int(os.environ['RANK'])
    train(dist_rank, args.world_size)

    end_time = datetime.datetime.now()
    print(end_time - start_time)
