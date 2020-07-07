import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import partition_dataset, get_data
from dfm import DeepFactorizationMachineModel

from sklearn.metrics import roc_auc_score
import csv
import argparse
import datetime

criterion = torch.nn.BCELoss()

parser = argparse.ArgumentParser()
parser.add_argument('--train_split', type=float, default=0.9)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--mlp_dims', type=list, default=(400, 400, 400))
parser.add_argument('--embed_dim', type=int, default=10)
parser.add_argument('--data_path', type=str, default='../data/')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--data_scale', type=str, default='small')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser.add_argument('--sync', type=str, default="bsp")
args = parser.parse_args()
print(args)


def setup():
    # initialize the process group
    print('initiate group')
    dist.init_process_group(backend="nccl")
    print('finished')
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
        if param.requires_grad:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            dist.all_reduce(m.running_mean.data, op=dist.reduce_op.SUM)
            m.running_mean.data /= size
            dist.all_reduce(m.running_var.data, op=dist.reduce_op.SUM)
            m.running_var.data /= size

def average_params(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            dist.all_reduce(m.running_mean.data, op=dist.reduce_op.SUM)
            m.running_mean.data /= size
            dist.all_reduce(m.running_var.data, op=dist.reduce_op.SUM)
            m.running_var.data /= size

def group_average_params(model, group_obj):
    size = float(dist.get_world_size())/2
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM, group=group_obj)
        param.data /= size
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            dist.all_reduce(m.running_mean.data, op=dist.reduce_op.SUM, group=group_obj)
            m.running_mean.data /= size
            dist.all_reduce(m.running_var.data, op=dist.reduce_op.SUM, group=group_obj)
            m.running_var.data /= size

def train_deepfm(rank, world_size, group_dict):
    print('rank:', rank, '; world_size:', world_size)
    n = torch.cuda.device_count() // world_size
    n = 1
    device_ids = list(range(rank * n, (rank + 1) * n))
    print('rank:', rank, '; n:', n, '; device_ids', device_ids)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)


    # load data
    loader_train, loader_val, field_dims = partition_dataset(
        args.batch_size, args.train_split, args.data_path, args.data_scale)

    # build models
    mp_model = DeepFactorizationMachineModel(
        field_dims,
        embed_dim=args.embed_dim,
        mlp_dims=args.mlp_dims,
        dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(
         mp_model.parameters(),
         lr=args.learning_rate,
         #betas=(0.0, 0.0),
         weight_decay=args.weight_decay)

    # train
    results_record = {'auc': [], 'logloss': []}
    for epoch in range(args.num_epochs):
        t = 0
        total_loss = 0
        for iteration, (fields, target) in enumerate(loader_train):
            s = time.time()
            mp_model.train()
            fields = fields.to(device)
            target = target.to(device, dtype=torch.float)
            y = mp_model(fields)
            loss = criterion(y, target)
            optimizer.zero_grad()
            loss.backward()
            if args.sync == "bsp":
                average_gradients(mp_model)
            optimizer.step()
            if args.sync == "ds_sync":
                group_obj = group_dict[dist.get_rank()][iteration%2]
                group_average_params(mp_model, group_obj)
            total_loss += loss.item()
            e = time.time()
            t += e - s
            if (iteration + 1) % args.log_interval == 0:
                print('Epoch %d Iteration %d Rank %d, LogLoss = %.4f, Time = %.4f' % (epoch, iteration, rank, total_loss / args.log_interval, t/args.log_interval))
                total_loss = 0
                t = 0

            if iteration%100 == 0:
                logloss = 0
                mp_model.eval()
                targets, predicts = list(), list()
                with torch.no_grad():
                    for fields, target in loader_val:
                        fields = fields.to(device)
                        target = target.to(device, dtype=torch.float)

                        # try:
                        y = mp_model(fields)
                        loss = criterion(y, target)
                        logloss += loss.item()
                        targets.extend(target.tolist())
                        predicts.extend(y.tolist())
                        # except Exception as e:
                        #     print(e)

                auc = roc_auc_score(targets, predicts)
                logloss /= len(loader_val)
                results_record['auc'].append(auc)
                results_record['logloss'].append(logloss)
                print('Validation: AUC:', auc, '; LogLoss:', logloss)

    if rank == 0:
        file_path = './result/deepfm' + str(args.num_epochs) + '_' + args.data_scale \
            + '_' + str(args.learning_rate) + '_' + str(args.weight_decay) \
            + '_' + str(args.batch_size) + '_'+str(42) +'.csv'
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(results_record['auc'])
            writer.writerow(results_record['logloss'])
        
    cleanup()


if __name__ == "__main__":

    start_time = datetime.datetime.now()
    group_dict = setup()
    train_deepfm(dist.get_rank(), dist.get_world_size(), group_dict)
    end_time = datetime.datetime.now()
    print(end_time - start_time)
