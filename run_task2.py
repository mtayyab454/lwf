from __future__ import print_function

import sys
sys.path.insert(1, '../')
from models.vgg_multitask import vgg16_multitask

import os
import random
import argparse
import torch.nn.parallel

from utils import Logger, get_time_str, create_dir, backup_code
from trainer import testing_loop, training_loop, training_loop_subtask
from task_utils import get_task_sets

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

model_names = ['vgg16', ]

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets

parser.add_argument('--jobid', type=str, default='test')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vgg16)')

parser.add_argument('--task_set', default='vehicles10-large_animals10-reset80', choices=['vehicles10-manmade_objects15-reset75', 'vehicles10-manmade_objects10-reset80', 'vehicles10-large_animals10-reset80'])
parser.add_argument('-d', '--dataset', default='cifar100', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')


parser.add_argument('--t1-weight', default=0.4, type=float,
                    help='weight assigned to the old task')
parser.add_argument('--lr-t2', default=0.004, type=float,
                    metavar='LR', help='initial learning rate')


# Task1 options
parser.add_argument('--epochs-t1', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--schedule-t1', type=int, nargs='+', default=[50, 70],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--lr-t1', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')

# Task2 options
parser.add_argument('--epochs-t2', default=45, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--schedule-t2', type=int, nargs='+', default=[15, 30],
                        help='Decrease learning rate at these epochs.')

parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--logs', default='logs', type=str, metavar='PATH',
                    help='path to save the training logs (default: logs)')
# Architecture
# Miscs
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Random seed
if args.manualSeed is None:
     args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

def main():
     print(args)
     exp_name = args.jobid + '_' + args.arch + '_t2'
     checkpoint_dir = os.path.join(args.checkpoint, exp_name)
     create_dir([checkpoint_dir, args.logs])

     task_idx, cp_name = get_task_sets(args.task_set)

     args.task1 = {'class_id':task_idx[0], 'out_id':0}
     args.task2 = {'class_id': task_idx[2], 'out_id': 1}

     t1_model = vgg16_multitask([args.task1['class_id'].__len__()])
     sd = torch.load(os.path.join('checkpoint', cp_name+'_vgg16_t1', cp_name+'_vgg16_t1_best.pth'))
     t1_model.load_state_dict(sd)
     t1_model.cuda()

     model = vgg16_multitask([args.task1['class_id'].__len__(), args.task2['class_id'].__len__()])
     model.features.load_state_dict(t1_model.features.state_dict())
     model.classifiers[0].load_state_dict(t1_model.classifiers[0].state_dict())
     model.cuda()
     print(model)

     logger = Logger(dir_path=args.logs, fname=exp_name,
                    keys=['time', 'acc1', 'acc5', 'ce_loss'])
     logger.one_time({'seed':args.manualSeed, 'comments': 'Train task 2 by optimizing all layers with 0.1 mse_err weight'})
     logger.set_names(['lr', 'train_stats', 'test_stats_nt', 'test_stats_ot'])

     testing_loop(model=model, args=args, task=args.task1['class_id'], out_id=args.task1['out_id'], keys=logger.keys)
     training_loop_subtask(model=model, logger=logger, args=args, save_best=True)

if __name__ == '__main__':
    main()