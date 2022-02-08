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

def class_id_from_names(names):
     class_to_idx = {'apple': 0,
 'aquarium_fish': 1,
 'baby': 2,
 'bear': 3,
 'beaver': 4,
 'bed': 5,
 'bee': 6,
 'beetle': 7,
 'bicycle': 8,
 'bottle': 9,
 'bowl': 10,
 'boy': 11,
 'bridge': 12,
 'bus': 13,
 'butterfly': 14,
 'camel': 15,
 'can': 16,
 'castle': 17,
 'caterpillar': 18,
 'cattle': 19,
 'chair': 20,
 'chimpanzee': 21,
 'clock': 22,
 'cloud': 23,
 'cockroach': 24,
 'couch': 25,
 'crab': 26,
 'crocodile': 27,
 'cup': 28,
 'dinosaur': 29,
 'dolphin': 30,
 'elephant': 31,
 'flatfish': 32,
 'forest': 33,
 'fox': 34,
 'girl': 35,
 'hamster': 36,
 'house': 37,
 'kangaroo': 38,
 'keyboard': 39,
 'lamp': 40,
 'lawn_mower': 41,
 'leopard': 42,
 'lion': 43,
 'lizard': 44,
 'lobster': 45,
 'man': 46,
 'maple_tree': 47,
 'motorcycle': 48,
 'mountain': 49,
 'mouse': 50,
 'mushroom': 51,
 'oak_tree': 52,
 'orange': 53,
 'orchid': 54,
 'otter': 55,
 'palm_tree': 56,
 'pear': 57,
 'pickup_truck': 58,
 'pine_tree': 59,
 'plain': 60,
 'plate': 61,
 'poppy': 62,
 'porcupine': 63,
 'possum': 64,
 'rabbit': 65,
 'raccoon': 66,
 'ray': 67,
 'road': 68,
 'rocket': 69,
 'rose': 70,
 'sea': 71,
 'seal': 72,
 'shark': 73,
 'shrew': 74,
 'skunk': 75,
 'skyscraper': 76,
 'snail': 77,
 'snake': 78,
 'spider': 79,
 'squirrel': 80,
 'streetcar': 81,
 'sunflower': 82,
 'sweet_pepper': 83,
 'table': 84,
 'tank': 85,
 'telephone': 86,
 'television': 87,
 'tiger': 88,
 'tractor': 89,
 'train': 90,
 'trout': 91,
 'tulip': 92,
 'turtle': 93,
 'wardrobe': 94,
 'whale': 95,
 'willow_tree': 96,
 'wolf': 97,
 'woman': 98,
 'worm': 99}
     class_idx =  [class_to_idx[x] for x in names]

     return class_idx

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

parser.add_argument('-d', '--dataset', default='cifar100', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--t1-weight', default=0.1, type=float,
                    help='weight assigned to the old task')
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
parser.add_argument('--lr-t2', default=0.005, type=float,
                    metavar='LR', help='initial learning rate')

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
     exp_name = args.jobid + '_' + args.arch
     checkpoint_dir = os.path.join(args.checkpoint, exp_name)
     create_dir([checkpoint_dir+'_t1', checkpoint_dir+'_t2', args.logs])

     task3_idx = class_id_from_names(['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'])
     task2_idx = class_id_from_names(['bottle', 'bowl', 'can', 'cup', 'plate', 'clock', 'keyboard', 'lamp', 'telephone', 'television',
      'bed', 'chair', 'couch', 'table', 'wardrobe'])
     task1_idx = class_id_from_names(['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout', 'orchid', 'poppy', 'rose', 'sunflower', 'tulip', 'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper', 'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 'bear', 'leopard', 'lion', 'tiger', 'wolf', 'bridge', 'castle', 'house', 'road', 'skyscraper', 'cloud', 'forest', 'mountain', 'plain', 'sea', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'crab', 'lobster', 'snail', 'spider', 'worm', 'baby', 'boy', 'girl', 'man', 'woman', 'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', 'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'])

     args.task1 = {'class_id':task1_idx, 'out_id':0}
     args.task2 = {'class_id': task3_idx, 'out_id': 1}

     model = vgg16_multitask([args.task1['class_id'].__len__(), args.task2['class_id'].__len__()])
     model.cuda()
     print(model)

     logger = Logger(dir_path=args.logs, fname=exp_name+'_t1',
                    keys=['time', 'acc1', 'acc5', 'ce_loss'])
     logger.one_time({'seed':args.manualSeed, 'comments': 'Train task1'})
     logger.set_names(['lr', 'train_stats', 'test_stats'])

     testing_loop(model=model, args=args, task=args.task1['class_id'], out_id=args.task1['out_id'], keys=logger.keys)
     model = training_loop(model=model, logger=logger, args=args, save_best=True)

     logger = Logger(dir_path=args.logs, fname=exp_name+'_t2',
                    keys=['time', 'acc1', 'acc5', 'ce_loss'])
     logger.one_time({'seed':args.manualSeed, 'comments': 'Train task 2 by optimizing all layers with 0.1 mse_err weight'})
     logger.set_names(['lr', 'train_stats', 'test_stats_nt', 'test_stats_ot'])

     training_loop_subtask(model=model, logger=logger, args=args, save_best=True)

if __name__ == '__main__':
    main()