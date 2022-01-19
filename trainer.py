import copy
import os
import torch
import time
import torch.nn as nn
import torch.optim as optim

from utils import AverageAccumulator, VectorAccumulator, accuracy, Progressbar, adjust_learning_rate, get_num_parameters
from datasets import get_cifar_sub_class

mse = nn.MSELoss()
mse.cuda()

def train(trainloader, model, optimizer, criterion, out_id, keys):
    print('Training...')
    model.train()

    accumulator = VectorAccumulator(keys)
    end = time.time()

    for batch_idx, (inputs, _, targets) in enumerate(Progressbar(trainloader)):
        # measure data loading time
        # print(batch_idx)
        inputs = inputs.cuda()
        targets = targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs[out_id], targets)

        # prec1 = sum(model_pred.squeeze(1) == targets)
        prec1, prec5 = accuracy(outputs[out_id].data, targets.data, topk=(1, 5))
        # gt_acc.update(prec1.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        accumulator.update( [(time.time() - end), prec1.item(), prec5.item(), loss.item()])
        end = time.time()

    return accumulator.avg

def test(testloader, model, criterion, out_id, keys):
    print('Testing...')
    # switch to evaluate mode
    model.eval()

    accumulator = VectorAccumulator(keys)
    end = time.time()

    for batch_idx, (inputs, _, targets) in enumerate(Progressbar(testloader)):

        inputs, targets  = inputs.cuda(), targets.cuda()

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs[out_id], targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs[out_id].data, targets.data, topk=(1, 5))

        accumulator.update( [(time.time() - end), prec1.item(), prec5.item(), loss.item()])

        end = time.time()

    return accumulator.avg

def testing_loop(model, args, task, out_id, keys):
    criterion = nn.CrossEntropyLoss()
    _, testloader, num_classes = get_cifar_sub_class(args.dataset, classes=task, split='test', batch_size=args.test_batch, num_workers=args.workers)
    test_stats = test(testloader, model, criterion, out_id, keys)
    print('\nTest loss: %.4f \nVal accuracy: %.2f%%' % (test_stats[3], test_stats[1]))

def training_loop(model, logger, args, save_best=False):
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    ###################### Initialization ###################
    lr = args.lr_t1
    # Load data
    _, trainloader, num_classes = get_cifar_sub_class(args.dataset, classes=args.task1['class_id'], split='train', batch_size=args.train_batch, num_workers=args.workers)
    _, testloader, num_classes = get_cifar_sub_class(args.dataset, classes=args.task1['class_id'], split='test', batch_size=args.test_batch, num_workers=args.workers)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    num_param = get_num_parameters(model)

    print('    Total params: %.2fM' % (num_param / 1000000.0))
    logger.one_time({'num_param': num_param})

    ###################### Main Loop ########################
    best_acc = 0
    for epoch in range(args.epochs_t1):
        lr = adjust_learning_rate(optimizer, lr, epoch, args.schedule_t1, args.gamma)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs_t1, lr))

        train_stats = train(trainloader, model, optimizer, criterion, args.task1['out_id'], logger.keys)
        test_stats = test(testloader, model, criterion, args.task1['out_id'], logger.keys)

        torch.save(model.state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '.pth'))

        if best_acc < test_stats[1]:
            best_acc = test_stats[1]
            if save_best:
                torch.save(model.state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '_best.pth'))

        print('\nKeys: ', logger.keys)
        print('Training: ', train_stats)
        print('Testing: ', test_stats)
        print('Best Acc: ', best_acc)

        logger.append([lr, train_stats, test_stats])

    return model

def train_subtask(trainloader, model, model_copy, optimizer, criterion, args, keys):
    print('Training...')
    model.train()

    accumulator = VectorAccumulator(keys)
    end = time.time()

    for batch_idx, (inputs, _, task2_targets) in enumerate(Progressbar(trainloader)):
        # measure data loading time
        # print(batch_idx)
        inputs = inputs.cuda()
        task2_targets = task2_targets.cuda()

        with torch.no_grad():
            temp_outs = model_copy(inputs)
            task1_targets = temp_outs[args.task1['out_id']].data
        # compute output
        outputs = model(inputs)

        task1_pred = outputs[args.task1['out_id']]
        task2_pred = outputs[args.task2['out_id']]

        mse_err = mse(task1_pred, task1_targets)
        # print(mse_err)
        ce_err = criterion(task2_pred, task2_targets)
        loss =  ce_err + args.t1_weight*mse_err

        # prec1 = sum(model_pred.squeeze(1) == targets)
        prec1, prec5 = accuracy(task2_pred, task2_targets.data, topk=(1, 5))
        # gt_acc.update(prec1.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        accumulator.update( [(time.time() - end), prec1.item(), prec5.item(), loss.item()])
        end = time.time()

    return accumulator.avg

def training_loop_subtask(model, logger, args, save_best=False):
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    model_copy = copy.deepcopy(model)

    ###################### Initialization ###################
    lr = args.lr_t2
    # Load data
    _, trainloader_t2, num_classes = get_cifar_sub_class(args.dataset, classes=args.task2['class_id'], split='train', batch_size=args.train_batch, num_workers=args.workers)
    _, testloader_t2, num_classes = get_cifar_sub_class(args.dataset, classes=args.task2['class_id'], split='test', batch_size=args.test_batch, num_workers=args.workers)
    _, testloader_t1, num_classes = get_cifar_sub_class(args.dataset, classes=args.task1['class_id'], split='test', batch_size=args.test_batch, num_workers=args.workers)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    num_param = get_num_parameters(model)

    print('    Total params: %.2fM' % (num_param / 1000000.0))
    logger.one_time({'num_param': num_param})

    ###################### Main Loop ########################
    best_acc = 0
    for epoch in range(args.epochs_t2):
        lr = adjust_learning_rate(optimizer, lr, epoch, args.schedule_t2, args.gamma)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs_t2, lr))

        train_stats = train_subtask(trainloader_t2, model, model_copy, optimizer, criterion, args, logger.keys)

        test_stats_t2 = test(testloader_t2, model, criterion, 1, logger.keys)
        test_stats_t1 = test(testloader_t1, model, criterion, 0, logger.keys)

        torch.save(model.state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '.pth'))

        if best_acc < test_stats_t2[1]:
            best_acc = test_stats_t2[1]
            if save_best:
                torch.save(model.state_dict(), os.path.join(args.checkpoint, logger.fname, logger.fname + '_best.pth'))

        print('\nKeys: ', logger.keys)
        print('Training: ', train_stats)
        print('Testing New Task: ', test_stats_t2)
        print('Testing Old Task: ', test_stats_t1)
        print('Best Acc: ', best_acc)

        logger.append([lr, train_stats, test_stats_t2, test_stats_t1])

    return model