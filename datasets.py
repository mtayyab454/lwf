import os
import torch
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np
from PIL import Image
import copy
from utils import AverageAccumulator, VectorAccumulator, accuracy, Progressbar, adjust_learning_rate, get_num_parameters

class CIFAR_Sub_Class(data.Dataset):
    def __init__(self, CIFAR_obj, class_idx):

        self.transform = CIFAR_obj.transform
        self.target_transform = CIFAR_obj.target_transform
        self.train = CIFAR_obj.train
        self.org_class_idx = class_idx
        self.org_class_names, self.remapped_class_names = self.get_class_names(CIFAR_obj.class_to_idx, class_idx)

        # Create sub set of data from classes

        # labels = CIFAR_obj.train_labels if self.train else CIFAR_obj.test_labels
        # data = CIFAR_obj.train_data if self.train else CIFAR_obj.test_data

        targets = CIFAR_obj.targets
        data = CIFAR_obj.data

        target_idx = self.get_match_index(targets, class_idx)

        data = data[target_idx, :, :, :]
        temp = np.array([targets])
        targets = temp[0][target_idx].tolist()

        # now load the picked numpy arrays
        self.remapped_targets = self.remap_targets(targets, class_idx)
        self.targets = targets
        self.data = data

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, remapped_target = self.data[index], self.targets[index], self.remapped_targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, remapped_target

    def get_match_index(self, targets, class_idx):
        target_indices = []
        # new_targets = []

        for i in range(len(targets)):
            if targets[i] in class_idx:
                target_indices.append(i)
                # new_targets.append(np.where(np.array(class_idx) == targets[i])[0].item())

        return target_indices

    def remap_targets(self, targets, class_idx):
        remapped_targets = []

        targets = np.array(targets)
        # unique_targets = np.unique(targets)
        for t in targets:
            remapped_targets.append(np.where(class_idx == t)[0][0])

        return remapped_targets

    def get_class_names(self, class_to_idx, class_idx):

        org_class_names = {}
        remapped_class_names = {}

        keys = list(class_to_idx.keys())
        values = list(class_to_idx.values())

        for i, cid in enumerate(class_idx):
            if cid in values:
                org_class_names.update({keys[cid]:values[cid]})
                remapped_class_names.update({keys[cid]:i})

        return org_class_names, remapped_class_names

# class CIFAR_Subtask(data.Dataset):
#     def __init__(self, dset, model, outid):
#
#         self.transform = copy.deepcopy(dset.transform)
#         self.target_transform = copy.deepcopy(dset.target_transform)
#         self.train = copy.deepcopy(dset.train)
#         self.org_class_idx = copy.deepcopy(dset.org_class_idx)
#         self.org_class_names = copy.deepcopy(dset.org_class_names)
#         self.remapped_class_names = copy.deepcopy(dset.remapped_class_names)
#         self.remapped_targets = copy.deepcopy(dset.remapped_targets)
#         self.targets = copy.deepcopy(dset.targets)
#         self.data = copy.deepcopy(dset.data)
#         self.features = self.get_features(dset, model, outid)
#
#     def get_features(self, dset, model, outid):
#
#         batch_size = 128
#         dloader = data.DataLoader(dset, batch_size, shuffle=False, num_workers=100)
#
#         print('Generating features...')
#         # switch to evaluate mode
#         model.eval()
#
#         features = []
#
#         for i in Progressbar(range(len(self.targets))):
#             inputs = dset[i]
#             inputs = inputs[0].unsqueeze(0)
#             inputs = inputs.cuda()
#             # print(batch_idx)
#             # compute output
#             with torch.no_grad():
#                 outputs = model(inputs)
#                 # print(outputs)
#                 features.append(outputs[outid].cpu().data)
#
#         # verify
#         return features
#
#     def __len__(self):
#         return len(self.targets)
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         img, target, remapped_target, features = self.data[index], self.targets[index], self.remapped_targets[index], self.features[index]
#
#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target, remapped_target, features

def get_cifar_data(data_set, split, batch_size=100, num_workers=4):
    print('==> Preparing dataset %s' % data_set)
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    # ])
    #
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    # ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if data_set in ['CIFAR10', 'cifar10']:
        dataloader = datasets.CIFAR10
        num_classes = 10
        trainset = dataloader(root=os.path.join('.','../../data/CIFAR'), train=True, download=True, transform=transform_train)
        testset = dataloader(root=os.path.join('.','../../data/CIFAR'), train=False, download=False, transform=transform_test)

    elif data_set in ['CIFAR100', 'cifar100']:
        dataloader = datasets.CIFAR100
        num_classes = 100
        trainset = dataloader(root=os.path.join('.','../../data/CIFAR'), train=True, download=True, transform=transform_train)
        testset = dataloader(root=os.path.join('.','../../data/CIFAR'), train=False, download=False, transform=transform_test)

    if split == 'train':
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return trainset, trainloader, num_classes
    elif split == 'test':
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return testset, testloader, num_classes

def get_cifar_sub_class(dataset, classes, split, batch_size=100, num_workers=4):
    temp_set, _, _ = get_cifar_data(dataset, split, batch_size, num_workers)
    dset = CIFAR_Sub_Class(temp_set, classes)

    if split == 'train':
        dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    elif split == 'test':
        dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dset, dloader, len(classes)

# def get_cifar_sub_task(dataset, classes, split, batch_size, num_workers, model, outid):
#
#     dset, _, num_classes = get_cifar_sub_class(dataset, classes, split, batch_size, num_workers)
#     dloader = data.DataLoader(CIFAR_Subtask(dset, model, outid), batch_size=2, shuffle=True, num_workers=0)
#
#     for batch_idx, (inputs, _, _, features) in enumerate(Progressbar(dloader)):
#         inputs = inputs.cuda()
#         # print(batch_idx)
#         # compute output
#         with torch.no_grad():
#             outputs = model(inputs)
#             # print(outputs)
#             print((features - outputs[outid].cpu().data).abs().sum())
#
#     return dset, [], num_classes

