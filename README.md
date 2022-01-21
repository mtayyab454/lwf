# Learning without Forgetting ([Link](https://arxiv.org/abs/1606.09282)).

This is a very basic implementation of the foundational paper in lifelong learning domain. I implemented this for a project I have been working on recently.

## Experimental Setup

Dataset Used: CIFAR100

Model: VGG16

Task1: Class 1 to 90

Task2: Class 90 to 100

<div align=center><img src="img/framework.png" height = "60%" width = "70%"/></div>

##### 1. Resnet56

| Flops         | Parameters      | Accuracy |
|---------------|-----------------|----------|
|89.80M(64.22%) | 0.32M(62.97%)   | 92.71%   | 

```shell
python run_cifar.py \
--jobid resnet56_test \
--arch resnet56 \
--dataset cifar10 \
--compress_rate :[6,4,4,6,4,4,4,4,4,4,4,4,4,13,4,10,6,4,4,12,18,16,4,15,4,16,4,12,7,13,4,15,4,18,4,12,4,32,26,36,16,32,13,29,23,32,16,36,10,23,13,20,10,13,7] \
--l2_weight 0.001 \
--add_bn True \
--epochs 120 \
--schedule 30 60 90 \
--lr 0.01
```
