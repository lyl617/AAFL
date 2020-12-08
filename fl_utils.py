__author__ = 'yang.xu'

import os
import time
import math
import numpy as np
import numpy.ma as ma
import random
import re
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
#from syft.frameworks.torch.nn import GRU, LSTM, RNN
from torch.utils.data import TensorDataset, DataLoader

#import syft as sy  # <--NEW: import the PySyft library

import fl_datasets

# <--NEW: hook PyTorch ie add extra functionalities to support Federated Learning
#hook = sy.TorchHook(torch)

# <--Tool functions
font1 = {'color':  'black',
        'weight': 'normal',
        'size': 16,
}

def printer(content, fid):
    print(content)
    content = content.rstrip('\n') + '\n'
    fid.write(content)
    fid.flush()


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{:>3}m {:2.0f}s'.format(m, s)


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def create_vm_by_id(id_str):
    vm_instance = sy.VirtualWorker(hook, id=id_str)

    return vm_instance


def create_vm(vm_num=2):
    vm_list = list()
    for vm_idx in range(vm_num):
        vm_id = 'vm'+str(vm_idx)
        vm_instance = create_vm_by_id(vm_id)
        vm_list.append(vm_instance)
    return vm_list


def create_bias_selected_data(args, selected_idxs, dataset):

    if not isinstance(dataset.targets, np.ndarray):
        dataset.targets = np.array(dataset.targets)

    if not isinstance(dataset.data, np.ndarray):
        dataset.data = np.array(dataset.data)

    indices = np.isin(dataset.targets, selected_idxs).astype("bool")
    selected_targets = dataset.targets[indices]
    selected_data = dataset.data[indices]
    # selected_targets = dataset.targets[indices].copy()
    # selected_data = dataset.data[indices].copy()

    return np.float32(selected_data), np.int64(selected_targets)

def create_bias_loader(args, kwargs, is_train, dataset, selected_idxs):
    vm_loaders = list()
    for vm_idx in range(0, args.vm_num):
        selected_data, selected_targets = create_bias_selected_data(
            args, selected_idxs[vm_idx], dataset)
        if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
            # <--for CIFAR10 & CIFAR100
            selected_data = np.transpose(selected_data, (0, 3, 1, 2))

        data_len = len(selected_data)
        if not args.train_flag:
            print('--[Debug] vm:{}-data len:{}'.format(vm_idx, data_len))

        data_transform = fl_datasets.load_default_transform(args.dataset_type)
        vm_dataset_instance = fl_datasets.VMDataset(
            selected_data, selected_targets, data_transform)

        if is_train:
            vm_loader_instance = DataLoader(  # <--this is now a FederatedDataLoader
                vm_dataset_instance, shuffle=True, batch_size=args.batch_size,  **kwargs)
        else:
            vm_loader_instance = DataLoader(  # <--this is now a FederatedDataLoader
                vm_dataset_instance, shuffle=False, batch_size=args.test_batch_size, **kwargs)

        vm_loaders.append(vm_loader_instance)

    return vm_loaders

def create_bias_federated_loader(args, kwargs, vm_list, is_train, dataset, selected_idxs):
    vm_loaders = list()
    for vm_idx in range(0, args.vm_num):
        selected_data, selected_targets = create_bias_selected_data(
            args, selected_idxs[vm_idx], dataset)
        if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
            # <--for CIFAR10 & CIFAR100
            selected_data = np.transpose(selected_data, (0, 3, 1, 2))

        data_len = len(selected_data)
        if not args.train_flag:
            print('--[Debug] vm:{}-data len:{}'.format(vm_idx, data_len))

        data_transform = fl_datasets.load_default_transform(args.dataset_type)
        vm_dataset_instance = fl_datasets.VMDataset(
            selected_data, selected_targets, data_transform).federate([vm_list[vm_idx]])

        if is_train:
            vm_loader_instance = sy.FederatedDataLoader(  # <--this is now a FederatedDataLoader
                vm_dataset_instance, shuffle=True, batch_size=args.batch_size,  **kwargs)
        else:
            vm_loader_instance = sy.FederatedDataLoader(  # <--this is now a FederatedDataLoader
                vm_dataset_instance, shuffle=False, batch_size=args.test_batch_size, **kwargs)

        vm_loaders.append(vm_loader_instance)

    return vm_loaders


'''
def create_bias_federated_loader(args, kwargs, vm_list, is_train, dataset, selected_idxs):
    vm_loaders = list()
    for vm_idx in range(0, args.vm_num, 2):
        selected_data, selected_targets = create_bias_selected_data(args, selected_idxs[int(vm_idx/2)], dataset)
        data_len = len(selected_data)
        print('--[Debug] vm:{}-data len:{}'.format(vm_idx, data_len))
        data_transform = fl_datasets.load_default_transform(args.dataset_type)
        vm_dataset_instance = fl_datasets.VMDataset(selected_data, selected_targets, data_transform).federate([vm_list[vm_idx]])
        if is_train:
            vm_loader_instance = sy.FederatedDataLoader( #<--this is now a FederatedDataLoader 
                                 vm_dataset_instance, shuffle = True, batch_size = args.batch_size, **kwargs)
        else:
            vm_loader_instance = sy.FederatedDataLoader( #<--this is now a FederatedDataLoader 
                                 vm_dataset_instance, shuffle = True, batch_size = args.vm_test_batch_size, **kwargs)
                                 
        vm_loaders.append(vm_loader_instance)

        print('--[Debug] vm:{}-data len:{}'.format(vm_idx + 1, data_len))
        data_transform = fl_datasets.load_customized_transform(args.dataset_type)
        vm_dataset_instance = fl_datasets.VMDataset(selected_data, selected_targets, data_transform).federate([vm_list[vm_idx + 1]])
        if is_train:
            vm_loader_instance = sy.FederatedDataLoader( #<--this is now a FederatedDataLoader 
                                 vm_dataset_instance, shuffle = True, batch_size = args.batch_size, **kwargs)
        else:
            vm_loader_instance = sy.FederatedDataLoader( #<--this is now a FederatedDataLoader 
                                 vm_dataset_instance, shuffle = True, batch_size = args.vm_test_batch_size, **kwargs)
                                 
        vm_loaders.append(vm_loader_instance)
        
    return vm_loaders

def create_bias_selected_data(args, selected_idxs, dataset):
    
    #<--create initial empty datasets and targets numpy.ndarray
    targets_array = np.array(dataset.targets)
    targets_shape = list(targets_array.shape)
    #print(targets_array.shape)
    targets_shape[0] = 0
    init_targets_shape = tuple(targets_shape)
    selected_targets = np.empty(init_targets_shape)
    
    data_shape = list(dataset.data.shape)
    data_shape[0] = 0
    init_data_shape = tuple(data_shape)
    selected_data = np.empty(init_data_shape)
    
    for idx in range(len(selected_idxs)):
        #print(len(targets_array[targets_array == selected_idxs[idx]]))
        #print(targets_array[targets_array == selected_idxs[idx]].shape)
        #print(dataset.data[targets_array == selected_idxs[idx]].shape)
        
        selected_targets = np.concatenate((selected_targets, targets_array[targets_array == selected_idxs[idx]]),axis = 0)
        selected_data = np.concatenate((selected_data, dataset.data[targets_array == selected_idxs[idx]]), axis = 0)
        
    if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
        selected_data = np.transpose(selected_data, (0, 3, 1, 2)) #<--for CIFAR10 & CIFAR100
    
    return np.float32(selected_data), np.int64(selected_targets)
'''

def create_random_selected_data(args, num_data, dataset):

    data_idx = [i for i in range(len(dataset.data))]
    random.shuffle(data_idx)

    select_data_idx = data_idx[:num_data]

    selected_targets = dataset.targets[select_data_idx]
    selected_data = dataset.data[select_data_idx]

    # selected_targets = dataset.targets[begin_idx:end_idx].copy()
    # selected_data = dataset.data[begin_idx:end_idx].copy()

    if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
        # <--for CIFAR10 & CIFAR100
        selected_data = np.transpose(selected_data, (0, 3, 1, 2))

    return np.float32(selected_data), np.int64(selected_targets)

def create_segment_selected_data(args, begin_idx, end_idx, dataset):

    selected_targets = dataset.targets[begin_idx:end_idx]
    selected_data = dataset.data[begin_idx:end_idx]
    # selected_targets = dataset.targets[begin_idx:end_idx].copy()
    # selected_data = dataset.data[begin_idx:end_idx].copy()

    if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
        # <--for CIFAR10 & CIFAR100
        selected_data = np.transpose(selected_data, (0, 3, 1, 2))

    return np.float32(selected_data), np.int64(selected_targets)


def create_labelwise_selected_data(args, label_wise_data, label_wise_targets):
    class_num = len(label_wise_targets)
    # <--create initial empty datasets and targets numpy.ndarray
    targets_shape = list(label_wise_targets[0][0].shape)
    # print(targets_array.shape)
    targets_shape[0] = 0
    init_targets_shape = tuple(targets_shape)
    selected_targets = np.empty(init_targets_shape)

    data_shape = list(label_wise_data[0][0].shape)
    data_shape[0] = 0
    init_data_shape = tuple(data_shape)
    selected_data = np.empty(init_data_shape)

    for idx in range(class_num):
        slice_idxs = list(range(len(label_wise_targets[idx])))
        random.shuffle(slice_idxs)
        selected_targets = np.concatenate(
            (selected_targets, label_wise_targets[idx].pop(slice_idxs[0])), axis=0)
        selected_data = np.concatenate(
            (selected_data, label_wise_data[idx].pop(slice_idxs[0])), axis=0)

    if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
        # <--for CIFAR10 & CIFAR100
        selected_data = np.transpose(selected_data, (0, 3, 1, 2))

    return np.float32(selected_data), np.int64(selected_targets)

##################

def create_random_loader(args, kwargs, tx2_idx, num_data, is_train, dataset):
    data_len = len(dataset.targets)
    #data_len = len(train_targets_array)
    print('--[Debug] tx2:{}- num_data:{}'.format(tx2_idx, num_data))


    selected_data, selected_targets = create_random_selected_data(
        args, num_data, dataset)

    #print('--[Debug] tx2:{}-piece len:{}'.format(tx2_idx, len(selected_targets)))

    data_transform = fl_datasets.load_default_transform(args.dataset_type)

    vm_dataset_instance = fl_datasets.VMDataset(selected_data, selected_targets, data_transform)
    if is_train:
        vm_loader = DataLoader(  # <--this is now a DataLoader
            vm_dataset_instance, shuffle=True, batch_size=args.batch_size, **kwargs)
    else:
        vm_loader = DataLoader(  # <--this is now a DataLoader
            vm_dataset_instance, shuffle=False, batch_size=args.test_batch_size, **kwargs)

    return vm_loader

# <--Partition the whole dataset into len(vm_list) equal-length pieces
# and create federated train/test dataloader
def create_segment_loader(args, kwargs, num_tx2, tx2_idx, is_train, dataset):
    data_len = len(dataset.targets)
    #data_len = len(train_targets_array)
    inter_num = np.int32(np.floor(data_len / num_tx2))
    tx2_idx = tx2_idx - 1
    begin_idx = tx2_idx * inter_num
    if tx2_idx != num_tx2 - 1:
        end_idx = (tx2_idx + 1) * inter_num
    else:
        end_idx = data_len

    print('--[Debug] tx2:{}-begin idx:{}'.format(tx2_idx, begin_idx))
    print('--[Debug] tx2:{}-end idx:{}'.format(tx2_idx, end_idx))

    selected_data, selected_targets = create_segment_selected_data(
        args, begin_idx, end_idx, dataset)

    print('--[Debug] tx2:{}-piece len:{}'.format(tx2_idx, len(selected_targets)))

    data_transform = fl_datasets.load_default_transform(args.dataset_type)

    vm_dataset_instance = fl_datasets.VMDataset(selected_data, selected_targets, data_transform)
    if is_train:
        vm_loader = DataLoader(  # <--this is now a DataLoader
            vm_dataset_instance, shuffle=True, batch_size=args.batch_size, **kwargs)
    else:
        vm_loader = DataLoader(  # <--this is now a DataLoader
            vm_dataset_instance, shuffle=False, batch_size=args.test_batch_size, **kwargs)

    return vm_loader

# <--Partition the whole dataset into len(vm_list) equal-length pieces
# and create federated train/test dataloader
def create_segment_federated_loader(args, kwargs, vm_list, is_train, dataset):
    vm_loaders = list()
    data_len = len(dataset.targets)
    #data_len = len(train_targets_array)
    inter_num = np.int32(np.floor(data_len / len(vm_list)))
    for vm_idx in range(len(vm_list)):
        begin_idx = vm_idx * inter_num
        if vm_idx != len(vm_list) - 1:
            end_idx = (vm_idx + 1) * inter_num
        else:
            end_idx = data_len

        print('--[Debug] vm:{}-begin idx:{}'.format(vm_idx, begin_idx))
        print('--[Debug] vm:{}-end idx:{}'.format(vm_idx, end_idx))

        selected_data, selected_targets = create_segment_selected_data(
            args, begin_idx, end_idx, dataset)

        print('--[Debug] vm:{}-piece len:{}'.format(vm_idx, len(selected_targets)))

        data_transform = fl_datasets.load_default_transform(args.dataset_type)

        vm_dataset_instance = fl_datasets.VMDataset(
            selected_data, selected_targets, data_transform).federate([vm_list[vm_idx]])
        if is_train:
            vm_loader_instance = sy.FederatedDataLoader(  # <--this is now a FederatedDataLoader
                vm_dataset_instance, shuffle=True, batch_size=args.batch_size, **kwargs)
        else:
            vm_loader_instance = sy.FederatedDataLoader(  # <--this is now a FederatedDataLoader
                vm_dataset_instance, shuffle=False, batch_size=args._test_batch_size, **kwargs)

        vm_loaders.append(vm_loader_instance)

    return vm_loaders


def create_labelwise_federated_loader(args, kwargs, vm_list, is_train, dataset, partition_ratios):
    vm_loaders = list()
    class_num = len(dataset.classes)
    label_wise_data = [[] for idx in range(class_num)]
    label_wise_targets = [[] for idx in range(class_num)]
    targets_array = np.array(dataset.targets)
    for c_idx in range(class_num):
        label_targets = targets_array[targets_array == c_idx]
        label_data = dataset.data[targets_array == c_idx]
        label_item_num = len(label_targets)
        begin_idx = 0
        for pr_idx in range(len(partition_ratios)):
            if pr_idx == len(partition_ratios) - 1:
                end_idx = label_item_num
            else:
                end_idx = np.min(
                    (begin_idx + np.int32(np.floor(label_item_num * partition_ratios[pr_idx])), label_item_num))
            print('--[Debug] begin_idx: {} end_idx: {}'.format(begin_idx, end_idx))
            label_wise_targets[c_idx].append(label_targets[begin_idx:end_idx])
            label_wise_data[c_idx].append(label_data[begin_idx:end_idx])
            print('--[Debug] label_data len:',
                  len(label_data[begin_idx:end_idx]))
            begin_idx = end_idx

    for vm_idx in range(len(vm_list)):
        selected_data, selected_targets = create_labelwise_selected_data(
            args, label_wise_data, label_wise_targets)
        print('--[Debug] vm:{}-data len:{}'.format(vm_idx, len(selected_data)))

        data_transform = fl_datasets.load_default_transform(args.dataset_type)

        vm_dataset_instance = fl_datasets.VMDataset(
            selected_data, selected_targets, data_transform).federate([vm_list[vm_idx]])

        if is_train:
            vm_loader_instance = sy.FederatedDataLoader(  # <--this is now a FederatedDataLoader
                vm_dataset_instance, shuffle=True, batch_size=args.batch_size, **kwargs)
        else:
            vm_loader_instance = sy.FederatedDataLoader(  # <--this is now a FederatedDataLoader
                vm_dataset_instance, shuffle=False, batch_size=args.test_batch_size, **kwargs)

        vm_loaders.append(vm_loader_instance)

    return vm_loaders


def create_test_loader(args, kwargs, test_dataset):
    #test_data = test_dataset.data
    if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
        # <--for CIFAR10  & CIFAR100
        test_data = np.transpose(test_dataset.data, (0, 3, 1, 2))
    else:
        test_data = test_dataset.data
    test_data = torch.tensor(np.float32(test_data))
    test_targets = torch.tensor(np.int64(test_dataset.targets))
    #print(test_data.shape, test_targets.shape)
    test_loader = DataLoader(TensorDataset(test_data, test_targets),
                             batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return test_loader

def create_server_test_loader(args, kwargs, test_dataset):
    if not isinstance(test_dataset.targets, np.ndarray):
        test_dataset.targets = np.array(test_dataset.targets)

    if not isinstance(test_dataset.data, np.ndarray):
        test_dataset.data = np.array(test_dataset.data)

    if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
        test_dataset.data = np.transpose(
                test_dataset.data, (0, 3, 1, 2))  # <--for CIFAR10 & CIFAR100

    data_transform = fl_datasets.load_default_transform(args.dataset_type)

    vm_dataset_instance = fl_datasets.VMDataset(np.float32(test_dataset.data), np.int64(
        test_dataset.targets), data_transform)

    test_loader = DataLoader(  # <--this is now a FederatedDataLoader
        vm_dataset_instance, shuffle=False, batch_size=args.test_batch_size, **kwargs)

    return test_loader

def create_ps_test_loader(args, kwargs, vm_instance, test_dataset):
    if not isinstance(test_dataset.targets, np.ndarray):
        test_dataset.targets = np.array(test_dataset.targets)

    if not isinstance(test_dataset.data, np.ndarray):
        test_dataset.data = np.array(test_dataset.data)

    if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
        test_dataset.data = np.transpose(
                test_dataset.data, (0, 3, 1, 2))  # <--for CIFAR10 & CIFAR100

    data_transform = fl_datasets.load_default_transform(args.dataset_type)

    vm_dataset_instance = fl_datasets.VMDataset(np.float32(test_dataset.data), np.int64(
        test_dataset.targets), data_transform).federate([vm_instance])

    test_loader = sy.FederatedDataLoader(  # <--this is now a FederatedDataLoader
        vm_dataset_instance, shuffle=False, batch_size=args.test_batch_size, **kwargs)

    return test_loader


def create_centralized_train_test_loader(args, kwargs, vm_instance, vm_dataset, is_test=False):
    # if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
    #     test_data = np.transpose(test_dataset.data, (0, 3, 1, 2)) #<--for CIFAR10  & CIFAR100
    # else:
    #     test_data = test_dataset.data
    if not isinstance(vm_dataset.targets, np.ndarray):
        vm_dataset.targets = np.array(vm_dataset.targets)

    if not isinstance(vm_dataset.data, np.ndarray):
        vm_dataset.data = np.array(vm_dataset.data)

    if args.dataset_type == 'FashionMNIST':
        data_transform = None
    else:
        data_transform = fl_datasets.load_default_transform(args.dataset_type)

    vm_dataset_instance = fl_datasets.VMDataset(np.float32(vm_dataset.data), np.int64(
        vm_dataset.targets), data_transform).federate([vm_instance])

    if is_test:
        vm_loader = sy.FederatedDataLoader(  # <--this is now a FederatedDataLoader
            vm_dataset_instance, shuffle=False, batch_size=args.test_batch_size, **kwargs)
    else:
        vm_loader = sy.FederatedDataLoader(  # <--this is now a FederatedDataLoader
            vm_dataset_instance, shuffle=True, batch_size=args.batch_size, **kwargs)

    return vm_loader


def add_model(dst_model, src_model):
    """Add the parameters of two models.

        Args:
            dst_model (torch.nn.Module): the model to which the src_model will be added.
            src_model (torch.nn.Module): the model to be added to dst_model.
        Returns:
            torch.nn.Module: the resulting model of the addition.

        """

    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(
                    param1.data + dict_params2[name1].data)
    return dst_model


def scale_model(model, scale):
    """Scale the parameters of a model.

    Args:
        model (torch.nn.Module): the models whose parameters will be scaled.
        scale (float): the scaling factor.
    Returns:
        torch.nn.Module: the module with scaled parameters.

    """
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model


def weight_scale_model(model, weight_dict):
    """Scale the parameters of a model.

    Args:
        model (torch.nn.Module): the models whose parameters will be scaled.
        scale (float): the scaling factor.
    Returns:
        torch.nn.Module: the module with scaled parameters.

    """
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * weight_dict[name])
    return model


def calculate_model_divergence(dst_model, src_model):
    """Add the parameters of two models.

        Args:
            dst_model (torch.nn.Module)
            src_model (torch.nn.Module)
        Returns:
            list(): the weight divergence between dst_model and src_model.

    """
    div_dict = dict()
    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                tmp_dist = torch.norm(
                    param1.data - dict_params2[name1].data, p=2)
                tmp_norm = torch.norm(dict_params2[name1].data, p=2)
                if tmp_norm == 0.0:
                    div_dict[name1] = 0.0
                else:
                    div_dict[name1] = tmp_dist / tmp_norm
                del tmp_dist
                del tmp_norm

    return div_dict


def adjust_learning_rate(args, optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = np.max(0.98 * lr, args.lr * 0.01)
        param_group['lr'] = lr


def plot_learning_curve(file_path, fig_root_path):
    file_in = open(file_path, "r")
    row_data = file_in.readlines()
    row_cnt = len(row_data)

    epoch_data = []
    loss_data = []
    acc_data = []

    if row_cnt <= 0:
        print("--[Debug] No data in log file!")
    else:
        for row in row_data:
            row = row.strip()
            if ('Test set' in row) and (not ('Debug' in row) and not ('-->' in row)):
                tmp = row.split(':')
                epoch_data.append(int(re.findall(r"\d+", tmp[2].strip())[0]))
                loss_data.append(float(tmp[3].split(',')[0]))
                acc_data.append(float(re.findall(r"[1-9]\d*", tmp[4])[0]) / float(re.findall("[1-9]\d*", tmp[4])[1]))


    file_in.close()

    if not len(epoch_data) == 0:

        legend_loc = 'best'

        plt.figure(1)
        plt.plot(epoch_data, loss_data, 'b-')
        plt.xlabel('Epochs', fontdict=font1)
        plt.ylabel('Loss', fontdict=font1)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(('Loss',), loc=legend_loc, fontsize=14)

        plt.grid(True)

        fig_path = fig_root_path.replace("FIGTYPE", 'loss')
        plt.savefig(fig_path, dpi=600, bbox_inches='tight')

        plt.figure(2)
        plt.plot(epoch_data, acc_data, 'b-')
        plt.xlabel('Epochs', fontdict=font1)
        plt.ylabel('Accuracy', fontdict=font1)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(('Accuracy',), loc=legend_loc, fontsize=14)

        plt.grid(True)

        fig_path = fig_root_path.replace("FIGTYPE", 'acc')
        plt.savefig(fig_path, dpi=600, bbox_inches='tight')
