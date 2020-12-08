import argparse
import torch
import time
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.distributed as dist
from fl_utils import printer, time_since

import fl_datasets
import fl_utils
import fl_models
from fl_train_test import train, test
import speed
# import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--world_size', type=int, default=2, metavar='N',
                        help='number of working devices (default: 2)')
parser.add_argument('--rank', type=int, default=1, metavar='N',
                        help='device index (default: 1)')
parser.add_argument('--addr', type=str, default='192.168.0.100', metavar='N',
                        help='master  ip address')
parser.add_argument('--port', type=str, default='23333',metavar='N',
                        help='port number')
parser.add_argument('--enable_vm_test', action="store_true", default=True)
parser.add_argument('--dataset_type', type=str, default='MNIST',metavar='N',
                        help='dataset type, default: MNIST')
parser.add_argument('--model_type', type=str, default='LR',metavar='N',
                        help='model type, default: Linear Regression')
parser.add_argument('--alpha', type=float, default=1.0,metavar='N',
                        help='The value of alpha')
parser.add_argument('--pattern_idx', type=int, default= 0, metavar='N',
                        help='0: IID, 1: Low-Non-IID, 2: Mid-Non-IID, 3: High-Non-IID')
parser.add_argument('--datanum_idx', type=int, default= 0, metavar='N',
                        help='0: 6000, 1: 4,000-8,000, 2: 1,000-11,000')
parser.add_argument('--epoch_start', type=int, default=0,
                    help='')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
parser.add_argument('--local_iters', type=int, default=1, metavar='N',
                        help='input local interval for training (default: 1)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')

args = parser.parse_args()
world_size = args.world_size
rank = args.rank
print("rank is: ", rank)
dist.init_process_group(backend="gloo",
                        init_method="tcp://"+args.addr+":"+args.port,
                        world_size=world_size+1,
                        rank=rank) 
print("init success!")
# # train_loss_plot = []
# test_loss_plot = []
# test_acc_plot = []

# exit conditions
exit_loss_threshold = 1.9 # loss threshold
loss_interval = 10 # the mean of last loss_iterval loss is calculated

def split_data(dataset):
    num_samples = len(dataset.data)
    temp_data = []
    temp_target = []
    dataset.data=torch.div(dataset.data/255.0-0.1307, 0.3081)
    for i in range(num_samples):
        if torch.randint(world_size, [1]) == rank-1:
            temp_data .append( dataset.data[i,:,:])
            temp_target.append(dataset.targets[i])
    return torch.utils.data.TensorDataset(torch.stack(temp_data), torch.stack(temp_target))

def load_data():
   global_train_dataset =  datasets.MNIST('./data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
   global_test_dataset =  datasets.MNIST('./data', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
   return split_data(global_train_dataset), split_data(global_test_dataset)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     start_time = time.time()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = torch.reshape(data.to(device), [-1, 1, 28, 28]), target.to(device)
#         # print(data.type())
#         # print(data.data.size())
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             train_loss_plot.append(loss.item())
#             printer('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss.item()),)
#     end_time = time.time()
#     train_time = end_time - start_time

#     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss.item()))


# def test(args, model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = torch.reshape(data.to(device), [-1, 1, 28, 28]), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     # test_loss /= len(test_loader.dataset)
#     # test_loss_plot.append(test_loss)
#     # test_acc_plot.append(correct / len(test_loader.dataset))

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

def apply_global_para(model, global_para):
    para_dict = model.state_dict()
    keys=list(model.state_dict())
    for i in range(len(keys)):
        para_dict.update({keys[i]: global_para[i]})
    model.load_state_dict(para_dict)
    del para_dict

def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    chunk_size = torch.tensor(0)
    dist.recv(chunk_size, src = 0)
    num_data = chunk_size.item()

    is_train = True

    alpha = args.alpha

    pattern_list = ['random', 'lowbias', 'midbias', 'highbias']

    datanum_list = ['balance', 'lowimbalance', 'highimbalance']

    checkpoint_dir = '/data/jcliu/FL/RE-AFL/client_result/client_' + str(args.rank) + '/'
    print("Create client dir success")
    fl_utils.create_dir(checkpoint_dir)

    fig_dir = checkpoint_dir + 'figure/'
    fl_utils.create_dir(fig_dir)

    MODEL_PATH = checkpoint_dir + 'model/'
    fl_utils.create_dir(MODEL_PATH)

    LOAD_MODEL_PATH = MODEL_PATH + 'alpha_' + str(alpha) + '_model-type_' + args.model_type + '_dataset-type' + args.dataset_type + \
        '_batch-size' + str(args.batch_size) + '_tx2nums' + str(args.world_size) + '_' + pattern_list[args.pattern_idx] + 'data-pattern' + \
            datanum_list[args.datanum_idx] + 'data' + '_exit-loss' + str(exit_loss_threshold) + '_lr' + str(args.lr) + '_epoch' + str(args.epochs) + '_local' + str(args.local_iters) + '.pth'

    SAVE_MODEL_PATH = MODEL_PATH + 'alpha_' + str(alpha) + '_model-type_' + args.model_type + '_dataset-type' + args.dataset_type + \
        '_batch-size' + str(args.batch_size) + '_tx2nums' + str(args.world_size) + '_' + pattern_list[args.pattern_idx] + 'data-pattern' + \
            datanum_list[args.datanum_idx] + 'data' + '_exit-loss' + str(exit_loss_threshold)  + '_lr' + str(args.lr) + '_epoch' + str(args.epochs) + '_local' + str(args.local_iters) + '.pth'

    LOG_ROOT_PATH = checkpoint_dir +  'log/' + '/alpha_' + str(alpha) + '/model-type_' + args.model_type + '_dataset-type' + args.dataset_type + \
        '_batch-size' + str(args.batch_size) + '_tx2nums' + str(args.world_size) + '_' + pattern_list[args.pattern_idx] + 'data-pattern' + \
            datanum_list[args.datanum_idx] + 'data' + '_exit-loss' + str(exit_loss_threshold) + '_lr' + str(args.lr) + '_epoch' + str(args.epochs) + '_local' + str(args.local_iters) +'/'

    fl_utils.create_dir(LOG_ROOT_PATH)

    LOG_PATH = LOG_ROOT_PATH + 'model_acc_loss.txt'

    log_out = open(LOG_PATH, 'w+')
    # if args.epoch_start == 0:
    #     log_out.write("%s\n" % LOG_PATH)

    # if not args.epoch_start == 0:
    #     model.load_state_dict(torch.load(LOAD_MODEL_PATH))

    # log_out = dict()
    # log_out["model_acc_loss"] = open(os.path.join(LOG_ROOT_PATH, "model_acc_loss.txt"), 'w+')

    # <--Load datasets
    train_dataset, test_dataset = fl_datasets.load_datasets(
        args.dataset_type)


    # train_dataset, test_dataset = load_data()
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)
    pattern_idx = args.pattern_idx
    datanum_idx = args.datanum_idx


# <--Create federated train/test loaders for virtrual machines
    if pattern_idx == 0:  # random data (IID)
        if datanum_idx !=0: # imbalance data
            is_train = True
            tx2_train_loader = fl_utils.create_random_loader(
                args, kwargs, args.rank, num_data, is_train, train_dataset)
            is_train = False
            tx2_test_loader = fl_utils.create_random_loader(
                args, kwargs, args.rank, num_data, is_train, test_dataset)
        else:# balance data
            is_train = True
            tx2_train_loader = fl_utils.create_segment_loader(
                args, kwargs, args.world_size, args.rank, is_train, train_dataset)
            is_train = False
            tx2_test_loader = fl_utils.create_segment_loader(
                args, kwargs, args.world_size, args.rank, is_train, test_dataset)

    else:  # bias data partition (Non-IID)
        if pattern_idx == 1:  # lowbias
            label_clusters = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9))
        elif pattern_idx == 2:  # midbias
            label_clusters = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
        elif pattern_idx == 3:  # highbias
            label_clusters = ((0,), (1,), (2,), (3,), (4,),
                            (5,), (6,), (7,), (8,), (9,))

        class_num = len(train_dataset.classes)
        cluster_len = len(label_clusters)

        for idx in range(cluster_len):
            train_data_tmp, train_targets_tmp = fl_utils.create_bias_selected_data(
                args, label_clusters[idx], train_dataset)
            test_data_tmp, test_targets_tmp = fl_utils.create_bias_selected_data(
                args, label_clusters[idx], test_dataset)
            if idx == 0:
                train_data = train_data_tmp
                train_targets = train_targets_tmp

                test_data = test_data_tmp
                test_targets = test_targets_tmp
            else:
                train_data = np.vstack((train_data, train_data_tmp))
                train_targets = np.hstack((train_targets, train_targets_tmp))

                test_data = np.vstack((test_data, test_data_tmp))
                test_targets = np.hstack((test_targets, test_targets_tmp))

        new_train_dataset = fl_datasets.train_test_dataset(
            train_data, train_targets, class_num)

        new_test_dataset = fl_datasets.train_test_dataset(
            test_data, test_targets, class_num)

        is_train = True
        tx2_train_loader = fl_utils.create_segment_loader(
            args, kwargs, args.world_size, args.rank, is_train, new_train_dataset)
        is_train = False
        tx2_test_loader = fl_utils.create_segment_loader(
            args, kwargs, args.world_size, args.rank, is_train, new_test_dataset)
    del train_dataset
    del test_dataset
    #test loader

    # self.test_loader = fl_utils.create_ps_test_loader(
    #     args, kwargs, self.param_server, test_dataset)


    # pattern_list = ['bias', 'partition', 'random']
    # pattern_idx = args.pattern_idx

    # # <--Create federated train/test loaders for virtrual machines
    # if pattern_idx == 0:
    #     # class_num = len(train_dataset.classes)
    #     # step = np.int32(np.floor(class_num / args.vm_num))
    #     if args.world_size == 5:
    #         # <--the number of items must equals to args.vm_num
    #         self.selected_idxs = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
    #     elif args.world_size == 10:
    #         # <--the number of items must equals to args.vm_num
    #         self.selected_idxs = (
    #             (0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,))
    #     else:
    #         class_num = len(train_dataset.classes)
    #         step = np.int32(np.floor(class_num / args.vm_num))
    #         self.selected_idxs = [
    #             [idx + n for n in range(step)] for idx in range(0, class_num - step + 1, step)]

    #     is_train = True
    #     self.vm_train_loaders = fl_utils.create_bias_federated_loader(
    #         args, kwargs, self.vm_list, is_train, train_dataset, self.selected_idxs)
    #     is_train = False
    #     self.vm_test_loaders = fl_utils.create_bias_federated_loader(
    #         args, kwargs, self.vm_list, is_train, test_dataset, self.selected_idxs)

    # elif pattern_idx == 1:
    #     # <--the number of items must equals to args.vm_num
    #     partition_ratios = [1/2, 1/4, 1/8, 1/16, 1/16]
    #     is_train = True
    #     self.vm_train_loaders = fl_utils.create_labelwise_federated_loader(
    #         args, kwargs, self.vm_list, is_train, train_dataset, partition_ratios)
    #     is_train = False
    #     self.vm_test_loaders = fl_utils.create_labelwise_federated_loader(
    #         args, kwargs, self.vm_list, is_train, test_dataset, partition_ratios)

    # else:
    #     is_train = True
    #     self.vm_train_loaders = fl_utils.create_segment_federated_loader(
    #         args, kwargs, self.vm_list, is_train, train_dataset)
    #     is_train = False
    #     self.vm_test_loaders = fl_utils.create_segment_federated_loader(
    #         args, kwargs, self.vm_list, is_train, test_dataset)


# <--Create Neural Network model instance
    if args.dataset_type == 'FashionMNIST':
        if args.model_type == 'LR':
            model = fl_models.MNIST_LR_Net().to(device)
        else:
            model = fl_models.MNIST_Net().to(device)

    elif args.dataset_type == 'MNIST':
        if args.model_type == 'LR':
            model = fl_models.MNIST_LR_Net().to(device)
        else:
            model = fl_models.MNIST_Small_Net().to(device)

    elif args.dataset_type == 'CIFAR10':

        if args.model_type == 'Deep':
            model = fl_models.CIFAR10_Deep_Net().to(device)
            args.decay_rate = 0.98
        else:
            model = fl_models.CIFAR10_Net().to(device)
            args.decay_rate = 0.98

    elif args.dataset_type == 'Sent140':

        if args.model_type == 'LSTM':
            model = fl_models.Sent140_Net().to(device)
            args.decay_rate = 0.99
        else:
            model = fl_models.Sent140_Net().to(device)
            args.decay_rate = 0.99
    else:
        pass

    model_layers_num = len(list(model.named_parameters()))

    if not args.epoch_start == 0:
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))

    print("Model and Dataset ok")
    #model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # global_para  =  model.state_dict().copy()
    # global_para = list(model.state_dict())
    global_para = [para[1].data for para in model.named_parameters()]

    start = time.time()

    for j in range(len(global_para)):
        temp = global_para[j].to('cpu')
        dist.recv(temp, src=0)
        global_para[j] = temp.to('cuda')
    global_epoch = torch.tensor(0)
    dist.recv(global_epoch, src=0)
    
    Speedp = speed.NetSpeed()
    # print("Recev global para from the server")
    apply_global_para(model, global_para)

    for epoch in range(1, args.epochs + 1):
        print("Epoch %d"%epoch)
        # plt.ioff()
        epoch_start_time = time.time()
        train(args, start, model, device, tx2_train_loader, tx2_test_loader, optimizer, epoch, log_out)
        epoch_end_time = time.time()
        print("train ok")
        global_para = [para[1].data for para in model.named_parameters()]
        # local_para = [para[1].data for para in model.named_parameters()]

        tx_start_KB = Speedp.get_rx_tx(device = 'wlan0', local = 'en')[1]
        for j in range(len(global_para)):
            dist.send(global_para[j].to('cpu'), dst=0)
        tx_end_KB = Speedp.get_rx_tx(device = 'wlan0', local = 'en')[1]

        dist.send(torch.tensor(epoch_end_time-epoch_start_time), dst=0)
        dist.send(torch.tensor(tx_end_KB-tx_start_KB), dst=0)
        # print("Send para to the server")
        for j in range(len(global_para)):
            temp = global_para[j].to('cpu')
            dist.recv(temp, src=0)
            global_para[j] = temp.to('cuda')
        dist.recv(global_epoch, src=0)
        # print("recved server epoch: ", global_epoch)
        if global_epoch == args.epochs:
            break
        apply_global_para(model, global_para)
        # print("Recev global para from the server")
        # print("Recev global para from the server")
        #test(args, model, device, test_loader, epoch, self.log_out["model_acc_loss"])
        # plt.close('all')
    # dist.send(torch.Tensor(1), dst=0)

    #     plt.figure(1)
        
    #     plt.ion()
    #     plt.plot(train_loss_plot)
    #     plt.title('Train loss')
    #     plt.xlabel('batches(*10)')
    #     plt.ylabel('train loss')
    #     plt.show()
    
    # plt.ioff()
    # plt.figure()
    # plt.plot(test_loss_plot)
    # plt.title('Test loss')
    # plt.xlabel('epochs')
    # plt.ylabel('test loss')
    # plt.savefig('./test_loss.png')
    # plt.figure()
    # plt.plot(test_acc_plot)
    # plt.title('Test accuracy')
    # plt.xlabel('epochs')
    # plt.ylabel('test accuracy')
    # plt.savefig('./test_acc.png')
    # plt.ioff()
    # plt.show()
    # if (args.save_model):
    #     torch.save(model.state_dict(), "mnist_cnn.pt")
main()
