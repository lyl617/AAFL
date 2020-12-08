import argparse
import shutil
import torch
import _thread
import os
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.distributed as dist
import matplotlib.pyplot as plt
from fl_utils import printer, time_since

import fl_datasets
import fl_utils
import fl_models
from fl_train_test import train, test
import threading
import numpy as np
from A2C import ActorCritic

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--world_size', type=int, default=2, metavar='N',
                        help='number of working devices (default: 2)')
parser.add_argument('--rank', type=int, default=0, metavar='N',
                        help='device index (default: 0)')
parser.add_argument('--addr', type=str, default='192.168.0.100', metavar='N',
                        help='master  ip address')
parser.add_argument('--port', type=str, default='23333',metavar='N',
                        help='port number')
parser.add_argument('--enable_vm_test', action="store_true", default=False)
parser.add_argument('--dataset_type', type=str, default='MNIST',metavar='N',
                        help='dataset type, default: MNIST')
parser.add_argument('--alpha', type=float, default=1.0,metavar='N',
                        help='The value of alpha')
parser.add_argument('--update_interval',type=int,default=1,
                    help='')
parser.add_argument('--model_type', type=str, default='LR',metavar='N',
                        help='model type, default: Linear Regression')
parser.add_argument('--pattern_idx', type=int, default= 0, metavar='N',
                        help='0: IID, 1: Low-Non-IID, 2: Mid-Non-IID, 3: High-Non-IID')
parser.add_argument('--datanum_idx', type=int, default= 0, metavar='N',
                        help='0: 6000, 1: 4,000-8,000, 2: 1,000-11,000')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epoch_start', type=int, default=0,
                    help='')
parser.add_argument('--log_save_interval', type=int, default=10, metavar='N',
                        help='the interval of log figure')
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
parser.add_argument('--resource', type=float, default=30000,
                    help='')

args = parser.parse_args()
world_size = args.world_size
rank = args.rank
print("rank is: ", rank)
dist.init_process_group(backend="gloo",
                        init_method="tcp://"+args.addr+":"+args.port,
                        world_size=world_size+1,
                        rank=rank) 
# NUM_RECV_PARAS = 0
test_loss_plot = []
test_acc_plot = []
train_time_list = [] # 记录每轮 time
resource_list = [] #记录每轮 resource
used_paras = []
recv_queue = []
recv_time = []
recv_B = []


# exit conditions
exit_loss_threshold = 0.8272 # loss threshold 1.3024  0.8222 0.5108
loss_interval = 10 # the mean of last loss_iterval loss is calculated

# unused_paras = []
lock = threading.Lock()

def split_data(dataset):
    num_samples = len(dataset.data)
    temp_data = []
    temp_target = []
    dataset.data=torch.div(dataset.data/255.0-0.1307, 0.3081)
    for i in range(num_samples):
        temp_data .append( dataset.data[i,:,:])
        temp_target.append(dataset.targets[i])
    return torch.utils.data.TensorDataset(torch.stack(temp_data), torch.stack(temp_target))

def load_test_data():
    global_test_dataset =  datasets.MNIST('./data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
    return split_data(global_test_dataset)


def apply_global_para(model, global_para):
    para_dict = model.state_dict()
    keys=list(model.state_dict())
    for i in range(len(keys)):
        para_dict.update({keys[i]: global_para[i]})
    model.load_state_dict(para_dict)
    del para_dict

def aggregate_nomalization(global_para, local_paras, num_aggre, alpha):
    # print("Collected paras from %d devices"%len(local_paras))
    for j in range(num_aggre):
        print("src = ", recv_queue[j], ", time = ", recv_time[j], ", B = ", recv_B[j])
    for i in range(len(global_para)):
        len_aggre = 0
        first = True
        for local_para in local_paras:
            # if len_aggre >= num_aggre : break
            if first:
                new_para_i = local_para[i]
            else:
                new_para_i = torch.add(global_para[i], local_para[i])#update the i-th para of global model
            global_para[i] = new_para_i
            len_aggre += 1
            if len_aggre == num_aggre: break
            first = False
        global_para[i] = torch.div(global_para[i], alpha * world_size + 0.0)
    # local_paras.clear()
    return global_para

def remove_alpha_paras(local_paras, alpha):
    num_paras = int(alpha * world_size)
    for i in range(num_paras):
        local_paras.remove(local_paras[i])

def recv_para(local_para, src):
    # print(src,": ", end="")
    for j in range(len(local_para)):
        # print(local_para[j].size(), end=" ")
        temp = local_para[j].to('cpu')
        dist.recv(temp, src=src)
        local_para[j]=temp.to('cuda')
        # del temp
    # if(len(used_paras) < num_paras):
        # lock.acquire()
    
    i_time = torch.tensor(0.0)
    i_B = torch.tensor(0.0)
    dist.recv(i_time, src=src)
    dist.recv(i_B, src=src)

    global recv_queue
    global recv_time
    global recv_B

    lock.acquire()
    recv_queue.append(src)
    recv_time.append(i_time)
    recv_B.append(i_B)

    global used_paras
    used_paras.append(local_para)
    del local_para
    lock.release()
    # else:
    #     unused_paras.append(local_para)
    # local_paras.append(local_para)
    # print(src, end = " ")

def send_para(global_para, epoch_index, dst):
    for j in range(len(global_para)):
        dist.send(global_para[j].to('cpu'), dst=dst)
    dist.send(torch.tensor(epoch_index), dst=dst)

class recv_paras_thread(threading.Thread):
    def __init__(self, src, local_para):
        threading.Thread.__init__(self)
        self.src = src
        self.local_para = local_para
    def run(self):
        recv_para(self.local_para, self.src)

def gen_chunk_sizes(num_clients, datanum_idx):
    if datanum_idx == 0:
        low = 6000
        high = 6000
    else:
        if datanum_idx == 1:
            low = 4000
            high = 8000
        else:
            low = 1000
            high = 11000
    
    raw_sizes = (torch.rand(size=[num_clients])*(high-low)+low).int()

    selected_low_index = (torch.rand([1])*num_clients).long()
    selected_high_index = (torch.rand([1])*num_clients).long()
    while selected_low_index == selected_high_index:
        selected_high_index = (torch.rand([1])*num_clients).long()
    raw_sizes[selected_low_index] = low
    raw_sizes[selected_high_index] = high
    return raw_sizes

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

def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    datanum_idx = args.datanum_idx

    tx2_chunk_sizes = gen_chunk_sizes(world_size, datanum_idx)
    for i in range(world_size):
        dist.send(tx2_chunk_sizes[i], dst=i+1)

    #model = Net().to(device)

    is_train = False

    alpha = args.alpha

    log_save_interval = args.log_save_interval

    args.save_model = True

    pattern_list = ['random', 'lowbias', 'midbias', 'highbias']

    datanum_list = ['balance', 'lowimbalance', 'highimbalance']

    checkpoint_dir = 'server_result/server_equalloss_' + str(args.rank) + '/'
    fl_utils.create_dir(checkpoint_dir)

    fig_dir = checkpoint_dir + 'figure/'
    fl_utils.create_dir(fig_dir)

    FIG_ROOT_PATH = fig_dir + 'alpha_' + str(alpha) + 'model-type_' + args.model_type + '_dataset-type' + args.dataset_type + \
        '_batch-size' + str(args.batch_size) + '_tx2nums' + str(args.world_size) + '_' + pattern_list[args.pattern_idx] + '_data-pattern' + \
            datanum_list[args.datanum_idx] + '_data' + '_exit-loss' + str(exit_loss_threshold) + '_lr' + str(args.lr) + '_epoch' + str(args.epochs) + '_local' + str(args.local_iters) + '_FIGTYPE.png'

    MODEL_PATH = checkpoint_dir + 'model/'
    fl_utils.create_dir(MODEL_PATH)

    LOAD_MODEL_PATH = MODEL_PATH + 'alpha_' + str(alpha) + '_model-type_' + args.model_type + '_dataset-type' + args.dataset_type + \
        '_batch-size' + str(args.batch_size) + '_tx2nums' + str(args.world_size) + '_' + pattern_list[args.pattern_idx] + '_data-pattern' + \
            datanum_list[args.datanum_idx] + '_data' + '_exit-loss' + str(exit_loss_threshold)  + '_lr' + str(args.lr) + '_epoch' + str(args.epochs) + '_local' + str(args.local_iters) + '.pth'

    SAVE_MODEL_PATH = MODEL_PATH + 'alpha_' + str(alpha) + '_model-type_' + args.model_type + '_dataset-type' + args.dataset_type + \
        '_batch-size' + str(args.batch_size) + '_tx2nums' + str(args.world_size) + '_' + pattern_list[args.pattern_idx] + '_data-pattern' + \
            datanum_list[args.datanum_idx] + '_data' + '_exit-loss' + str(exit_loss_threshold) + '_lr' + str(args.lr) + '_epoch' + str(args.epochs) + '_local' + str(args.local_iters) + '.pth'
    

    LOG_ROOT_PATH = checkpoint_dir +  'log/' + 'alpha_' + str(alpha) + '/model-type_' + args.model_type + '_dataset-type' + args.dataset_type + \
        '_batch-size' + str(args.batch_size) + '_tx2nums' + str(args.world_size) + '_' + pattern_list[args.pattern_idx] + '_data-pattern' + \
            datanum_list[args.datanum_idx] + '_data' + '_exit-loss' + str(exit_loss_threshold) + '_lr' + str(args.lr) + '_epoch' + str(args.epochs) + '_local' + str(args.local_iters) +'/' 

    fl_utils.create_dir(LOG_ROOT_PATH)

    LOG_PATH = LOG_ROOT_PATH + 'model_acc_loss.txt'
    LOG_TIME_PATH = LOG_ROOT_PATH + 'train_time.txt'
    LOG_RESOURCE_PATH = LOG_ROOT_PATH + 'train_resource.txt'

    log_out = open(LOG_PATH, 'w+')
    log_out_time = open(LOG_TIME_PATH, 'w+')
    log_out_resource = open(LOG_RESOURCE_PATH, 'w+')

    AGENT_DIR = checkpoint_dir + 'RL_Agent/'
    AGENT_MODEL_PATH = AGENT_DIR + 'A2C.py'
    LOG_STATE_REWARD_PATH = AGENT_DIR + 'STATE_REWARD_log.txt'
    log_alpha_dir = checkpoint_dir + 'alpha/'
    fl_utils.create_dir(AGENT_DIR)
    fl_utils.create_dir(log_alpha_dir)
    log_alpha_path =  log_alpha_dir + 'alpha_' + str(alpha) + args.dataset_type + '_' + args.model_type + '_vm' + str(world_size) + '_' + pattern_list[args.pattern_idx]  + '_data-pattern' + \
            datanum_list[args.datanum_idx] + 'data' + \
               '_lr' + str(args.lr) + '_epoch' + str(args.epochs) + \
               '_vmtest' + str(args.enable_vm_test) + '_alpha.txt'
    log_agent_loss_path =  log_alpha_dir + 'alpha_' + str(alpha) + args.dataset_type + '_' + args.model_type + '_vm' + str(world_size) + '_' + pattern_list[args.pattern_idx]  + '_data-pattern' + \
            datanum_list[args.datanum_idx] + 'data' + \
               '_lr' + str(args.lr) + '_epoch' + str(args.epochs) + \
               '_vmtest' + str(args.enable_vm_test) + '_agent_loss.txt'
    log_alpha_out = open(log_alpha_path, 'a+')
    log_state_reward_out = open(LOG_STATE_REWARD_PATH, 'a+')
    log_agent_loss_out = open(log_agent_loss_path, 'a+')
    reward_weights = [500, 500, 1, 0.02]
    rz = 1.5

    agent = ActorCritic(4, world_size).to(device)
    shutil.copy('../A2C.py', AGENT_MODEL_PATH)
    log_alpha_out.write("%s\n" % alpha)
    log_state_reward_out.write("reward_weights:%s, %s, %s, %s\nrz:%s\n\n\n" % \
            (reward_weights[0], reward_weights[1], reward_weights[2], reward_weights[3], rz))

    current_resource = args.resource
    time_avg = 0
    # if args.epoch_start == 0:
    #     log_out.write("%s\n" % LOG_PATH)

    # log_out = dict()
    # log_out["model_acc_loss"] = open(os.path.join(LOG_ROOT_PATH, "model_acc_loss.txt"), 'w+')

    # <--Load datasets
    train_dataset, test_dataset = fl_datasets.load_datasets(
        args.dataset_type)

    test_loader = fl_utils.create_server_test_loader(args, kwargs, test_dataset)
    
    #test_dataset = load_test_data()
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)

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

    if not args.epoch_start == 0:
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if alpha == 0.0:
        alpha = 1.0

    global_para = [para[1].data for para in model.named_parameters()]
        # for j in range(len(global_para)):
        #     dist.send(global_para[j].to('cpu'), dst=i)


    for i in range(1, world_size+1):
        _thread.start_new_thread(send_para, (global_para, 0, i))
    

    global recv_queue
    global used_paras
    for i in range(1, world_size+1):
        local_para = [para[1].data for para in model.named_parameters()]
        recv_queue.append(i)
        used_paras.append(local_para)
        recv_time.append(torch.tensor(0.0))
        recv_B.append(torch.tensor(0.0))
        del local_para
    num_paras = world_size
    
    start = time.time()
    test(args, start, model, device, test_loader, 0, log_out)
    for epoch in range(1, args.epochs + 1):
        for i in range(num_paras):
            recv_src = recv_queue[0]
            recv_queue.remove(recv_src)            
            recv_time.remove(recv_time[0])
            recv_B.remove(recv_B[0])
            used_paras.remove(used_paras[0])

            local_para = [para[1].data for para in model.named_parameters()]
            recv_thread = recv_paras_thread(src=recv_src, local_para=local_para)
            recv_thread.start()
            # _thread.start_new_thread(recv_para, (local_para, i, int(alpha * world_size)))
            # for j in range(len(global_para)):
            #     temp = local_para[j].to('cpu')
            #     dist.recv(temp, src=i)
            #     local_para[j]=temp.to('cuda')
            #     del temp
            # .append(local_para)
        while True:
            if len(recv_queue) >= int(alpha * world_size):
            # if len(used_paras) == world_size:
                break
        num_paras = int(alpha * world_size)

        total_KB = np.sum([recv_B[i] for i in range(num_paras)])
        max_epoch_time = np.max([recv_time[i] for i in range(num_paras)])
        
        # print(type(total_KB))
        # print(type(max_epoch_time))
        print("total_KB = ", total_KB, ", max_epoch_time = ", max_epoch_time)
        
        global_para = aggregate_nomalization(global_para, used_paras, num_paras, alpha)
        apply_global_para(model, global_para)

        test_loss, test_acc = test(args, start, model, device, test_loader, epoch, log_out)
        test_loss_plot.append(test_loss)
        test_acc_plot.append(test_acc)
        train_time_list.append(max_epoch_time)
        resource_list.append(total_KB)

        log_out_time.write("Epoch:{}, Time:{}\n".format(epoch, max_epoch_time))
        log_out_resource.write("Epoch:{}, Resource:{}\n".format(epoch, total_KB))
        if epoch % log_save_interval == 0:
            fl_utils.plot_learning_curve(LOG_PATH, FIG_ROOT_PATH)
            if (args.save_model):
                # pass
                torch.save(model.state_dict(), SAVE_MODEL_PATH)
            # for j in range(len(global_para)):
            #     dist.send(global_para[j].to('cpu'), dst=i)
        # # plt.close('all')
        for i in range(num_paras):
            _thread.start_new_thread(send_para, (global_para, epoch, recv_queue[i]))
        
        ## exit when loss is lower than pre-defined threshold
        if len(test_loss_plot) < loss_interval:
            # print("1",np.mean(test_loss_plot))
            # print(type(np.mean(test_loss_plot) ))
            # print(type(np.float64(exit_loss_threshold)))
            if np.mean(test_loss_plot) <= np.float64(exit_loss_threshold):
                # print("exit ok")
                break
        else:
            # print("2",np.mean(test_loss_plot[-loss_interval::]))
            # print(type(np.mean(test_loss_plot[-loss_interval::])))
            if np.mean(test_loss_plot[-loss_interval::]) <= np.float64(exit_loss_threshold):
                # print("exit ok") 
                break
        
        #<--update alpha
        update_interval = args.update_interval
        if args.alpha == 0.0 and epoch % update_interval == 0 and epoch > update_interval:
            #TODO: RL and update alpha  
            states = []
            states.append(test_loss_plot[-1-update_interval] - test_loss_plot[-1])
            states.append(test_acc_plot[-1] - test_acc_plot[-1-update_interval])
            states.append(sum(train_time_list[-update_interval:]))
            states.append(sum(resource_list[-update_interval:]))

            if epoch > 2 * update_interval:
                # caculate reward
                time_avg = 0.2 * states[2] + 0.8 * time_avg
                # reward = reward_weights[0]*states[0] + reward_weights[1]*states[1] - \
                #     reward_weights[2]*states[2] - reward_weights[3]*states[3]
                reward1 = - np.power(rz, (states[2] - time_avg) / time_avg)

                reward2 = (exit_loss_threshold - test_loss_plot[-1]) / exit_loss_threshold

                reward3 = - states[3] / current_resource

                current_resource -= states[3]

                reward = reward1 + reward2 + reward3

                log_state_reward_out.write("\t\treward1:%s reward2:%s reward3:%s reward:%s\n" % \
                    (reward1, reward2, reward3, reward))
                # agent.record_reward(torch.Tensor([reward]).unsqueeze(0))
                # agent.train_model(torch.Tensor(states).unsqueeze(0))
                actor_loss, critic_loss, entropy, total_loss = agent.train_model(torch.Tensor([reward]).unsqueeze(0).to(device), torch.Tensor(states).unsqueeze(0).to(device))

                log_agent_loss_out.write("actor_loss:%s, critic_loss:%s, entropy_loss:%s, total_loss:%s\n" % \
                    (actor_loss[0][0], critic_loss[0][0], entropy[0][0], total_loss))
            # print('DEBUG-----------------------------------> no')
            # states = [1,1,1,1]
            actions, values, probs = agent.choose_action(torch.Tensor(states).unsqueeze(0).to(device))
            # print('DEBUG-----------------------------------> yes')
            alpha = actions[0] / world_size
            print("Alpha: ", alpha)
            # alpha = 1.0

            # write into log_alpha_path
            # print("states:", states)
            # print("alpha:", alpha)
            log_alpha_out.write("%s\n" % alpha)
            log_state_reward_out.write("epoch:%s\n\t\tdelta_loss:%s delta_acc:%s time:%s resourse:%s time_avg:%s current_resource:%s\n" % \
                (epoch, states[0], states[1], states[2], states[3], time_avg, current_resource))
            
            log_state_reward_out.write("\t\tvalue:%s\n" % values[0][0])
            log_state_reward_out.write("\t\tprobilities:%s\n" % list(probs[0]))
            log_state_reward_out.write("\t\talpha:%s\n\n" % alpha)

            log_state_reward_out.close()
            log_alpha_out.close()
            log_alpha_out = open(log_alpha_path, 'a+')
            log_state_reward_out = open(LOG_STATE_REWARD_PATH, 'a+')
    # print("escape from the for loop")
    time.sleep(5)

    local_para = [para[1].data for para in model.named_parameters()]
    if(epoch != args.epochs):
        for i in range(num_paras):
            for j in range(len(local_para)):
                # print(local_para[j].size(), end=" ")
                temp = local_para[j].to('cpu')
                dist.recv(temp, src=recv_queue[i])
        num_paras = 0
    # print(recv_queue)
    for i in recv_queue:
        if i not in recv_queue[0:num_paras]:
            # print("send stop to: ", i)
            for j in range(len(local_para)):
                dist.send(local_para[j].to('cpu'), dst=i)
            dist.send(torch.tensor(args.epochs), dst=i)


    
    plt.ioff()
    plt.figure()
    plt.plot(test_loss_plot)
    plt.title('Test loss')
    plt.xlabel('epochs')
    plt.ylabel('test loss')
    plt.savefig(fig_dir + './test_loss.png')
    plt.figure()
    plt.plot(test_acc_plot)
    plt.title('Test accuracy')
    plt.xlabel('epochs')
    plt.ylabel('test accuracy')
    plt.savefig(fig_dir + './test_acc.png')
    plt.ioff()

    # plt.figure()
    # plt.plot(train_time_list)
    # plt.title('Train time')
    # plt.xlabel('epochs')
    # plt.ylabel('train time')
    # plt.savefig(fig_dir + './train_time.png')

    # plt.figure()
    # plt.plot(test_loss_plot)
    # plt.title('Train resource')
    # plt.xlabel('epochs')
    # plt.ylabel('train resource')
    # plt.savefig(fig_dir + './train_resource.png')
    # plt.show()
    # if (args.save_model):
    #     torch.save(model.state_dict(), "mnist_cnn.pt")
main()
