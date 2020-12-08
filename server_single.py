import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.distributed as dist
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--world-size', type=int, default=2, metavar='N',
                        help='number of working devices (default: 2)')
parser.add_argument('--rank', type=int, default=0, metavar='N',
                        help='device index (default: 0)')
parser.add_argument('--addr', type=str, default='192.168.0.100', metavar='N',
                        help='master  ip address')
parser.add_argument('--port', type=str, default='22222',metavar='N',
                        help='port number')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')

args = parser.parse_args()
world_size = args.world_size
rank = args.rank
print("rank is: ", rank)
dist.init_process_group(backend="gloo",
                        init_method="tcp://"+args.addr+":"+args.port,
                        world_size=world_size+1,
                        rank=rank) 

test_loss_plot = []
test_acc_plot = []

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

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = torch.reshape(data.to(device), [-1, 1, 28, 28]), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_loss_plot.append(test_loss)
    test_acc_plot.append(correct / len(test_loader.dataset))

    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def apply_global_para(model, global_para):
    para_dict = model.state_dict()
    keys=list(model.state_dict())
    for i in range(len(keys)):
        para_dict.update({keys[i]: global_para[i]})
    model.load_state_dict(para_dict)
    del para_dict

def aggregate_nomalization(global_para, local_paras):

    for i in range(len(global_para)):
        for local_para in local_paras:
            new_para_i = torch.add(global_para[i], local_para[i])
            global_para[i] = new_para_i
        global_para[i] = torch.div(global_para[i], world_size+0.0)
    return global_para

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

    model = Net().to(device)
    test_dataset = load_test_data()
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    global_para = [para[1].data for para in model.named_parameters()]
    for i in range(1, world_size+1):
        for j in range(len(global_para)):
            dist.send(global_para[j].to('cpu'), dst=i)

    print("Epoch 0: ", end='')
    test(args, model, device, test_loader)
    for epoch in range(1, args.epochs + 1):
        print("Epoch %d: "%epoch, end='')
        local_para = [para[1].data for para in model.named_parameters()]
        local_paras = []
        
        for i in range(1, world_size+1):
            for j in range(len(global_para)):
                temp = local_para[j].to('cpu')
                dist.recv(temp, src=i)
                local_para[j]=temp.to('cuda')
                del temp
            local_paras.append(local_para)
        global_para = aggregate_nomalization(global_para, local_paras)
        apply_global_para(model, global_para)
        test(args, model, device, test_loader)
        del local_paras
        for i in range(1, world_size+1):
            for j in range(len(global_para)):
                dist.send(global_para[j].to('cpu'), dst=i)
        plt.close('all')


        # plt.figure(1)
        
        # plt.ion()
        # plt.plot(test_loss_plot)
        # plt.title('test loss')
        # plt.xlabel('epochs')
        # plt.ylabel('test loss')
        # plt.show()
    
    plt.ioff()
    plt.figure()
    plt.plot(test_loss_plot)
    plt.title('Test loss')
    plt.xlabel('epochs')
    plt.ylabel('test loss')
    plt.savefig('./test_loss.png')
    plt.figure()
    plt.plot(test_acc_plot)
    plt.title('Test accuracy')
    plt.xlabel('epochs')
    plt.ylabel('test accuracy')
    plt.savefig('./test_acc.png')
    plt.ioff()
    # plt.show()
    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")
main()


# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
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
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
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

#     test_loss /= len(test_loader.dataset)
#     test_loss_plot.append(test_loss)
#     test_acc_plot.append(correct / len(test_loader.dataset))

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

# def apply_global_para(model, global_para):
#     para_dict = model.state_dict()
#     keys=list(model.state_dict())
#     for i in range(len(keys)):
#         para_dict.update({keys[i]: global_para[i]})
#     model.load_state_dict(para_dict)
#     del para_dict