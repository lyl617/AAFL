__author__ = 'yang.xu'
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

from fl_utils import printer, time_since
import gc
import resource

def train(args, start, tx2_model, device, tx2_train_loader, tx2_test_loader, tx2_optimizer, epoch, fid):
    # param_server.clear_objects()

    vm_start = time.time()
    tx2_model.train()

    for li_idx in range(args.local_iters):

        for batch_idx, (vm_data, vm_target) in enumerate(tx2_train_loader, 1):
            # print(data.location)
            # print("vm data:", vm_data)

            if args.dataset_type == 'FashionMNIST' or args.dataset_type == 'MNIST':
                if args.model_type == 'LR':
                    vm_data = vm_data.squeeze(1) 
                    vm_data = vm_data.view(-1, 28 * 28)
                else:
                    # vm_data = vm_data.unsqueeze(1)  # <--for FashionMNIST
                    pass

            if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
                if args.model_type == 'LSTM':
                    vm_data = vm_data.permute(0, 2, 3, 1)
                    vm_data = vm_data.contiguous().view(-1, 32, 32 * 3)                    
                else:
                    # vm_data = vm_data.permute(0, 3, 1, 2) #<--for CIFAR10 & CIFAR100
                    pass
                
            vm_data, vm_target = vm_data.to(device), vm_target.to(device)
            # print('--[Debug] vm_data = ', vm_data.get())
            if args.model_type == 'LSTM':
                hidden = tx2_model.initHidden(args.batch_size)
                hidden = hidden.send(vm_data.location)
                for col_idx in range(32):
                    vm_data_col = vm_data[:, col_idx, :]
                    vm_output, hidden = tx2_model(vm_data_col, hidden)
            else:
                vm_output = tx2_model(vm_data)

            tx2_optimizer.zero_grad()
            
            vm_loss = F.nll_loss(vm_output, vm_target)
            # print(vm_output)
            vm_loss.backward()
            tx2_optimizer.step()
            # vm_data = vm_data.get()

            if batch_idx % args.log_interval == 0:
                vm_loss = vm_loss.item()  # <-- NEW: get the loss back
                #print("Epoch :{} batch_idx: {} print ok".format(epoch, batch_idx))
                # print(vm_loss)
                printer('-->[{}] Train Epoch: {} Local Iter: {} tx2: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time_since(vm_start), epoch, li_idx, args.rank, batch_idx * args.batch_size, 
                    len(tx2_train_loader) * args.batch_size,
                    100. * batch_idx / len(tx2_train_loader), vm_loss), fid)

            # vm_data = vm_data.get()
            # vm_target = vm_target.get()
            # vm_output = vm_output.get()

            if args.model_type == 'LSTM':
                #hidden = hidden.get()
                del hidden

            # if not batch_idx % args.log_interval == 0:
            #     vm_loss = vm_loss.get()
            
            del vm_data
            del vm_target
            del vm_output
            del vm_loss
        # break

    # if epoch == args.epochs:
    if args.enable_vm_test:
        printer('-->[{}] Test set: Epoch: {} tx2: {}'.format(time_since(vm_start), epoch, args.rank), fid)
        # <--test for each vm
        test(args, start, tx2_model, device, tx2_test_loader, epoch, fid)

        # vm_models[vm_idx].move(param_server)
        # vm_models[vm_idx] = vm_models[vm_idx].get()
        # torch.cuda.empty_cache()
        gc.collect()

def test(args, start, model, device, test_loader, epoch, fid):
    model.eval()

    test_loss = 0.0
    test_accuracy = 0.0

    correct = 0

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            if args.dataset_type == 'FashionMNIST' or args.dataset_type == 'MNIST':
                if args.model_type == 'LR':
                    data = data.squeeze(1) 
                    data = data.view(-1, 28 * 28)
                else:
                    # vm_data = vm_data.unsqueeze(1)  # <--for FashionMNIST
                    pass

            if args.dataset_type == 'CIFAR10' or args.dataset_type == 'CIFAR100':
                if args.model_type == 'LSTM':
                    data = data.view(-1, 32, 32 * 3)                    
                else:
                    # vm_data = vm_data.permute(0, 3, 1, 2) #<--for CIFAR10 & CIFAR100
                    pass  

            if args.model_type == 'LSTM':
                hidden = model.initHidden(args.test_batch_size)
                hidden = hidden.send(data.location)
                for col_idx in range(32):
                    data_col = data[:, col_idx, :]
                    output, hidden = model(data_col, hidden)
            else:
                output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()

            correct += batch_correct
            #print('--[Debug][in Test set] batch correct:', batch_correct)

            # if not args.enable_vm_test:
            #     printer('--[Debug][in Test set] batch correct: {}'.format(batch_correct), fid)
            
            # data =  data.get()
            # target = target.get()
            # output = output.get()
            # pred = pred.get()
            if args.model_type == 'LSTM':
                #hidden = hidden.get()
                del hidden
                
            del data
            del target
            del output
            del pred
            del batch_correct

    test_loss /= len(test_loader.dataset)
    test_accuracy = np.float(1.0 * correct / len(test_loader.dataset))

    if args.enable_vm_test:  
        printer('-->[{}] Test set: Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            time_since(start), epoch, test_loss, correct, len(test_loader.dataset),
            100. * test_accuracy), fid)
    else:
        printer('[{}] Test set: Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            time_since(start), epoch, test_loss, correct, len(test_loader.dataset),
            100. * test_accuracy), fid)

    gc.collect()

    return test_loss, test_accuracy

