import torch
import torch.nn as nn
from dataset import * #get_dataset, partition_dataset, make_iter_loader_list, get_item_from_index, get_targets_from_index
from optimization import *#ZO_output_optim
from compressor import * #Scale_Compressor
from utils import * #init_models, init_optimizers, init_log_file, get_update_seq

from torch.autograd import Variable


import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, required=False, default=12341)
parser.add_argument('--framework_type', type=str, required=False, default="ZOFO", help='"ZOFO", "ZO", "FO"')
parser.add_argument('--dataset_name', type=str, required=False, default="MNIST", help='"CIFAR10" "MNIST"')
parser.add_argument('--model_type', type=str, required=False, default="MLP", help='MLP, CNN, SimpleResNet18')
parser.add_argument('--n_party', type=int, required=False, default=3)
parser.add_argument('--client_output_size', type=int, required=False, default=64, help='MLP: 64. CNN 128. SimpleResNet18 10.')
parser.add_argument('--server_embedding_size', type=int, required=False, default=128, help="for MLP experiment only. ")
parser.add_argument('--client_lr', type=float, required=False, default=0.02)
parser.add_argument('--server_lr', type=float, required=False, default=0.02)
parser.add_argument('--batch_size', type=int, required=False, default=64, help="MNIST:64, CIFAR10:128")
parser.add_argument('--n_epoch', type=int, required=False, default=100, help="MNIST:100, CIFAR10:50")
parser.add_argument('--u_type', type=str, required=False, default="Uniform", help="Uniform, Normal, Coordinate")
parser.add_argument('--mu', type=float, required=False, default=0.001)
parser.add_argument('--d', type=float, required=False, default=1)
parser.add_argument('--sample_times', type=int, required=False, default=1, help="q")
parser.add_argument('--compression_type', type=str, required=False, default="None", help="None, Scale") ## remember to change to None. 
parser.add_argument('--compression_bit', type=int, required=False, default=8)
parser.add_argument('--response_compression_type', type=str, required=False, default="None", help="None, Scale") ## remember to change to None. 
parser.add_argument('--response_compression_bit', type=int, required=False, default=8)
parser.add_argument('--local_update_times', type=int, required=False, default=1, help="extra content")
parser.add_argument('--log_file_name', type=str, required=False, default="None")
args = parser.parse_args()

random_seed = args.random_seed

# framework
framework_type = args.framework_type # "ZOFO", "ZO", "FO"
dataset_name =  args.dataset_name # "CIFAR10" "MNIST"
# model
model_type =  args.model_type # MLP, CNN, SimpleResNet18, LinearResNet18
# model
n_party = args.n_party
n_client = n_party - 1
client_output_size = args.client_output_size #MLP: 64. CNN 128. SimpleResNet18 10.
server_embedding_size = args.server_embedding_size
# Training
server_lr = args.server_lr
client_lr = args.client_lr
batch_size = args.batch_size
n_epoch = args.n_epoch
u_type = args.u_type
mu = args.mu
d = args.d
sample_times = args.sample_times
# Special
compression_type = args.compression_type
compression_bit = args.compression_bit
response_compression_type = args.response_compression_type
response_compression_bit = args.response_compression_bit
# depreciate
local_update_times = args.local_update_times
# Log
log_file_name = args.log_file_name



# 
random.seed(random_seed)
torch.manual_seed(random_seed)
# 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# make dataset for server 0 and clients 
trainset, testset = get_dataset(dataset_name)
train_dataset_list, train_loader_list = partition_dataset(dataset_name, trainset, n_party, batch_size)
train_iter_loader_list = make_iter_loader_list(train_loader_list, n_party)
n_training_batch_per_epoch_client = len(train_loader_list[0])

# loss function
loss_fn = nn.CrossEntropyLoss()
loss_fn_coordinate = nn.CrossEntropyLoss(reduction="none")
# Zeroth order optimization. 
ZOO = ZO_output_optim(mu, u_type, client_output_size)
# Compression. 
compressor = Scale_Compressor(bit=compression_bit)
response_compressor = Scale_Compressor(bit=response_compression_bit)
# Build the model list. 
models = init_models(model_type, n_party, train_dataset_list, device, client_output_size, server_embedding_size)
# optimizers list. 
optimizers = init_optimizers(models, server_lr, client_lr)


# Record training d
field = ["epoch", "comm_round", "training_loss", "train_acc", "Bit"]
init_log_file(field, log_file_name)


# generate the stimulate update sequence for clients. 
update_seq = get_update_seq(n_epoch, n_client, n_training_batch_per_epoch_client)

# train with ZO on client and FO on server. 
epoch = 0
batch = 0
running_loss = 0.0
correct = 0
raw_communication_size = 0

# each iteration is one round. 
for m in update_seq:
    other_m_list = list(range(1, n_party))
    other_m_list.remove(m)
    
    ### Forward pass client 
    ## Get data. 
    # deal with client m. 
    try:
        inputs, labels, index = next(train_iter_loader_list[m])
    except StopIteration:
        # re initialize the loader 
        train_iter_loader_list[m] = iter(train_loader_list[m])
        inputs, labels, index = next(train_iter_loader_list[m])
    inputs = inputs.to(device)
    labels = labels.to(device)

    ### client side. forward
    # client output list. 
    inputs_list=[None] * n_party
    outputs_list=[None] * n_party
    compressed_list = [None] * n_party
    low_list = [None] * n_party
    high_list = [None] * n_party
    # deal with the other clients. Actually this should be a table keep in the server. 
    for all_m in range(1, n_party):
        inputs_list[all_m], NA, NA = get_item_from_index(index, train_dataset_list[all_m]) # index is just for check.
        inputs_list[all_m] = inputs_list[all_m].to(device)
        outputs_list[all_m] = models[all_m](inputs_list[all_m]).detach()
        outputs_list[all_m] = compress_decompress(outputs_list[all_m], compression_type, compressor)

    embedding_comm_cost_in_bit = compression_cost_in_bit(outputs_list[m], compression_type, compressor) # 注意这个是解码后矩阵。

    #raise Exception(embedding_comm_cost_in_bit )

    if framework_type == "ZOFO":
        output_client = torch.cat(outputs_list[1:], dim=-1)
        raw_communication_size += embedding_comm_cost_in_bit
        ### server side. 
        ## forward. initialize the server target based on the smaller index. 
        server_target = get_targets_from_index(torch.tensor(index), train_dataset_list[0]).to(device)
        ## Backward: Server calculate loss and update with gradient. 
        output_server = models[0](output_client)
        loss = loss_fn(output_server, server_target)
        try:
            optimizers[0].zero_grad()
            loss.backward(inputs=list(models[0].parameters()))
            optimizers[0].step()
        except :
            pass
        partial = torch.zeros(inputs.shape[0], client_output_size).to(device)
        if u_type == "Normal" or u_type == "Uniform":
            # generate 
            u_list = [None] * sample_times
            delta_list = [None] * sample_times
            for i in range(sample_times):
                u_list[i], perturbed_output_m, output_m = ZOO.forward(outputs_list[m])
                outputs_list_perturb = outputs_list.copy()
                outputs_list_perturb[m] = perturbed_output_m
                perturbed_output_client = torch.cat(outputs_list_perturb[1:], dim=-1)
                # server calculate perturbed loss. 
                perturbed_output_server = models[0](perturbed_output_client)
                perturbed_loss = loss_fn(perturbed_output_server, server_target)
                ### client side. 
                # Client m, update with chain rule. 
                delta_list[i] = perturbed_loss - loss # delta 是一个数。
            # the delta passed through network. 
            deltas = torch.tensor(delta_list).to(device)
            deltas_with_error = compress_decompress(deltas, response_compression_type, response_compressor)
            response_message_size = compression_cost_in_bit(deltas, response_compression_type, response_compressor)
            # client calculate the partial_tmp with the delta sent from the server. 
            for i in range(sample_times):
                partial_tmp = ZOO.backward(u_list[i], deltas_with_error[i])
                partial += partial_tmp
            partial = partial / sample_times
            raw_communication_size += response_message_size

        elif u_type == "Coordinate":
            u_list = [None] * client_output_size
            delta_list = [None] * client_output_size
            for l in range(client_output_size):
                u_list[l], perturbed_output_m_plus, perturbed_output_m_minus, output = ZOO.forward(outputs_list[m], l)
                # get the perturbed loss for plus 
                outputs_list_perturb_plus = outputs_list.copy()
                outputs_list_perturb_plus[m] = perturbed_output_m_plus
                perturbed_output_server_plus = models[0](torch.cat(outputs_list_perturb_plus[1:], dim=-1))
                perturbed_loss_plus = loss_fn_coordinate(perturbed_output_server_plus, server_target)
                # and minus. 
                outputs_list_perturb_minus = outputs_list.copy()
                outputs_list_perturb_minus[m] = perturbed_output_m_minus
                perturbed_output_server_minus = models[0](torch.cat(outputs_list_perturb_minus[1:], dim=-1))
                perturbed_loss_minus = loss_fn_coordinate(perturbed_output_server_minus, server_target)
                # get the partial. 
                delta_list[l] = perturbed_loss_plus - perturbed_loss_minus  # delta  torch.Size([batch_size])

            deltas = torch.stack(delta_list, dim=1).detach()
            deltas_with_error = compress_decompress(deltas, response_compression_type, response_compressor)
            response_message_size = compression_cost_in_bit(deltas, response_compression_type, response_compressor)

            for l in range(client_output_size):
                # client calcualte the partial with. 
                partial_tmp = ZOO.backward(u_list[l], deltas_with_error[:, l])
                #raise Exception("partial tmp. ", partial_tmp)
                # partial tmp.shape is [batch_size, client otuptu size ]
                partial += partial_tmp
            partial = partial/batch_size # to assimilar the FO gradient. (mean on the )
            raw_communication_size += response_message_size
            # Calculate the communication cost from 
            #raise Exception(embedding_comm_cost_in_bit, response_message_size, raw_communication_size)
        optimizers[m].zero_grad()
        output_local_m = models[m](inputs_list[m] )
        output_local_m.backward(gradient=partial, inputs=list(models[m].parameters()))
        optimizers[m].step()

    elif framework_type == "ZO":
        u, output_m, perturbed_output_m = ZOO.ZO_forward(inputs_list[m], models[m])
        output_client = torch.cat(outputs_list[1:], dim=-1)
        raw_communication_size += 2 * embedding_comm_cost_in_bit
        # cat the perturbed output from client. 
        outputs_list_perturb = outputs_list.copy()
        outputs_list_perturb[m] = perturbed_output_m
        perturbed_output_client = torch.cat(outputs_list_perturb[1:], dim=1)
        ### server side. 
        ## forward. initialize the server target based on the smaller index. 
        server_target = get_targets_from_index(torch.tensor(index), train_dataset_list[0]).to(device)
        ## Backward: Server calculate loss and update with gradient. 
        u_s, output_server_local, perturbed_output_server_local = ZOO.ZO_forward(output_client, models[0])
        loss = loss_fn(output_server_local, server_target)
        perturbed_loss_server_local = loss_fn(perturbed_output_server_local, server_target)
        ZOO.ZO_backward_step(perturbed_loss_server_local, loss, models[0], u_s, server_lr)
        # server calculate perturbed loss. 
        perturbed_output_server = models[0](perturbed_output_client)
        perturbed_loss = loss_fn(perturbed_output_server, server_target)

        ### client side. ZO backward. 
        # Client m, update with chain rule. 
        ZOO.ZO_backward_step(perturbed_loss, loss, models[m], u, client_lr)
        # Calculate the communication cost from 
        raw_communication_size += 2 * tensor_size_in_bit(perturbed_loss)  #loss and perturbed loss. 

    elif framework_type == "FO":
        # cat the output from client.  
        output_client = Variable(torch.cat(outputs_list[1:], dim=-1), requires_grad=True)
        raw_communication_size += embedding_comm_cost_in_bit
        ### server side. 
        ## forward. initialize the server target based on the smaller index. 
        server_target = get_targets_from_index(torch.tensor(index), train_dataset_list[0]).to(device)
        ## Backward: Server calculate loss and update with gradient. 
        output_server = models[0](output_client)
        optimizers[0].zero_grad()
        loss = loss_fn(output_server, server_target)
        try:
            loss.backward(inputs=list(models[0].parameters()).append(output_client))
        except:
            loss.backward(inputs=[output_client])
        optimizers[0].step()
        ### client side. 
        # Client m, update with chain rule. the partial for m. 
        partial = output_client.grad[:, (m-1)*client_output_size: m*client_output_size]
        
        # Calculate the communication cost from 
        raw_communication_size += tensor_size_in_bit(partial) 
        # Client update. 
        optimizers[m].zero_grad()
        output_local_m = models[m](inputs_list[m])
        output_local_m.backward(gradient=partial, inputs=list(models[m].parameters()))
        optimizers[m].step()
        #raise Exception(embedding_comm_cost_in_bit, tensor_size_in_bit(partial), raw_communication_size)
    else:
        raise Exception("No selected framework. ")
    
    ##### training acc and statics during training. #####
    # print statistics count batch.  
    batch += 1
    # predict the 1000 samples. 
    for all_m in range(1, n_party):
        inputs_m, NA, NA = get_item_from_index(index, train_dataset_list[all_m]) # index is just for check.
        inputs_m = inputs_m.to(device) 
        outputs_list[all_m] = models[all_m](inputs_m)
    output_client = torch.cat(outputs_list[1:], dim=-1)
    output_server = models[0](output_client)

    NA, predicted = torch.max(output_server.data, 1)
    correct += (predicted == labels).sum().item()
    running_loss += loss.item()
    # if server have updated n_client * n_batch. 
    if  batch % (n_client * n_training_batch_per_epoch_client) == 0:
        epoch += 1
        r_loss = running_loss/(len(trainset)*n_client)
        train_acc = correct/(len(trainset)*n_client)
        # add the row to the log file.  
        row = [epoch, batch, r_loss, train_acc, raw_communication_size]
        append_log(row, log_file_name)
        print(f'{epoch},{r_loss:.10f},{train_acc}, {raw_communication_size}')
        #
        correct = 0
        running_loss = 0.0

print('Finished Training')


# Testing

test_dataset_list, test_loader_list = partition_dataset(dataset_name, testset, n_party, batch_size)
# testloader = make_data_loader(testset, batch_size)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for NA, label, index in test_loader_list[0]:
        label = label.to(device)
        index = index.long()
        # calculate 
        m_list = list(range(1, n_party))
        outputs_list = [None] * n_party
        for m in m_list:
            input_m, NA, NA = get_item_from_index(index, test_dataset_list[m]) # get the corresponding inputs. 
            input_m = input_m.to(device) 
            outputs_list[m] = models[m](input_m)

        out_client = torch.cat(outputs_list[1:], dim=-1)
        outputs = models[0](out_client)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += label.shape[0]
        correct += (predicted == label).sum().item()

test_acc = correct / total
print(f'Accuracy of the network on the 10000 test images: {100 * test_acc} %')

append_log([0, 0, 0, test_acc, 0], log_file_name)

