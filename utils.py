import random
from random import sample, shuffle
import copy
import csv
from models.CNN import *
from models.MLP import *
from models.ResNet18 import *
from optimization import *


def get_update_seq(n_epoch, n_client, n_batch):
    train_batch_per_client = n_epoch * n_batch
    update_seq = [*range(1, n_client+1)] * train_batch_per_client
    shuffle(update_seq)
    return update_seq
	
	
def init_models(model_type, n_party, train_dataset_list, device, client_output_size=32, server_embedding_size=128):
    n_client = n_party - 1
    models = []
    if model_type == "LinearResNet18":
        for rank in range(n_party):
            if rank == 0:
                models.append(LinearResNetServer(n_client).to(device)) 
            else:
                models.append(FullResNetClient().to(device))
    if model_type == "SimpleResNet18":
        for rank in range(n_party):
            if rank == 0:
                models.append(SimpleResNetServer(n_client).to(device)) 
            else:
                models.append(FullResNetClient().to(device))
    elif model_type == "CNN":
        for rank in range(n_party):
            if rank == 0:
                models.append(CNNServer(n_client).to(device)) 
            else:
                models.append(CNNClient(n_client).to(device))
    elif model_type == "MLP":
        for rank in range(n_party):
            if rank == 0:
                server_model = Server_MNIST_Net(n_client=n_client, input_size=client_output_size, embedding_size=server_embedding_size ).to(device)
                models.append(server_model)
            else:
                input_size = train_dataset_list[rank][0][0].shape[0]
                client_model = Client_MNIST_Net(input_size=input_size, output_size=client_output_size).to(device)
                models.append(client_model)
    else:
        raise Exception("Not chosen model.")
    return models



def init_optimizers(models, server_lr, client_lr):
    optimizers = []
    for rank in range(len(models)):
        if rank == 0:
            optimizers.append(get_optimizer(models[rank], server_lr))
        else:
            optimizers.append(get_optimizer(models[rank], client_lr))
    return optimizers
	
	
	
def init_log_file(field, log_file_name):
    if log_file_name == "None":
        return
    with open(f"{log_file_name}.csv", 'x') as f:
        write = csv.writer(f)
        write.writerow(field)

def append_log(row, log_file_name):
    with open(f"{log_file_name}.csv", 'a') as f:
        write = csv.writer(f)
        write.writerow(row)