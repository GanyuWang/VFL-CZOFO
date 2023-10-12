import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

def get_dataset(dataset_name, ):
    if dataset_name == "CIFAR10":
        print('==> Preparing data..CIFAR')
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

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        
    elif dataset_name == "MNIST":
        trainset = datasets.MNIST('data', download=True, train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ]))
        testset  = datasets.MNIST('data', download=True, train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ]))
    else:
        raise Exception("Not chosen dataset.")
    return trainset, testset
	
	

def get_targets_from_index(index, server_dataset):
    targets = torch.zeros_like(index)
    for i, idx in enumerate(index):
        targets[i] = server_dataset[idx][1]
    return targets

# special
def get_item_from_index(index, dataset):
    bs = len(index)

    data_shape = dataset[0][0].shape

    inputs = torch.zeros(bs, *data_shape)
    labels = torch.zeros(bs)
    indics = torch.zeros(bs)
    
    for i, idx in enumerate(index):
        inputs[i] = dataset[idx][0]
        labels[i] = dataset[idx][1]
        indics[i] = dataset[idx][2]
    return inputs, labels, indics
	
# only works for 2 clients. 
# 1 server. 
# server 0. client 1, client 2. 
class Vertical_Partition_Dataset(Dataset):
    def __init__(self, dataset_name, dataset, rank, n_party):
        self.dataset = dataset
        self.rank = rank # note that client index is 1... n_parties - 1. 
        self.n_party = n_party
        self.n_client = n_party - 1
        # Allocate feature for clients with rank "rank". 
        self.dataset_name = dataset_name
        if dataset_name == "MNIST":
            # Allocate feature for clients with rank "rank". 
            self.sample_size = torch.numel(self.dataset[0][0])
            if self.rank == 0: # server
                self.start_feature = self.end_feature = self.feature_sizes = 0
            else: # clients: get equal number of attributes (except the last one)
                self.feature_sizes = self.sample_size // self.n_client
                self.start_feature_index = self.feature_sizes * (self.rank-1)
                # End feature index, deal with the last client. 
                self.end_feature_index = self.feature_sizes * (self.rank)
                if self.rank == n_party - 1: 
                    self.end_feature_index = self.sample_size
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # return the partition dataset. 
        data, label = self.dataset[idx]

        if self.dataset_name == "MNIST":
            if self.rank == 0:
                flatten_sample = torch.flatten(data)
                return flatten_sample, label, idx
            # client, return the partition feature. 
            else:
                flatten_sample = torch.flatten(data)
                return flatten_sample[self.start_feature_index: self.end_feature_index], label, idx
            
        elif self.dataset_name == "CIFAR10":
            if self.n_client == 2:
                if self.rank == 0:
                    return data, label, idx
                else:
                    d = data[:, :, (self.rank-1)*16 : (self.rank)*16]
                    d1 = F.pad( d, (8, 8), "constant")
                    return d1, label, idx
            elif self.n_client == 4:
                if self.rank == 0:
                    return data, label, idx
                else:
                    x = (self.rank-1) // 2
                    y = (self.rank-1) % 2
                    d = data[:, x*16: (x+1)*16, (y)*16 : (y+1)*16]
                    d1 = F.pad( d, (8, 8, 8, 8), "constant")
                    return d1, label, idx
            elif self.n_client == 1:
                return data, label, idx
        else:
            raise Exception("Not chosen dataset.")
			
			
# make data loader, (do not shuffle each time, or send index. )
def make_data_loader(dataset, batch_size):
    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    return data_loader
    
def partition_dataset(dataset_name, dataset, n_party, batch_size):
    dataset_list = []
    loader_list = []
    for i in range(n_party):
        dataset_list.append( Vertical_Partition_Dataset(dataset_name, dataset, i, n_party) )
        loader_list.append(make_data_loader(dataset_list[i], batch_size))
    return dataset_list, loader_list

def make_iter_loader_list(loader_list, n_party):
    iter_loader_list = []
    for m in range(n_party):
        iter_loader_list.append(iter(loader_list[m]))
    return iter_loader_list