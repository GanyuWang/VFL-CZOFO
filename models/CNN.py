import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClient(nn.Module):
    def __init__(self, n_client, client_output_size=128):
        super(CNNClient, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pooling = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(64, 16, 3, padding=1)
        self.linear1 = nn.Linear(800, 256)
        self.linear2 = nn.Linear(256, client_output_size)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        return x
        
class CNNServer(nn.Module):
    def __init__(self, n_client, client_output_size=128, num_classes=10):
        super(CNNServer, self).__init__()
        self.linear1 = nn.Linear(n_client*client_output_size, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        return x 
