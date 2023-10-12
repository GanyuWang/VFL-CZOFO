from torch import nn
import torch.nn.functional as F

# Flatten in the model. 
class Client_MNIST_Net(nn.Module):
    def __init__(self, input_size, embedding_size=128, output_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # x = torch.sigmoid(x)
        return x

class Server_MNIST_Net(nn.Module):
    def __init__(self, n_client, input_size=128, embedding_size=128, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size*n_client, embedding_size)
        self.fc2 = nn.Linear(embedding_size, output_size)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output