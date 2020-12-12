"""Core nn script

"""
import torch
import torch.nn as nn
import time
from progress.bar import Bar
import visualization as vi
from data import (generate_data_from_model, CustomDataset, verification,
                    generate_dataset)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaseNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model_name = str(self)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name:str):
        if name is None:
            prefix = self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        else:
            prefix = name + '_'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name


class Model(BaseNN):
    """
    Basic test-model, to be sure that everything is works :)
    """
    def __init__(self, batch_size, input_dim, output_dim):
        super().__init__()
        N1 = 2
        self.layers = []
        self.layer1 = nn.Linear(input_dim, N1, bias=False)
        self.layer2 = nn.Linear(N1, output_dim, bias=False)

        self.layers.append(self.layer1)
        self.layers.append(self.layer2)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

    def print_info(self):
        string = ''
        for layer in self.layers:
            string += str(layer.weight) + '\n'
        return string



class FDModel(BaseNN):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.layer2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x


if __name__ == "__main__":
    net = FDModel(1,1,4).to(DEVICE)
    t, u, y = generate_data_from_model()
    dataset = CustomDataset(u, y)
    dataset = generate_dataset(dataset, batch_size=1)
    print(dataset.next())

    fit(net, dataset, batch_size=1)
