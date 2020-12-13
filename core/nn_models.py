"""Core nn script

"""
import torch
import torch.nn as nn
import time
import core.visualization as vi

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
    pass

