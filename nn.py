"""Core nn script

"""
import torch
import torch.nn as nn
import time
from progress.bar import Bar

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
    def __init__(self, batch_size, input_dim, output_dim):
        super().__init__()
        N1 = 10
        self.layer1 = nn.Linear(input_dim, N1, bias=False)
        self.layer2 = nn.Linear(N1, output_dim, bias=False)
        self.layer_output = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.layer_output(x)
        return x


def fit(x_train, y_train, lr=0.001, epochs=1000, batch_size=32):
    bar = Bar('Epochs', max=epochs)
    model = Model(1,2,3)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(epochs):
        bar.next()

    bar.finish()

if __name__ == "__main__":
    fit(1,2)

