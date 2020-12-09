"""Core nn script

"""
import torch
import torch.nn as nn
import time
from progress.bar import Bar
import matplotlib.pyplot as plt
import numpy as np

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

def fit(net, x_train, y_train, lr=0.05, epochs=1000):
    bar = Bar('Epochs', max=epochs)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss_data = []
    for i in range(epochs):
        y = net.forward(x_train)
        loss = loss_function(y, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bar.next()
        loss_data.append(loss.item())

    bar.finish()
    plt.plot(np.linspace(0,epochs,epochs), loss_data)
    plt.title("Loss function")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    x_train = torch.tensor([[1.0, 0.0]])
    y_train = torch.tensor([[1.0]])
    net = Model(1,2,1)

    fit(net, x_train, y_train)

    print(f"[TEST] 1 = {net.forward(x_train)}")

