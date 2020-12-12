"""
Data and Dataset generation
"""
import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch


def u_sin(t):
    return np.sin(t)

def u_step(t):
    return 0 if t < 20 else 1

def model(x, t, b, k, m):
    """
    ODE model: spring, mass damp.

    :param x: input x value
    :param t: time value
    :param b: damping
    :param k: spring stiffness
    :param m: mass

    :return dx: derivative x
    """
    dx0 = x[1]
    dx1 = -k/m*x[0] - b/m*x[1] + u_step(t)/m
    dx = [dx0, dx1]

    return dx

def test_model():
    """
    Test model function

    if we pass k,b,m = 0 => no spring, damp, mass in system =>  achieve
    some fault.
    """
    n = 5001
    t = np.linspace(0, 50, n)

    x0 = [0.1, 0]
    b = 1
    k = 100
    m = 1

    y = odeint(model, x0, t, args=(b, k, m))
    plt.plot(t, y)
    plt.show()

class CustomDataset(Dataset):
    """
    Create a dataset from data

    After we can load it with DataLoader:
    >dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True,
    num_workers=2)
    *num_workers - multiprocessing


    Can be part of training loop:
        >dataiter = iter(dataloader)
        >data = dataiter.next()
        >features, labels = data
        >print(features, labels)

    """
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def generate_dataset(dataset, batch_size=None, shuffle=False, num_workers=1):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers)
    return iter(dataloader)

def generate_data_from_model():
    """
    Dataset generator

    """
    n = 5001
    t = np.linspace(0, 50, n)

    x0 = [0, 0]
    b = 1
    k = 100
    m = 1

    y = odeint(model, x0, t, args=(b, k, m))
    u = np.asarray(list(map(u_step, t)), dtype="float")    # control signal to model
    y = y[:,0]

    return t, u, y

def verification(net, u):
    y_hat = torch.zeros([len(u)])
    for i in range(len(u)):
        x = u[i]
        y_hat[i] = net.forward(x)

    return y_hat

if __name__ == "__main__":
    t, u, y = generate_data_from_model()
    dataset = CustomDataset(u,y)
    print(f"Dataset len: {len(dataset)}")
    dataset = generate_dataset(dataset)
    for _ in range(5):
        print(dataset.next())
