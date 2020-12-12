"""Visualization dashboard

source: https://plotly.com/dash/
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(epochs, loss):
    plt.plot(np.linspace(0,epochs,epochs), loss)
    plt.title("Loss function")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    pass
