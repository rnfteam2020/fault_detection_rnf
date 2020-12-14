"""Visualization dashboard

source: https://plotly.com/dash/
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(x, y, title, x_lab, y_lab):
    plt.plot(np.linspace(0,x,x), y)
    plt.title(title)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.show()

def plot_data(x, y, title, x_lab, y_lab):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.show()


if __name__ == "__main__":
    pass
