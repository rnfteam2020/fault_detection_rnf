import data
import os
import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt


def fft(x, fs):
    """
    FFT function

    :param x: input data for fft
    :param fs: sample frequency [Hz]
    :return: frequency [Hz], magnitude
    """

    y = np.abs(rfft(x))
    N = len(y)
    f = rfftfreq(N, 1/fs)
    return f, y


def plot_fft(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Frequency (Hz)', ylabel='Amplitude (-)', title='FFT')
    ax.legend(loc=1)
    ax.grid()
    plt.show()


def get_fs(data, axis, duration):
    """
    returns sampling rate of input data

    :param data: input data sets (list)
    :param axis: index of desired axes {0, 1, 2, ...}
    :param duration: duration of measurement [s]
    :return: sampling rate [Hz]
    """
    fs = len(data[axis])/duration
    return fs


def generate_training_dataset():
    pass


if __name__ == "__main__":
    path = os.getcwd()
    data = data.load(path+'/data/data_RAE.csv')
    print(data)

