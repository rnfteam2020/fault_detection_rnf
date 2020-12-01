import data
import os
import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt


def fft(x, fs):
    """
    FFT function

    Args:
        x: input data for fft
        fs: sample frequency

    Returns:
        f: frequency
        y: magnitude

    Raises:
        None
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

def get_sampling_rate(data, axis, measurement_duration):
    samp_rate = len(data[axis])/measurement_duration
    return samp_rate

if __name__ == "__main__":
    path = os.getcwd()
    data = data.load(path+'/data/data_RAE.csv')
    print(data)

