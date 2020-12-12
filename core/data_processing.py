from core.data import generate_signals_with_labels
import os
import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import statistics
import math

def fft(x, fs):
    """
    FFT function

    :param x: input data for fft
    :param fs: sample frequency [Hz]
    :return: frequency [Hz], magnitude
    """

    N = len(x)
    y = np.abs(rfft(x))
    f = rfftfreq(N, 1/fs)
    return f, y


def plot_fft(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Frequency (Hz)', ylabel='Amplitude (-)', title='FFT')
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
    f_s = len(data[axis])//duration
    return f_s


def rms(x):
    """
    calc RMS of array

    :param x: input array
    :return: RMS (root mean square)
    """
    ms = 0
    for i in range(len(x)):
        ms = ms + x[i]^2
    ms = ms/len(x)
    rms = math.sqrt(ms)
    return rms


def generate_statistic_features():
    # TODO
    """
    Generate statistic features from signals
    :return x_train, y_train: features(statistic data in np.array([])),
                                labels(np.array([1/0])
    """
    # Generated data from model
    # data is in dictionary format
    # data = {'label': 0/1, 'signals': np.array([t,u,y])}
    data = generate_signals_with_labels()
    """
    for d in data:
        for label, signal in d.items():
            print(f'{label} : {signal}')
    """
    # data = {'label': 0/1, 'signals': np.array([t,u,[x, y]]])}

    pos = data['signals'][2][0]     # position
    vel = data['signals'][2][0]     # velocity

    x_min = np.amin(pos)
    y_min = np.amin(vel)

    x_max = np.amax(pos)
    y_max = np.amax(vel)

    x_mean = np.mean(pos)
    x_mean = np.mean(vel)

    x_median = np.median(pos)
    x_median = np.median(vel)

    x_stdev = np.std(pos)
    x_stdev = np.std(vel)

    x_min = rms(pos)
    y_min = rms(vel)


    return x_train, y_train



if __name__ == "__main__":
    path = os.getcwd()
    data = data.load(path+'/data/data_RAE.csv')
    print(data)

    f_s = get_fs(data, 'x', 8)
    print(f_s)
    f, y = fft(data['x'], f_s)
    plot_fft(f, y)

