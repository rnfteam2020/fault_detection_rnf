from core.data import generate_signals_with_labels
import os
import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import statistics
from math import sqrt
from scipy.signal import find_peaks

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
    rms = sqrt(ms)
    return rms


def function(x)
    x_min = np.amin(x)

    x_max = np.amax(x)

    x_mean = np.mean(x)

    x_median = np.median(x)

    x_stdev = np.std(x)

    x_variance = np.square(x_stdev)

    x_rms = rms(pos)

    x_f, x_mag = fft(data, get_fs([pos, vel], 0, t_max))

    x_f_peaks_idx, _ = find_peaks(x_mag, height=0)

    x_f_peaks_idx_sorted = np.argsort(x_f_peaks_idx)

    x_f_peaks_idx_max3 = x_f_peaks_idx_sorted[-3:]

    x_max_freqs = x_f[x_f_peaks_idx_max3]
    return np.asarray([])


def generate_statistic_features(t_max):
    # TODO
    """
    Generate statistic features from signals
    :param t_max: duration of simulation
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

    for d in data:
        for label, signal in d.items():

            pos = signal[2][0]     # position
            vel = signal[2][1]     # velocity

            x_min = np.amin(pos)
            y_min = np.amin(vel)

            x_max = np.amax(pos)
            y_max = np.amax(vel)

            x_mean = np.mean(pos)
            x_mean = np.mean(vel)

            x_median = np.median(pos)
            x_median = np.median(vel)

            x_stdev = np.std(pos)
            y_stdev = np.std(vel)

            x_variance = np.square(x_stdev)
            y_variance = np.square(y_stdev)

            x_rms = rms(pos)
            y_rms = rms(vel)

            x_f, x_mag = fft(data, get_fs([pos, vel], 0, t_max))
            y_f, y_mag = fft(data, get_fs([pos, vel], 1, t_max))

            x_f_peaks_idx, _ = find_peaks(x_mag, height=0)
            y_f_peaks_idx, _ = find_peaks(x_mag, height=0)

            x_f_peaks_idx_sorted = np.argsort(x_f_peaks_idx)
            y_f_peaks_idx_sorted = np.argsort(y_f_peaks_idx)

            x_f_peaks_idx_max3 = x_f_peaks_idx_sorted[-3:]
            y_f_peaks_idx_max3 = y_f_peaks_idx_sorted[-3:]

            x_max_freqs = x_f[x_f_peaks_idx_max3]
            y_max_freqs = y_f[y_f_peaks_idx_max3]

            x_train = {'Min': x_min }

            # x_train[i] = np.stack(function(pos),function(vel))
            # y_train[i] = np.asarray(label)


    return x_train, y_train



if __name__ == "__main__":
    path = os.getcwd()
    data = data.load(path+'/data/data_RAE.csv')
    print(data)

    f_s = get_fs(data, 'x', 8)
    print(f_s)
    f, y = fft(data['x'], f_s)
    plot_fft(f, y)

