from core.data import generate_signals_with_labels
import os
import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from math import sqrt
from scipy.signal import find_peaks


def fft(x, fs):
    """
    FFT function

    :param x: input data for fft
    :param fs: sample frequency [Hz]
    :return: frequency [Hz], magnitude
    """

    n = len(x)
    y = np.abs(rfft(x))
    f = rfftfreq(n, 1/fs)
    return f, y


def plot_fft(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Frequency (Hz)', ylabel='Amplitude (-)', title='FFT')
    ax.grid()
    plt.show()


def get_fs(x, duration):
    """
    returns sampling rate of input data

    :param x: input data set (list)
    :param duration: duration of measurement [s]
    :return: sampling rate [Hz]
    """
    f_s = len(x)//duration
    return f_s


def rms(x):
    """
    calc RMS of array

    :param x: input array
    :return: RMS (root mean square)
    """
    ms = 0
    for i in range(len(x)):
        ms = ms + x[i] ^ 2
    ms = ms/len(x)
    rms = sqrt(ms)
    return rms


def calc_statistics(pos, vel, t_max):
    """
    calculates statistics from two input signals
    :param pos: first signal - position
    :param vel: second signal - velocity
    :param t_max: simulation duration (from data generation)
    :return: stat_data_pos, stat_data_vel: stacked statical data for position and velocity
             [min, max, mean, median, stdev, variance, rms, max_freqs]
    """
    xy_min = np.stack(np.amin(pos), np.amin(vel))

    xy_max = np.stack(np.amax(pos), np.amax(vel))

    xy_mean = np.stack(np.mean(pos), np.mean(vel))

    xy_median = np.stack(np.median(pos), np.median(vel))

    xy_stdev = np.stack(np.std(pos), np.std(vel))

    xy_variance = np.stack(np.square(xy_stdev[0]), np.square(xy_stdev[1]))

    xy_rms = np.stack(rms(pos), rms(vel))

    xy_f_mag = np.stack(fft(pos, get_fs(pos, t_max)), fft(vel, get_fs(vel, t_max)))

    xy_f_peaks_idx, _ = np.stack(find_peaks(xy_f_mag[0], height=0), find_peaks(xy_f_mag[1], height=0))

    xy_f_peaks_idx_sorted = np.stack(np.argsort(xy_f_peaks_idx[0]), np.argsort(xy_f_peaks_idx[1]))

    xy_f_peaks_idx_max3 = np.stack(xy_f_peaks_idx_sorted[0][-3:], xy_f_peaks_idx_sorted[1][-3:])

    xy_max_freqs = np.stack(xy_f_mag[0][xy_f_peaks_idx_max3[0]], xy_f_mag[1][xy_f_peaks_idx_max3[1]])

    stat_data_pos = np.stack(xy_min[0], xy_max[0], xy_mean[0], xy_median[0], xy_stdev[0], xy_variance[0], xy_rms[0],
                             xy_max_freqs[0])
    stat_data_vel = np.stack(xy_min[1], xy_max[1], xy_mean[1], xy_median[1], xy_stdev[1], xy_variance[1], xy_rms[1],
                             xy_max_freqs[1])

    return stat_data_pos, stat_data_vel


def generate_statistic_features(t_max):
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

    # stat_data_pos, stat_data_vel = np.empty(len(data))
    x_train, y_train = np.empty(len(data))
    for d in data:

        for label, signal in d.items():

            pos = signal[2][0]     # position
            vel = signal[2][1]     # velocity

            stat_data_pos, stat_data_vel = calc_statistics(pos, vel, t_max)
            x_train[d] = np.stack(stat_data_pos, stat_data_vel)
            y_train[d] = np.asarray(label)

    return x_train, y_train


if __name__ == "__main__":
    path = os.getcwd()
    # data = data.load(path+'/data/data_RAE.csv')
    # print(data)
    #
    # f_s = get_fs(data, 8)
    # print(f_s)
    # f, y = fft(data['x'], f_s)
    # plot_fft(f, y)
