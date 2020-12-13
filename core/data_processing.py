from core.data import generate_signals_with_labels
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
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
        ms = ms + x[i]**2
    ms = ms/len(x)
    rms = np.sqrt(ms)
    return rms


def calc_statistics(x, t_max):
    """
    calculates statistics from two input signals
    :param pos: first signal - position
    :param vel: second signal - velocity
    :param t_max: simulation duration (from data generation)
    :return: stat_data_pos, stat_data_vel: stacked statical data for position and velocity
             [min, max, mean, median, stdev, variance, rms, max_freqs]
    """
    data = [None] * 10

    x_min = np.amin(x)
    x_max = np.amax(x)
    x_mean = np.mean(x)
    x_median = np.median(x)
    x_stdev = np.std(x)
    x_variance = np.square(x_stdev)
    x_rms = rms(x)

    x_f_mag, f = fft(x, get_fs(x, t_max))
    x_f_peaks_idx, _ = find_peaks(x_f_mag, height=0)
    x_f_peaks_idx_sorted = np.argsort(x_f_peaks_idx)
    x_f_peaks_idx_max3 = x_f_peaks_idx_sorted[-3:]

    x_max_freqs = x_f_mag[x_f_peaks_idx_max3]

    stat_data = [x_min, x_max, x_mean, x_median, x_stdev,
        x_variance, x_rms]
    for i in range(len(stat_data)):
        data[i] = stat_data[i]

    for i in range(len(x_max_freqs)):
        data[i+len(stat_data)] = x_max_freqs[i]

    data = [d if d is not None else 0 for d in data]

    return np.asarray(data)


def generate_statistic_features():
    """
    Generate statistic features from signals
    :return x_train, y_train: features(statistic data in np.array([])),
                                labels(np.array([1/0])
    """
    # Generated data from model
    # data is in dictionary format
    # data = {'label': 0/1, 'signals': np.array([t,u,y])}
    data, t_max, _ = generate_signals_with_labels()

    # stat_data_pos, stat_data_vel = np.empty(len(data))
    # ?????
    # x_train, y_train = np.empty(len(data))

    x_train = np.zeros((100,20))
    y_train = np.zeros((100,1))

    for i, d in enumerate(data):
        for label, signal in d.items():

            pos = signal[2][:,0]     # position
            vel = signal[2][:,1]     # velocity

            x_train[i] = np.concatenate((calc_statistics(pos, t_max), calc_statistics(vel, t_max)))
            y_train[i] = np.array(label)

    return x_train, y_train

def generate_verification_data():
    # TODO
    data, t_max, _ = generate_signals_with_labels_verification()

    return x_verif, y_verif


if __name__ == "__main__":
    x_train, y_train = generate_statistic_features()
    print(x_train, y_train)

