import data
import os
from scipy.fft import rfft, rfftfreq
from matplotlib.pyplot import plt


def fft(x, N, fs):
    y = np.abs(rfft(value))
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

if __name__ == "__main__":
    path = os.getcwd()
    data = data.load(path+'/data/data_RAE.csv')
    print(data)

