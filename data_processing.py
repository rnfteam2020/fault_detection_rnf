import data
import os
from scipy.signal import rfft, rfftfreq
from matplotlib.pyplot import plt


def fft(x, N, fs):
    y = np.abs(rfft(value))
    f = rfftfreq(N, 1/SAMPLE_RATE)
    return f, y

def plot(x, y):
    fig, ax = plt.subplots()
    ax.plot(f, y)
    ax.set(xlabel='Frequency (Hz)', ylabel='Amplitude (-)', title='FFT')
    ax.legend(loc=1)
    ax.grid()
    plt.show()

if __name__ == "__main__":
    path = os.getcwd()
    data = data.load(path+'/data/data_RAE.csv')
    print(data)

