import data
import os
from scipy.fft import fft, fftfreq






if __name__ == "__main__":
    path = os.getcwd()
    data = data.load(path+'/data/data_RAE.csv')
    print(data)

def do_fft(sample_rate):
    return

