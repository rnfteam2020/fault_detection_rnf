""" Import data

For debug purpose sine wave is ok :)

Possible import examples:
-----------------------------------------------------------
    import numpy as np
    import h5py
    f = h5py.File('somefile.mat','r')
    data = f.get('data/variable1')
    data = np.array(data) # For converting to a NumPy array
-----------------------------------------------------------
    import scipy.io
    mat = scipy.io.loadmat('file.mat')
    save('mat.mat', '-v7')
-----------------------------------------------------------


"""
import pandas as pd
import numpy as np
import scipy as sp
from scipy.integrate import odeint
import h5py
import matplotlib.pyplot as plt

def load(file_name:str, data_type='h5py'):
    """Data loading

    :param file_name:   path to file with data
    :param data_type:   dataset datatype (.mat, .csv, .txt, h5py, etc.)

    :return data:       loaded data
    """

    if data_type == 'csv':
        data = pd.read_csv(file_name, skipinitialspace=True)
    elif data_type == 'h5py':
        data = h5py.File(file_name, 'r')
    else:
        data = None

    return data

def h5py_parser(data_object):
    """
    Parser for h5py data objects

    :param data_object: input data in h5py format

    :return data_dict: dictionary representation of data
    """

    data_dict = {}
    for name, data in data_object.items():
        if type(data) is h5py.Dataset:
            value = data.value
            data_dict[str(name)] = value

    return data_dict

def u_sin(t):
    return np.sin(t)

def u_step(t):
    return 0 if t < 20 else 1

def model(x, t, b, k, m):
    """
    ODE model: spring, mass damp.

    :param x: input x value
    :param t: time value
    :param b: damping
    :param k: spring stiffness
    :param m: mass

    :return dx: derivative x
    """
    dx0 = x[1]
    dx1 = -k/m*x[0] - b/m*x[1] + u_step(t)/m
    dx = [dx0, dx1]

    return dx

def test_model():
    """
    Test model function

    if we pass k,b,m = 0 => no spring, damp, mass in system =>  achieve
    some fault.
    """
    n = 5001
    t = np.linspace(0, 50, n)

    x0 = [0.1, 0]
    b = 1
    k = 100
    m = 1

    y = odeint(model, x0, t, args=(b, k, m))
    plt.plot(t, y)
    plt.show()

if __name__ == "__main__":
    test_model()
