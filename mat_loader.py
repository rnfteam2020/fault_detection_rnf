""".mat format import options

Possible import examples for .mat formats:
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
import h5py
import pandas as pd
import scipy as sp


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
