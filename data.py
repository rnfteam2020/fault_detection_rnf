"""
Import data

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

def load(file_name:str, data_type=None):
    """Data loading


    :param file_name:   path to file with data
    :param data_type:   dataset datatype (.mat, .csv, .txt, h5py, etc.)
    :return data:       loaded data
    """
    data = pd.read_csv(file_name,skipinitialspace=True)
    data = {"t"     : np.asarray(data['t']),
            "x"     : np.asarray(data['x']),
            "y"     : np.asarray(data['y']),
            "z"     : np.asarray(data['z'])}

    return data


if __name__ == "__main__":
    pass
