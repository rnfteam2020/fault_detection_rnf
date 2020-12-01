"""
Import data

For debug purpose sine wave is ok :)
"""
import pandas as pd
import numpy as np

def load(file_name:str):
    """Data importing

    :file_name: path to file with data
    :return: data
    """
    data = pd.read_csv(file_name,skipinitialspace=True)
    data = {"t"     : np.asarray(data['t']),
            "x"     : np.asarray(data['x']),
            "y"     : np.asarray(data['y']),
            "z"     : np.asarray(data['z'])}

    return data


if __name__ == "__main__":
    pass
