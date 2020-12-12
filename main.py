"""
Main
"""

from core.nn_fit import fit
from core.dataset import generate_dataset
from core.nn_models import FDModel

def train():
    train_loader = generate_dataset(batch_size=1, shuffle=False,
            num_workers=1)

    dataset_shape = train_loader
    net = FDModel(

def run():
    pass

if __name__ == "__main__":
    pass
