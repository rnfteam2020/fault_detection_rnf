"""
Main
"""

from core.nn_fit import fit
from core.dataset import generate_dataset
from core.nn_models import FDModel
import core.visualization as vi

def train():
    batch_size = 1
    shape, train_loader = generate_dataset(batch_size=batch_size, shuffle=False,
                                           num_workers=1)

    net = FDModel()
    epochs, losses_data = fit(net, train_loader, batch_size=batch_size)
    vi.plot_loss(epochs, losses_data)

def run():
    pass

if __name__ == "__main__":
    pass
