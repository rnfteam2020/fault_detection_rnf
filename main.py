"""
Main
"""

from core.nn_fit import fit
from core.dataset import generate_dataset
from core.nn_models import FDModel
import torch
import core.visualization as vi

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    batch_size = 1
    features_shape, labels_shape, train_loader, val_loader = generate_dataset(batch_size=batch_size,
                                                                    shuffle=False, num_workers=5)
    datasets = train_loader, val_loader

    net = FDModel(20,1,16).to(DEVICE)
    epochs, losses_data, val_losses_data = fit(net, datasets, batch_size=batch_size,
                                               epochs=200)
    vi.plot_loss(epochs, losses_data)
    vi.plot_loss(epochs, val_losses_data)
    net.save('demo_net')

def run():
    pass

if __name__ == "__main__":
    train()
