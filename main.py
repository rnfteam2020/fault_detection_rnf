"""
Main
"""

from core.nn_fit import fit
from core.dataset import generate_dataset
from core.nn_models import FDModel
from core.data_processing import generate_verification_data
import torch
import core.visualization as vi

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(path2save=None):
    batch_size = 5
    features_shape, labels_shape, train_loader, val_loader = generate_dataset(batch_size=batch_size,
                                                                    shuffle=False, num_workers=5)
    datasets = train_loader, val_loader

    net = FDModel(20,1,16).to(DEVICE)
    epochs, losses_data, val_losses_data = fit(net, datasets, batch_size=batch_size,
                                               epochs=500)
    vi.plot_loss(epochs, losses_data, 'Loss function', 'epochs', 'loss')
    vi.plot_loss(epochs, val_losses_data, 'Evaluation', 'epochs', 'evaluation loss')

    if path2save is not None:
        torch.save(net.state_dict(), path2save)

def run(path2net):

    x_verif, y_verif, signals_data = generate_verification_data()

    net = FDModel(20,1,16).to(DEVICE)
    net.load_state_dict(torch.load(path2net))
    net.eval()


    for i,signal in enumerate(signals_data):
        t = signal[0]
        u = signal[1]
        y = signal[2]

        title = 'Health data' if i == 0 else 'Fault data'
        vi.plot_data(t, y, title, 't [s]' , 'position, velocity [m]')

    for i, (x, y) in enumerate(zip(x_verif, y_verif)):
        x = torch.from_numpy(x).float().to(DEVICE)
        y = torch.from_numpy(y).float().to(DEVICE)
        title = 'Health data' if i == 0 else 'Fault data '
        print(f'[TEST] CASE {i+1} "{title}" : label={y[0]} net_output={net(x)[0]:.4f}')




if __name__ == "__main__":
    path2net = './net_saved/demo_net.pth'

    # train(path2net)
    run(path2net)
