import torch
import torch.nn as nn
import core.visualization as vi

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_train_step(net, loss_function, optimizer):
    def train_step(x_train, y_train):
        net.train()
        y_hat = net.forward(x_train)
        loss = loss_function(y_hat, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step


def fit(net, dataset, lr=0.05, epochs=1000, batch_size=None):

    bar = Bar('Epochs', max=epochs)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # train_loader
    train_loader = generate_train_loader() # TODO
    train_step = make_train_step(net, loss_function, optimizer)

    losses_data = []

    for epoch in range(epochs):

        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            loss = train_step(x_batch, y_batch)
            losses_data.append(loss)
        bar.next()

    bar.finish()

    vi.plot_loss(epochs, losses_data)

if __name__ == "__main__":
    pass
