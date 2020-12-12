def fit(net, dataset, lr=0.05, epochs=1000, batch_size=None):

    bar = Bar('Epochs', max=epochs)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss_data = []

    if batch_size is None:
        x_train, y_train = dataset
        for epoch in range(epochs):
            y = net.forward(x_train)
            loss = loss_function(y, y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            bar.next()
            loss_data.append(loss.item())
        bar.finish()

    else:

        total_samples = len(dataset)
        n_iterations = total_samples//batch_size

        for epoch in range(epochs):
            net.train()
            for i in range(n_iterations):
                x_train, y_train = dataset.next()
                y = net.forward(x_train)
                loss = loss_function(y, y_train)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_data.append(loss.item())
            bar.next()
        bar.finish()

    vi.plot_loss(epochs, loss_data)



if __name__ == "__main__":
    net = FDModel(1,1,4).to(DEVICE)
    t, u, y = generate_data_from_model()
    dataset = CustomDataset(u, y)
    dataset = generate_dataset(dataset, batch_size=1)
    print(dataset.next())

    fit(net, dataset, batch_size=1)
