"""
Create a dataset

"""
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import torch
from core.data_processing import generate_statistic_features

class CustomDataset(Dataset):
    """
    Create a dataset from data

    >TensorDataset can be used instead this CustomDataset

    """
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return self.n_samples


def generate_dataset(batch_size=1, shuffle=False, num_workers=1):
    features, labels = generate_statistic_features()
    dataset = CustomDataset(features, labels)

    train_dataset, val_dataset = random_split(dataset,
                                                [int(len(dataset)*0.8),
                                                 int(len(dataset)*0.2)])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers)

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers)

    return features.shape, labels.shape, train_loader, val_loader


if __name__ == "__main__":
    pass
