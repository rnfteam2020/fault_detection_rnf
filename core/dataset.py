"""
Create a dataset
"""
from torch.utils.data import Dataset, DataLoader
import torch

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


def generate_trainloader(dataset, batch_size=None, shuffle=False, num_workers=1):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    pass
