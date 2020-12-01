"""Core nn script

"""
import torch
import time

class BaseNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model_name = str(self)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name:str):
        if name is None:
            prefix = self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        else:
            prefix = name + '_'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name






if __name__ == "__main__":
    pass
