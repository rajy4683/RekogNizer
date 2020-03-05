import torch
import torchvision
import torchvision.transforms as transforms
from RekogNizer.hyperparams import *



classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=2):
    return torch.utils.data.DataLoader(dataset, batch_size,
                                         shuffle=shuffle, num_workers=num_workers)
