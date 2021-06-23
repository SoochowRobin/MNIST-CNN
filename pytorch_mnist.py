import os
import torch
import torch.nn as nn

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# check if cuda could use
use_cuda = torch.cuda.is_available()

# make data folder to store dataset
root = './data'
if not os.path.exists(root):
    os.mkdir(root)

trans = transforms.Compose([
    transforms.ToTensor(),
    # because these is one channel, so the Normalize((mean), (std)) is a tuple and one element in it
    # for 3 channels, RGB transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.5,), (1.0,))
])


train_set = dset.MNIST(root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root, train=False, transform=trans, download=True)

# define hyperparameter
batch_size, lr = 100, 0.01

# dataloader
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))