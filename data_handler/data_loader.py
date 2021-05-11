from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import numpy as np

class MnistDataLoader:
    def __init__(self, train_batch_size, test_batch_size, data_dir):
        # Download dataset
        transform = transforms.ToTensor()
        self.training_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        self.testing_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

        #splt daraser
        self.train_data, self.val_data = torch.utils.data.random_split(self.training_dataset, [50000, 10000])

        # Load the data
        self.train_loader = DataLoader(self.train_data, batch_size=train_batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=test_batch_size, shuffle=False)
        self.test_loader = DataLoader(self.testing_dataset, batch_size=len(self.testing_dataset), shuffle=False)
