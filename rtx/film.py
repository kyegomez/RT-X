import torch
from torch import nn
from PIL import Image


class Film(nn.Module):
    def __init__(self, num_channels):
        super(Film, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        return x * self.gamma * self.beta
