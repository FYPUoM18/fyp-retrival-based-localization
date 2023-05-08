from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import torch.nn.functional as F
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class ContrastiveModel(nn.Module):
    def __init__(self):
        super(ContrastiveModel, self).__init__()

        # Load Pretrained VGG19 features
        self.vgg = models.vgg19(pretrained=True).features
        # Freeze all VGG19 layers
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Add new layers
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)

    def forward_once(self, x):
        x = self.vgg(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x1, x2):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2