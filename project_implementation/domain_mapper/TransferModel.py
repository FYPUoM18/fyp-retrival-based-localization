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
    def __init__(self, encoder):
        super(ContrastiveModel, self).__init__()
        self.encoder = encoder

    def forward_once(self, x):
        h = self.encoder(x)
        z = h.view(h.size(0), -1)
        return z

    def forward(self, x1, x2):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2