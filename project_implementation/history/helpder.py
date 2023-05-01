import config
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from os import path as osp
import pickle
import matplotlib.pyplot as plt

extracted_data_out_dir = osp.join(config.extracted_data_output_loc, "val")
features=None
labels=None

with open(extracted_data_out_dir + "\\features.pickle", "rb") as f:
    features = pickle.load(f)
with open(extracted_data_out_dir + "\\lables.pickle", "rb") as f:
    labels = pickle.load(f)

processed_features=[]
processed_labels=[]

print(len(features),len(labels))

for i in range(len(features)):

    print("i :",i)

    np_feature = np.array(features[i])

    remove_from_top=True

    if np_feature.shape[0]<5:
        continue

    else:
        while np_feature.shape[0]!=5:
            if remove_from_top:
                np_feature = np_feature[1:, :, :, :]
                remove_from_top=False
            else:
                np_feature = np_feature[:-1, :, :, :]
                remove_from_top=True

    np_feature = np_feature[:, :, [0,29,59,89,119], :]
    print(np_feature.shape)

    processed_labels.append(labels[i])
    processed_features.append(np_feature)

with open(extracted_data_out_dir + "\\p-features.pickle", "wb") as f:
    pickle.dump(processed_features,f)
with open(extracted_data_out_dir + "\\p-lables.pickle", "wb") as f:
    pickle.dump(processed_labels,f)
