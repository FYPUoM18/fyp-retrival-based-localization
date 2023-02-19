import h5py
import argparse
import json
import os
import uuid
import shutil
import sys
import matplotlib.pyplot as plt
import collections
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
from pathlib import Path
from PIL import Image
from os import path as osp


class FeatureExtractor:

    def __init__(self,conf) -> None:
        self.conf=conf
        base_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(base_model.features.children()) + [torch.nn.Flatten(),
                                               base_model.classifier[0], base_model.classifier[1]])
        self.model.eval()
    
    def extract(self, img):
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        img = transform(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            feature = self.model(img)[0]
            return feature / torch.norm(feature)

    def generate_feature_db(self):

        # Create DB Location
        if os.path.exists(self.conf.feature_db_dir):
                shutil.rmtree(self.conf.feature_db_dir)
        os.makedirs(self.conf.feature_db_dir)

        # Loop Over Images, Generate Features, Save Features
        for img_name in os.listdir(osp.join(self.conf.traj_drawing_out_dir,"train")):
            feature = self.extract(img=Image.open(osp.join(self.conf.traj_drawing_out_dir,"train",img_name)).convert('RGB'))
            feature_name = str(img_name).split(".")[0]+".pt"
            feature_path=osp.join(self.conf.feature_db_dir,feature_name)
            torch.save(feature, feature_path)
            print(feature)
            print("Generated Features For ",img_name)
    
    def load_features_db(self):
        feature_nps = []
        feature_locs= []
        for feature in os.listdir(self.conf.feature_db_dir):
            feature_nps.append(torch.load(osp.join(self.conf.feature_db_dir,feature)).numpy())
            feature_locs.append(str(feature).split("-")[0])
        print("Feature DB Loaded")
        return np.array(feature_nps),feature_locs
        
    # Find Most Matching
    def find_best_matchings(self,expected_label,image_path,feature_nps,feature_locs,no_of_indexes=10):
        query = self.extract(img=Image.open(image_path).convert('RGB')).numpy()
        dists=np.linalg.norm(feature_nps-query, axis=1)
        ids = np.argsort(dists)
        found=False
        for i in range(len(ids)):
            if feature_locs[ids[i]]==expected_label:
                print("Expected Label :",expected_label)
                print("Rank :",i+1)
                print("Distance :",dists[ids[i]])
                found=True
                break
        if not found:
            print("Not Found")
        