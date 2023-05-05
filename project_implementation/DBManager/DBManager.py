import os
import pickle
import uuid
import numpy as np
from multiprocessing import Pool
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from os import path as osp
from PIL import Image
from matplotlib import pyplot as plt
from scipy.spatial import KDTree


class DBManager:

    def __init__(self, conf):
        self.conf = conf
        # self.model = models.inception_v3(pretrained=True)
        # self.model.eval()
        self.model = models.vgg19(pretrained=True).features
        self.model.eval()

    def generateImageDB(self):

        fail_list = []
        csv_data = [["Image_name", "invariant folder", "invariant dataset", "start", "end"]]

        # Initiate Window Parameters
        window_size = self.conf.window_size
        step_size = self.conf.step_size
        print("Window Size", window_size)
        print("Step Size", step_size)

        # Iterate Over DB
        for folder in os.listdir(self.conf.invariant_domain_output_dir):
            folder_root = osp.join(self.conf.invariant_domain_output_dir, folder)
            for dataset in os.listdir(folder_root):
                dataset_root = osp.join(folder_root, dataset)
                csv_path = os.path.join(dataset_root, "invariant_traj.csv")

                # Load Sequence
                try:
                    new_seq = np.genfromtxt(csv_path, delimiter=',')[1:, :]
                except:
                    continue

                # Check Whether Minimum Length Found
                if len(new_seq) < window_size:
                    fail_list.append(csv_path)
                    print("Failed DB :", folder, dataset)
                    continue

                else:
                    print("Processing DB :", folder, dataset)

                # Building Sliding Window Parameter
                max_index = len(new_seq) - 1
                current_start_index = 0
                current_end_index = current_start_index + window_size - 1

                while current_end_index <= max_index:

                    # Initialize
                    image_name = str(uuid.uuid4())
                    sub_db_loc = osp.join(self.conf.image_db_loc, folder)
                    image_loc = osp.join(sub_db_loc, image_name + ".png")
                    if not os.path.exists(sub_db_loc):
                        os.makedirs(sub_db_loc)

                    # Maintain Meta Data
                    csv_data.append([image_name, folder, dataset, current_start_index, current_end_index])
                    # "Image_name","folder","dataset","start","end"

                    # Get A Sliding Window
                    window = new_seq[current_start_index:current_end_index + 1, :]
                    current_start_index += step_size
                    current_end_index += step_size

                    # Generate and Save Image
                    # if folder =="db":
                    #     x = window[:, 3] - window[0,3]
                    #     y = window[:, 4] - window[0,4]
                    # else:
                    x = window[:, 1]
                    y = window[:, 2]


                    plt.plot(x, y,linewidth=10, linestyle='-')
                    plt.axis('off')
                    plt.savefig(image_loc, dpi=40)
                    plt.clf()

        print("No of failures :", len(fail_list))
        np.savetxt(self.conf.image_db_meta_file, csv_data, delimiter=",", fmt='%s')
        print("Save DB Info")

    def extract_features(self, img):
        img = img.convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            features = self.model(img)
        features = features.detach().numpy()
        features = np.ravel(features)
        return features


    # Define function to extract features from a list of images using multiple processes
    def extract_features_all(self, images):
        total=len(images)
        count=0
        feature_list=[]
        for img in images:
            feature_list.append(self.extract_features(img))
            count+=1
            if count%10==0:
                print("Completed {} / {}".format(count,total))
        features_array = np.array(feature_list)
        return features_array

    def buildKDTree(self):

        tags = []
        #images=[]
        features = []
        i=0
        print("Loading Images . . .")
        for image in os.listdir(self.conf.image_db_loc_kdtree):
            i=i+1
            image_loc = osp.join(self.conf.image_db_loc_kdtree, image)
            pil_img = Image.open(image_loc)
            #images.append(pil_img)
            feature = self.extract_features_all([pil_img])[0]
            if i%100==0:
                print("Feature Shape :",i,",", feature.shape)
            features.append(feature)
            tags.append(image)
            pil_img.close()
        print("Extracted Features")


        tree=KDTree(features)
        with open(self.conf.kdtree_features_loc, "wb") as f:
            pickle.dump(tree, f)
        with open(self.conf.kdtree_tags_loc, "wb") as f:
            pickle.dump(tags, f)

