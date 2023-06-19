
import pickle
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import shutil
import pickle
import csv
from os import path as osp
from PIL import Image
from DBManager.DBManager import DBManager
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


class DataStorer:

    def __init__(self,conf):
        self.conf=conf

    def fetchRealLocs(self,image_name_in_db):
        data_row = None
        with open(self.conf.db_meta_csv, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            column_index = header.index('Image_name')

            for row in reader:
                if row[column_index] == image_name_in_db.split(".")[0]:
                    data_row = row
                    break
        data_loc = osp.join(self.conf.invariant_dir, data_row[1], data_row[2], "invariant_traj.csv")
        start = int(data_row[3])
        end = int(data_row[4])
        data = np.genfromtxt(data_loc, delimiter=',')[start + 1:end + 2, [3, 4]]
        return data

    def generateImages(self,filepath,data_out_dir):

        # Initiate Window Parameters
        window_size = self.conf.window_size
        step_size = self.conf.step_size
        # print("File Path",filepath)
        # print("Window Size", window_size)
        # print("Step Size", step_size)

        # Load Sequence
        csv_path = os.path.join(filepath, "invariant_traj.csv")
        new_seq = np.genfromtxt(csv_path, delimiter=',')[1:, :]

        # Check Whether Minimum Length Found
        if len(new_seq) < window_size:
            # print("Failed Insufficient Length !!!")
            exit()
        # else:
            # print("Started Processing ")

        # Building Sliding Window Parameter
        max_index = len(new_seq) - 1
        current_start_index = 0
        current_end_index = current_start_index + window_size - 1

        # List to Store PIL Images
        locs=[]
        i=0
        while current_end_index <= max_index:
            i+=1
            # Get A Sliding Window
            window = new_seq[current_start_index:current_end_index + 1, :]
            current_start_index += step_size
            current_end_index += step_size

            # Generate and Save Image
            x = window[:, 1]
            y = window[:, 2]

            plt.plot(x, y, linestyle='-')
            plt.axis('off')
            # fig = plt.gcf()
            # img = self.PLTToPIL(fig)
            # pil_imgs.append(img)
            img_loc=osp.join(data_out_dir,str(i))
            locs.append(img_loc)
            plt.savefig(img_loc, dpi=40)
            plt.clf()

        return [Image.open(img+".png") for img in locs]


    def process(self,loc,name):

        # Loading KDTree
        tree = None
        tags = None
        with open(self.conf.kdtree_features_loc, "rb") as f:
            tree = pickle.load(f)
            print("Loaded KDTree")
        with open(self.conf.kdtree_tags_loc, "rb") as f:
            tags = pickle.load(f)
            print("Loaded Tags")

        # List Directory
        files=os.listdir(loc)
        print("No of Files Found :",len(files))
        count=0

        # Setup Directories
        extracted_data_out_dir = osp.join(self.conf.extracted_data_output_loc, name)
        if os.path.exists(extracted_data_out_dir):
            shutil.rmtree(extracted_data_out_dir)
        os.makedirs(extracted_data_out_dir)

        # Data
        input_data=[]
        output_data=[]

        for file in files:
            count+=1
            print(count,"/",len(files))

            try:
                # Setup Directories
                data_out_dir = osp.join(self.conf.extracted_data_output_loc,name,file)
                if os.path.exists(data_out_dir):
                    shutil.rmtree(data_out_dir)
                os.makedirs(data_out_dir)

                # Generate Images
                fullPath=osp.join(loc,file)
                pil_imgs = self.generateImages(fullPath,data_out_dir)
                # print("Generated {length} Images".format(length=len(pil_imgs)))

                # Get Real Location
                csv_path = os.path.join(fullPath, "invariant_traj.csv")
                real_loc = np.genfromtxt(csv_path, delimiter=',')[1:, [3,4]]


                # Extract Features
                # print("Extracting Features  . . .")
                dbmanager=DBManager(self.conf)
                features = dbmanager.extract_features_all(pil_imgs)
                # print("Features Shape :",features.shape)

                # Find Best Matching Locations
                print("Finding Best Matching Locations . . .")
                subTraj=0
                traj_layers=[]
                for feature in features:
                    subTraj+=1
                    print("-- For Sub Trajectory:",subTraj)

                    layer=[]
                    img_dist, ind = tree.query(feature, k=self.conf.no_of_candidates)
                    for i in range(len(ind)):
                        best_image_name = tags[ind[i]]
                        predicted_real_loc = self.fetchRealLocs(best_image_name)
                        layer.append(predicted_real_loc)
                        #plt.scatter(x=predicted_real_loc[:, 0], y=predicted_real_loc[:, 1],s=0.01, linestyle='-')
                    traj_layers.append(layer)

                input_data.append(traj_layers)
                output_data.append(np.mean(real_loc, axis=0))

            except:
                continue

            if count%10==0:
                with open(extracted_data_out_dir+"\\features-"+str(count)+".pickle", "wb") as f:
                    pickle.dump(input_data, f)
                with open(extracted_data_out_dir+"\\labels-"+str(count)+".pickle", "wb") as f:
                    pickle.dump(output_data, f)

        with open(extracted_data_out_dir+"\\features.pickle", "wb") as f:
            pickle.dump(input_data, f)
        with open(extracted_data_out_dir+"\\lables.pickle", "wb") as f:
            pickle.dump(output_data, f)

        print("Data Stored . . .")