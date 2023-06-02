import csv
import math
import os
import uuid
import pickle
import random
import shutil
from os import path as osp
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
from DBManager.DBManager import DBManager

# TUNE
#NO OF CANDIDATES
#FRECHET DISTANCE
class Evaluator:

    def fetchRealLocs(self,image_name_in_db,config):
        data_row = None
        # print(config.image_db_meta_file, image_name_in_db)
        with open(config.image_db_meta_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            column_index = header.index('Image_name')

            for row in reader:

                if row[column_index] == image_name_in_db.split(".")[0]:
                    data_row=row
                    break
        data_loc = osp.join(config.invariant_domain_output_dir,data_row[1],data_row[2],"invariant_traj.csv")
        start=int(data_row[3])
        end=int(data_row[4])
        data = np.genfromtxt(data_loc, delimiter=',')[start+1:end+2,[3,4]]
        return data


    def frechet_distance(self,P, Q):
        """
        Compute the Fréchet distance between two trajectories P and Q.
        """
        n = len(P)
        m = len(Q)
        if n != m:
            raise ValueError("Trajectories must have the same length")
        D = cdist(P, Q)
        L = np.zeros((n, m))
        L[0, 0] = D[0, 0]
        for i in range(1, n):
            L[i, 0] = max(L[i-1, 0], D[i, 0])
        for j in range(1, m):
            L[0, j] = max(L[0, j-1], D[0, j])
        for i in range(1, n):
            for j in range(1, m):
                L[i, j] = max(min(L[i-1, j], L[i-1, j-1], L[i, j-1]), D[i, j])
        return L[n-1, m-1]

    def getDTW(self,nplist1, nplist2):
        distance, path = fastdtw(nplist1, nplist2, dist=euclidean)
        return distance

    def calculate_ate(self,expected_locs, predicted_real_loc):
        # Calculate Euclidean distances between corresponding points
        distances = np.linalg.norm(expected_locs - predicted_real_loc, axis=1)

        # Calculate ATE
        ate = np.mean(distances)

        return ate

    def err_dst(self,arr1,i,arr2,j):
        a_row = arr1[i]
        b_row = arr2[j]

        # Calculate the distance using the Euclidean distance formula
        distance = np.sqrt((b_row[0] - a_row[0]) ** 2 + (b_row[1] - a_row[1]) ** 2)

        return distance

    def evaluate(self,new_data_config,building_data_config):
        print("Started Evaluating")
        # Loading KDTree
        tree = None
        tags = None
        with open(building_data_config.kdtree_features_loc, "rb") as f:
            tree = pickle.load(f)
            print("Loaded KDTree")
        with open(building_data_config.kdtree_tags_loc, "rb") as f:
            tags = pickle.load(f)
            print("Loaded Tags")


        dbmanager = DBManager(new_data_config)
        image_files=os.listdir(new_data_config.to_eval_dir)
        random.shuffle(image_files)

        path = new_data_config.root_dir + "\\predictions"
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        expected_locs = []
        layers = []

        for image_file in image_files:

            image_name = image_file
            image_loc = osp.join(new_data_config.to_eval_dir, image_file)
            pil_img = Image.open(image_loc)

            feature = dbmanager.extract_features(pil_img)
            img_dist, ind = tree.query(feature,k=new_data_config.no_of_candidates)
            expected_real_loc = self.fetchRealLocs(image_name, new_data_config)
            expected_locs.append(expected_real_loc)

            fig, ax = plt.subplots(1, 5)
            layer = []
            figname = str(uuid.uuid4())
            ates=[str(figname)]
            errs = [str(figname)]
            for i in range(len(ind)):

                best_image_name = tags[ind[i]]
                best_img_loc = osp.join(building_data_config.image_db_loc_kdtree, best_image_name)
                pil_img_best_match = Image.open(best_img_loc)

                predicted_real_loc = self.fetchRealLocs(best_image_name,building_data_config)
                ax[i].scatter(expected_real_loc[:, 0], expected_real_loc[:, 1], color="red", s=0.1)
                ax[i].scatter(predicted_real_loc[:, 0], predicted_real_loc[:, 1], color="blue", s=0.1)
                ax[i].set_xlim([0, building_data_config.x_lim])
                ax[i].set_ylim([0, building_data_config.y_lim])

                ate_1 = self.calculate_ate(expected_real_loc,predicted_real_loc)
                ate_2 = self.calculate_ate(expected_real_loc, predicted_real_loc[::-1])

                err1 = self.err_dst(expected_real_loc,0,predicted_real_loc,0)
                err2 = self.err_dst(expected_real_loc,0,predicted_real_loc,-1)
                err3 = self.err_dst(expected_real_loc,-1,predicted_real_loc,0)
                err4 = self.err_dst(expected_real_loc,-1,predicted_real_loc,-1)

                ate = min([ate_1,ate_2])
                err = min([err1,err2,err3,err4])
                ax[i].text(0.25, 0.95, f'ATE: {ate:.2f}', transform=ax[i].transAxes, ha='left', va='top')
                ax[i].text(0.15, 0.95, f'ERR: {err:.2f}', transform=ax[i].transAxes, ha='left', va='bottom')

                ates.append(str(ate))
                errs.append(str(err))
                layer.append(predicted_real_loc)



            fig.savefig(path+"\\"+str(figname))
            plt.clf()
            layers.append(layer)

        return layers,expected_locs
