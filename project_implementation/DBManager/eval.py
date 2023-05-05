import csv
import math
import os
import pickle
import random
from os import path as osp
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist

import config
from DBManager.DBManager import DBManager

# TUNE
#NO OF CANDIDATES
#FRECHET DISTANCE
class Evaluator:

    def __init__(self,conf):
        self.conf=conf

# DTW
    def fetchRealLocs(self,image_name_in_db):
        data_row = None
        with open(self.conf.image_db_meta_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            column_index = header.index('Image_name')

            for row in reader:

                if row[column_index] == image_name_in_db.split(".")[0]:
                    data_row=row
                    break
        data_loc = osp.join(self.conf.invariant_domain_output_dir,data_row[1],data_row[2],"invariant_traj.csv")
        start=int(data_row[3])
        end=int(data_row[4])
        data = np.genfromtxt(data_loc, delimiter=',')[start+1:end+2,[3,4]]
        return data

    def fetchRoNINLocs(self,image_name_in_db):
        data_row = None
        with open(self.conf.image_db_meta_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            column_index = header.index('Image_name')

            for row in reader:

                if row[column_index] == image_name_in_db.split(".")[0]:
                    data_row=row
                    break
        data_loc = osp.join(self.conf.invariant_domain_output_dir,data_row[1],data_row[2],"invariant_traj.csv")
        start=int(data_row[3])
        end=int(data_row[4])
        data = np.genfromtxt(data_loc, delimiter=',')[start+1:end+2,[1,2]]
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

    def frechet_dist_1d(self,curve1, curve2):
        x = np.linspace(0, 1, len(curve1))
        curve1_2D = np.column_stack((x, curve1))
        curve2_2D = np.column_stack((x, curve2))
        distance = cdist(curve1_2D, curve2_2D, metric='euclidean')
        frechet_dist = np.max(np.minimum(np.max(distance, axis=1), np.max(distance, axis=0)))
        return frechet_dist

    def evaluate(self):

        print("Started Evaluating")

        # Loading KDTree
        tree = None
        tags = None
        with open(self.conf.kdtree_features_loc, "rb") as f:
            tree = pickle.load(f)
            print("Loaded KDTree")
        with open(self.conf.kdtree_tags_loc, "rb") as f:
            tags = pickle.load(f)
            print("Loaded Tags")

        count = 0
        passed = 0

        eval_files=os.listdir(self.conf.to_eval_dir)
        random.shuffle(eval_files)

        for eval_file in eval_files:

            count += 1

            csv_loc = osp.join(self.conf.to_eval_dir, eval_file)
            eval_seq = np.genfromtxt(csv_loc, delimiter=',')[1:, :]

            eval_ronin_loc = eval_seq[:, [1, 2]]
            expected_real_loc = eval_seq[:,[3,4]]
            eval_feature = eval_seq[:,5]

            print("Finding Best Matching . . .")
            # distance, ind = tree.query(eval_feature, p=float('inf'), eps=1e-9, distance_func=self.frechet_distance,k=1)
            dist=math.inf

            j=0
            for tag in tags:
                j+=1
                # best_prediction = tags[ind]
                predicted_csv_loc = osp.join(self.conf.db_loc_kdtree, tag)
                predicted_seq = np.genfromtxt(predicted_csv_loc, delimiter=',')[1:, :]

                predicted_real_loc = predicted_seq[:,[3,4]]
                predicted_feature = predicted_seq[:, 5]

                current_dist = self.frechet_distance(eval_ronin_loc,predicted_real_loc)
                if j%1000==0:
                    print(j)
                if current_dist<1.00000:
                    dist=current_dist
                    break

            print("Distance :",dist)

            fig, axs = plt.subplots(5, 1, figsize=(6, 6))

            # Add the scatter plots to the plot
            axs[0].scatter(predicted_real_loc[:, 0], predicted_real_loc[:, 1])
            axs[0].set_title('Original Real Trajectory')

            axs[1].scatter(eval_ronin_loc[:, 0], eval_ronin_loc[:, 1])
            axs[1].set_title('Original Evaluated Trajectory')

            xs = [i for i in range(len(predicted_feature))]
            axs[2].plot(xs, predicted_feature)
            axs[2].set_title('Real -> AAL')

            xs = [i for i in range(len(eval_feature))]
            axs[3].plot(xs, eval_feature)
            axs[3].set_title('Eval -> AAL')

            axs[4].scatter(predicted_real_loc[:,0],predicted_real_loc[:,1],color='red',s=0.01)
            axs[4].scatter(expected_real_loc[:,0],expected_real_loc[:,1],color='green',s=0.01)
            axs[4].set_xlim(0,self.conf.x_lim)
            axs[4].set_ylim(0,self.conf.y_lim)
            axs[4].set_title('Real/Predicted Locs')

            # Show the plot
            plt.show()

            # all = {}
            #
            # for i in range(len(ind)):
            #
            #     best_image_name = tags[ind[i]]
            #     best_img_loc = osp.join(self.conf.image_db_loc_kdtree, best_image_name)
            #     pil_img_best_match = Image.open(best_img_loc)
            #
            #     ronin_traj= self.fetchRoNINLocs(best_image_name)
            #     dist = self.frechet_distance(expected_real_loc,ronin_traj)
            #
            #     all[best_image_name] = dist
            #
            # all = sorted(all.items(), key=lambda x: x[1])
            # top_images = [item[0] for item in all[:10]]
            #
            #
            # for i in range(len(top_images)):
            #     best_image_name = top_images[i]
            #     best_img_loc = osp.join(self.conf.image_db_loc_kdtree, best_image_name)
            #     pil_img_best_match = Image.open(best_img_loc)
            #
            #     ##pil_img.show()
            #     #pil_img_best_match.show()
            #
            #     # if img_dist[i]>20:
            #     #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
            #     #     ax1.imshow(pil_img)
            #     #     ax2.imshow(pil_img_best_match)
            #     #     ax3.text(0.1, 0.5, 'Image Distance :'+str(img_dist[i]), fontsize=12)
            #     #     ax1.set_title('Input')
            #     #     ax2.set_title('Best Match')
            #     #     plt.show()
            #
            #
            #     predicted_real_loc = self.fetchRealLocs(best_image_name)
            #
            #     #print(expected_real_loc[0])
            #     #print(expected_real_loc[-1])
            #
            #     traj_dist=self.frechet_distance(expected_real_loc,predicted_real_loc)
            #     if traj_dist<=config.frechet_distance_threshold :
            #         # fig, ax = plt.subplots(figsize=(5, 5))
            #         #
            #         # ax.scatter(x=expected_real_loc[:, 0], y=expected_real_loc[:, 1],s=0.1,  c="blue", label="Pred")
            #         # ax.scatter(x=predicted_real_loc[:, 0], y=predicted_real_loc[:, 1],s=0.1, c="red",
            #         #             label="Real")
            #         # ax.set_xlim(0, 200)
            #         # ax.set_ylim(0, 200)
            #         # text = "Traj Disrance (Frechet) : " + str(traj_dist)
            #         # x_lim = plt.gca().get_xlim()
            #         # y_lim = plt.gca().get_ylim()
            #         # x_text = (x_lim[1] - x_lim[0]) / 2 + x_lim[0]
            #         # y_text = y_lim[1] - (y_lim[1] - y_lim[0]) * 0.05
            #         # plt.text(x_text, y_text, text, ha='center', va='top', fontsize=12)
            #         # plt.legend()
            #         # plt.show()
            #         ccount+=1
            #         ck+=i
            #         ctraj_dist+=traj_dist
            #         print("K:",i)
            #         print("Traj Dist",traj_dist)
            #         passed+=1
            #         break
            #
            #
            # print("Processed : {} / {}, Current Acc : {} %".format(count,len(image_files),passed*100/count))

        print()
        print("total :",count)
        print("passed :",passed)
        print("acc :",passed*100/count,"%")