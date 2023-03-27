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

to_eval_dir = "C:\\Users\\mashk\\MyFiles\\Semester 8\\FYP\\code\\project_implementation\\outputs\\5. imageDB\\test"
db_meta_csv = "C:\\Users\\mashk\\MyFiles\\Semester 8\\FYP\\code\\project_implementation\\outputs\\image_db_meta_file" \
              ".csv"
db_dir = "C:\\Users\\mashk\\MyFiles\\Semester 8\\FYP\\code\\project_implementation\\outputs\\5. imageDB\\db"
kdtree_features = "C:\\Users\\mashk\\MyFiles\\Semester 8\\FYP\\code\\project_implementation\\outputs\\kdtree_features" \
                  ".pickle"
kdtree_tags = "C:\\Users\\mashk\\MyFiles\\Semester 8\\FYP\\code\\project_implementation\\outputs\\kdtree_tags.pickle"
invariant_dir = "C:\\Users\\mashk\\MyFiles\\Semester 8\\FYP\\code\\project_implementation\\outputs\\4. invariant"

# DTW
def fetchRealLocs(image_name_in_db):
    data_row = None
    with open(db_meta_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        column_index = header.index('Image_name')

        for row in reader:

            if row[column_index] == image_name_in_db.split(".")[0]:
                data_row=row
                break
    data_loc = osp.join(invariant_dir,data_row[1],data_row[2],"invariant_traj.csv")
    start=int(data_row[3])
    end=int(data_row[4])
    data = np.genfromtxt(data_loc, delimiter=',')[start+1:end+2,[3,4]]
    return data


def frechet_distance(P, Q):
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

def getDTW(nplist1, nplist2):
    distance, path = fastdtw(nplist1, nplist2, dist=euclidean)
    return distance


print("Started")
# Loading KDTree
tree = None
tags = None
with open(kdtree_features, "rb") as f:
    tree = pickle.load(f)
    print("Loaded KDTree")
with open(kdtree_tags, "rb") as f:
    tags = pickle.load(f)
    print("Loaded Tags")

count = 0
passed = 0
ccount=0
ctraj_dist=0
ck=0
dbmanager = DBManager(config)
image_files=os.listdir(to_eval_dir)
random.shuffle(image_files)
for image_file in image_files:
    count += 1

    image_name = image_file
    image_loc = osp.join(to_eval_dir, image_file)
    pil_img = Image.open(image_loc)

    feature = dbmanager.extract_features(pil_img)
    img_dist, ind = tree.query(feature,k=50)
    for i in range(len(ind)):
        best_image_name = tags[ind[i]]
        best_img_loc = osp.join(db_dir, best_image_name)
        pil_img_best_match = Image.open(best_img_loc)

        ##pil_img.show()
        #pil_img_best_match.show()

        # if img_dist[i]>20:
        #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
        #     ax1.imshow(pil_img)
        #     ax2.imshow(pil_img_best_match)
        #     ax3.text(0.1, 0.5, 'Image Distance :'+str(img_dist[i]), fontsize=12)
        #     ax1.set_title('Input')
        #     ax2.set_title('Best Match')
        #     plt.show()




        expected_real_loc = fetchRealLocs(image_name)
        predicted_real_loc = fetchRealLocs(best_image_name)

        #print(expected_real_loc[0])
        #print(expected_real_loc[-1])

        traj_dist=frechet_distance(expected_real_loc,predicted_real_loc)
        if traj_dist<=10 :
            # fig, ax = plt.subplots(figsize=(5, 5))
            #
            # ax.scatter(x=expected_real_loc[:, 0], y=expected_real_loc[:, 1],s=0.1,  c="blue", label="Pred")
            # ax.scatter(x=predicted_real_loc[:, 0], y=predicted_real_loc[:, 1],s=0.1, c="red",
            #             label="Real")
            # ax.set_xlim(0, 200)
            # ax.set_ylim(0, 200)
            # text = "Traj Disrance (Frechet) : " + str(traj_dist)
            # x_lim = plt.gca().get_xlim()
            # y_lim = plt.gca().get_ylim()
            # x_text = (x_lim[1] - x_lim[0]) / 2 + x_lim[0]
            # y_text = y_lim[1] - (y_lim[1] - y_lim[0]) * 0.05
            # plt.text(x_text, y_text, text, ha='center', va='top', fontsize=12)
            # plt.legend()
            # plt.show()
            # plt.clf()
            ccount+=1
            ck+=i
            ctraj_dist+=traj_dist
            print("K:",i)
            print("Traj Dist",traj_dist)
            passed+=1
            break


    print("Processed : {} / {}, Current Acc : {} %".format(count,len(image_files),passed*100/count))

print()
print("Avg Traj Dist:",ctraj_dist/ccount)
print("Avg K:",ck/ccount)
print("total :",count)
print("passed :",passed)
print("acc :",passed*100/count,"%")