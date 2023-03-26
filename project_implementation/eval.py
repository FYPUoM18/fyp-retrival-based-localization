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
cdtw=0
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
    dist, ind = tree.query(feature,k=100)
    for i in range(len(ind)):
        best_image_name = tags[ind[i]]
        best_img_loc = osp.join(db_dir, best_image_name)
        pil_img_best_match = Image.open(best_img_loc)

        ##pil_img.show()
        #pil_img_best_match.show()

        expected_real_loc = fetchRealLocs(image_name)
        predicted_real_loc = fetchRealLocs(best_image_name)

        #print(expected_real_loc[0])
        #print(expected_real_loc[-1])

        dtw=getDTW(expected_real_loc,predicted_real_loc)
        if dtw<=400:
            # plt.scatter(x=expected_real_loc[:, 0], y=expected_real_loc[:, 1],  c="blue", label="Pred")
            # plt.scatter(x=predicted_real_loc[:, 0], y=predicted_real_loc[:, 1], c="red",
            #             label="Real")
            # plt.show()
            # plt.clf()
            ccount+=1
            ck+=i
            cdtw+=dtw
            print("K:",i)
            print("DTW",dtw)
            passed+=1
            break


    print("Processed : {} / {}, Current Acc : {} %".format(count,len(image_files),passed*100/count))

print()
print("Avg DTW:",cdtw/ccount)
print("Avg K:",ck/ccount)
print("total :",count)
print("passed :",passed)
print("acc :",passed*100/count,"%")