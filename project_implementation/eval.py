import csv
import math
import os
import pickle
from os import path as osp
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import config
from DBManager.DBManager import DBManager

train_dir = "C:\\Users\\mashk\\MyFiles\\Semester 8\\FYP\\code\\project_implementation\\outputs\\5. imageDB\\train"
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


def getDTW(nplist1, nplist2):
    distance, path = fastdtw(nplist1, nplist2, dist=euclidean)
    return distance



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
dbmanager = DBManager(config)
image_files=os.listdir(train_dir)
for image_file in image_files:
    count += 1

    image_name = image_file
    image_loc = osp.join(train_dir, image_file)
    pil_img = Image.open(image_loc)

    feature = dbmanager.extract_features(pil_img)
    dist, ind = tree.query(feature, k=10)

    best_image_name = tags[ind[0]]
    best_img_loc = osp.join(db_dir, best_image_name)
    pil_img_best_match = Image.open(best_img_loc)

    #pil_img.show()
    #pil_img_best_match.show()

    expected_real_loc = fetchRealLocs(image_name)
    predicted_real_loc = fetchRealLocs(best_image_name)

    dtw=getDTW(expected_real_loc,predicted_real_loc)
    if dtw<=100:
        passed+=1

    print("Processed : {} / {}, Current Acc : {} %".format(count,len(image_files),passed*100/count))

print()
print("total :",count)
print("passed :",passed)
print("acc :",passed*100/count,"%")