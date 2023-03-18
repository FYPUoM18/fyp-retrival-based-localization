import math
import os
import numpy as np
from matplotlib import pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


# DTW
def getDTW(nplist1,nplist2):

    distance, path = fastdtw(nplist1, nplist2, dist=euclidean)

    # print the distance
    return distance

# Check Overlapping
def jaccard_overlapping(seq1, seq2):
    set1 = set(map(tuple, seq1))
    set2 = set(map(tuple, seq2))
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    jaccard = intersection / union
    overlap_percentage = jaccard * 100
    return overlap_percentage

# Convert To Polar Coordinates
def toPolar(coordinates_list_2d):
    x0, y0 = coordinates_list_2d[0]  # get the x, y coordinates of the first point
    r_theta = []
    for x, y in coordinates_list_2d:
        dx = x - x0
        dy = y - y0
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        r_theta.append(r)
        r_theta.append(theta)
    return np.array(r_theta)


new_seq_dir = "C:\\Users\\mashk\\MyFiles\\Semester 8\\FYP\\code\\project_implementation\\outputs\\invariant\\train"
db_directory = "C:\\Users\\mashk\\MyFiles\\Semester 8\\FYP\\code\\project_implementation\\outputs\\invariant\\db"

# Loading DB
db = {}
for dataset in os.listdir(db_directory):
    csv_path = os.path.join(db_directory, dataset, "invariant_traj.csv")
    db[dataset] = np.genfromtxt(csv_path, delimiter=',')[1:, :]
print("DB Loaded\n==========\n")


ispassed=[]
count=0
for dataset in os.listdir(new_seq_dir):
    count+=1
    # Load New Sequence

    print("==========================\nDataSet :",dataset)
    print("Count :",count)
    new_csv_path = os.path.join(new_seq_dir,dataset, "invariant_traj.csv")
    new_seq = np.genfromtxt(new_csv_path, delimiter=',')[1:, :]
    polar_r_new_seq = toPolar(new_seq[:, [1, 2]])
    print("New Seq Loaded")

    # Window Size
    window_size = len(new_seq)
    step_size = 1
    print("Window Size", window_size)
    print("Step Size", step_size)

    # Track Sliding
    current_best_dataset = None
    current_best_window = None
    current_best_dist = math.inf

    # Start Sliding
    for key, val in db.items():

        # Building Sliding Window Parameter
        max_index = len(val) - 1
        current_start_index = 0
        current_end_index = current_start_index + window_size - 1

        while current_end_index <= max_index:

            # Get A Sliding Window
            window = val[current_start_index:current_end_index + 1, :]
            current_start_index += 1
            current_end_index += 1

            # Calc Polar Values
            polar_r_windows = toPolar(window[:, [1, 2]])


            # Cal Similarity
            dist = np.sqrt(np.sum((polar_r_new_seq - polar_r_windows) ** 2))
            if dist < current_best_dist:
                current_best_dist = dist
                current_best_window = window
                current_best_dataset = key

    gtw=getDTW(current_best_window[:, [3, 4]],new_seq[:, [3, 4]])
    if gtw<=100:
        ispassed.append(1)
    else:
        ispassed.append(0)

    print("Best Dist :", current_best_dist)
    print("Best Dataset :", current_best_dataset)
    print("Best Location :", current_best_window[0, [3, 4]])
    print("Expected Location :", new_seq[0, [3, 4]])
    print("DTW Distance:",gtw)
    print("Is Passed :",ispassed[-1])
    print("==========================\n\n")

    #
    # plt.scatter(x=current_best_window[:,1]-current_best_window[0,1], y=current_best_window[:,2]-current_best_window[0,2], s=1, linewidths=0.1, c="blue", label="Best Matching")
    # plt.scatter(x=new_seq[:,1], y=new_seq[:,2], s=1, linewidths=0.1, c="red", label="Input")
    # #plt.axis('off')
    # plt.show()

print(ispassed)
print("Total :",len(ispassed))
print("Train Accuracy :",len([i for i in ispassed if i==1])/len(ispassed))
