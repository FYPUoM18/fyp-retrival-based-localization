import math
import os
import numpy as np
from numpy.lib.stride_tricks import as_strided

new_seq_loc="C:\\Users\\mashk\\MyFiles\\Semester 8\\FYP\\code\\project_implementation\\outputs\\invariant\\train\\0eaa4181-859b-4409-9f34-bd146e134c2c"
db_directory="C:\\Users\\mashk\\MyFiles\\Semester 8\\FYP\\code\\project_implementation\\outputs\\invariant\\db"

# Loading DB
db={}
for dataset in os.listdir(db_directory):
    csv_path=os.path.join(db_directory,dataset,"invariant_traj.csv")
    db[dataset]=np.genfromtxt(csv_path, delimiter=',')[1:,:]
print("DB Loaded")


# Load New Sequence
new_csv_path=os.path.join(new_seq_loc,"invariant_traj.csv")
new_seq=np.genfromtxt(new_csv_path, delimiter=',')[1:,:]
print("New Seq Loaded")

# Window Size
window_size=len(new_seq)
step_size=1
print("Window Size",window_size)
print("Step Size",step_size)

# Track Sliding
current_best_dataset=None
current_best_window=None
current_best_dist=math.inf

# Start Sliding
for key,val in db.items():

    # Building Sliding Window Parameter
    max_index=len(val)-1
    current_start_index=0
    current_end_index=current_start_index+window_size-1

    while current_end_index<=max_index:

        # Get A Sliding Window
        window = val[current_start_index:current_end_index+1,:]
        current_start_index+=1
        current_end_index+=1

        # Cal Similarity
        dist = np.sqrt(np.sum((window[:,5] - new_seq[:,5]) ** 2))
        if dist<current_best_dist:
            current_best_dist=dist
            current_best_window=window
            current_best_dataset=key


print("\nResults\n=======")
print("Best Dist :",current_best_dist)
print("Best Dataset :",current_best_dataset)
print("\nBest Location :\n---------------\n",current_best_window[:,[3,4]])
print("\nExpected Location :\n-------------------\n",new_seq[:,[3,4]])



