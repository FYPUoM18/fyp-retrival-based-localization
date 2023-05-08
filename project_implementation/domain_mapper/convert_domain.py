import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from os import path as osp
import os
import pickle
import uuid
import numpy as np
from multiprocessing import Pool
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from domain_mapper.TransferModel import ContrastiveModel
from os import path as osp
from PIL import Image
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
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

class DomainConverter:

    def __init__(self, conf):
        self.conf = conf

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
            L[i, 0] = max(L[i - 1, 0], D[i, 0])
        for j in range(1, m):
            L[0, j] = max(L[0, j - 1], D[0, j])
        for i in range(1, n):
            for j in range(1, m):
                L[i, j] = max(min(L[i - 1, j], L[i - 1, j - 1], L[i, j - 1]), D[i, j])
        return L[n - 1, m - 1]

    def make_time_invariant_ronin(self, data, segment_length):
        # Input coordinate array (latitude, longitude)
        # Skip Header
        coords = np.array(data[1:, [0, 1]])

        # Separate the latitude and longitude coordinates into separate 1D arrays
        lat_coords = coords[:, 0]
        lon_coords = coords[:, 1]

        # Interpolate the latitude and longitude coordinates
        interp_func_lat = interp1d(np.arange(len(lat_coords)), lat_coords, kind='cubic')
        interp_func_lon = interp1d(np.arange(len(lon_coords)), lon_coords, kind='cubic')

        # Calculate the length of the curve
        curve_length = np.sum(np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1)))

        # Divide the curve into segments of segment_length
        num_segments = int(curve_length / segment_length)
        t = np.linspace(0, len(lat_coords) - 1, num_segments)
        lat = interp_func_lat(t)
        lon = interp_func_lon(t)

        # Combine the latitude and longitude coordinates into a single array
        segments = np.column_stack((lat, lon))

        return num_segments, segments

    def make_time_invariant_ground(self, data, num_segments):
        # Input coordinate array (latitude, longitude)
        # Skip Header
        coords = np.array(data[1:, [2, 3]])

        # Separate the latitude and longitude coordinates into separate 1D arrays
        lat_coords = coords[:, 0]
        lon_coords = coords[:, 1]

        # Interpolate the latitude and longitude coordinates
        interp_func_lat = interp1d(np.arange(len(lat_coords)), lat_coords, kind='cubic')
        interp_func_lon = interp1d(np.arange(len(lon_coords)), lon_coords, kind='cubic')

        # Divide the curve into segments of segment_length
        t = np.linspace(0, len(lat_coords) - 1, num_segments)
        lat = interp_func_lat(t)
        lon = interp_func_lon(t)

        # Combine the latitude and longitude coordinates into a single array
        segments = np.column_stack((lat, lon))

        return segments

    def cal_angular_code(self, ronin):
        codes = [np.NaN]
        p1 = ronin[0]
        p2 = ronin[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        first_angle = np.arctan2(dy, dx) * 180 / np.pi
        prev_angle = first_angle

        for i in range(1, len(ronin) - 1):
            p1 = ronin[i]
            p2 = ronin[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            angle = np.arctan2(dy, dx) * 180 / np.pi
            code = angle - prev_angle
            prev_angle = angle
            codes.append(code / self.conf.smooth_by_rounding)
        codes.append(np.NaN)
        return np.array(codes)

    def make_time_invariant(self):

        folders = list(os.listdir(self.conf.traj_drawing_out_dir))

        for folder in folders:
            datasets = list(os.listdir(os.path.join(self.conf.traj_drawing_out_dir, folder)))
            for dataset in datasets:

                try:

                    # Get Required Paths
                    csv_path = os.path.join(self.conf.traj_drawing_out_dir, folder, dataset, "full_traj.csv")
                    png_path1 = os.path.join(self.conf.traj_drawing_out_dir, folder, dataset, "ground_full_traj.png")
                    png_path2 = os.path.join(self.conf.traj_drawing_out_dir, folder, dataset, "ronin_full_traj.png")
                    png_path3 = os.path.join(self.conf.traj_drawing_out_dir, folder, dataset, "traj_in_db.png")
                    data_out_dir = os.path.join(self.conf.invariant_domain_output_dir, folder, dataset)

                    # Create Folders
                    if os.path.exists(data_out_dir):
                        shutil.rmtree(data_out_dir)
                    os.makedirs(data_out_dir)

                    # Copy PNG
                    try:
                        shutil.copy(png_path1, os.path.join(data_out_dir, "ground_full_traj.png"))
                        shutil.copy(png_path2, os.path.join(data_out_dir, "ronin_full_traj.png"))
                        shutil.copy(png_path3, os.path.join(data_out_dir, "traj_in_db.png"))
                    except:
                        pass
                    # Read CSV, Get No Of Segments, Make Time Invariant
                    data = np.genfromtxt(csv_path, delimiter=',')
                    segment_length = self.conf.segment_length
                    num_segments, time_invariant_ronin = self.make_time_invariant_ronin(data, segment_length)
                    time_invariant_ground = self.make_time_invariant_ground(data, num_segments)
                    segments_ids = [id for id in range(num_segments)]

                    # Combine All
                    combined = np.column_stack((segments_ids, time_invariant_ronin, time_invariant_ground))[1:-1, :]

                    # Plot and Save Traj
                    x = combined[:, 1]
                    y = combined[:, 2]
                    img_path = os.path.join(data_out_dir, "invariant_traj.png")

                    plt.scatter(x=x, y=y, s=0.1, linewidths=0.1, c="blue", label="RoNIN")
                    plt.axis('off')
                    plt.savefig(img_path, dpi=1000)
                    plt.clf()

                    # Save CSV
                    np.savetxt(os.path.join(data_out_dir, "invariant_traj.csv"), combined, delimiter=",", fmt='%.16f',
                               comments='',
                               header=','.join(["id", "i_ronin_x", "i_ronin_y", "i_ground_x", "i_ground_y"]))
                    print("Saved Time Invariant Domain For", dataset)

                except Exception as ex:
                    # Remove Folders
                    if os.path.exists(data_out_dir):
                        shutil.rmtree(data_out_dir)
                    print("Failed !!! :", dataset)
                    print(ex)

    def select_one_random(self, csvs, window_size,adjust=False):
        array1 = random.choice(csvs)

        # Select two random 60-length sub-arrays from each array
        start1 = random.randint(0, len(array1) - window_size)
        subarray1 = array1[start1:start1 + window_size]

        if adjust:
            subarray1 = subarray1-subarray1[0]

        return subarray1
    def select_two_randoms(self,csvs,window_size):
        array1, array2 = random.sample(csvs, 2)

        # Select two random 60-length sub-arrays from each array
        start1 = random.randint(0, len(array1) - window_size)
        subarray1 = array1[start1:start1 + window_size]

        start2 = random.randint(0, len(array2) - window_size)
        subarray2 = array2[start2:start2 + window_size]

        return subarray1,subarray2
    def generatePairs(self,type,count):

        # Initiate Window Parameters
        window_size = self.conf.window_size
        pairs_count=count
        print("Window Size", window_size)
        print("Pairs", pairs_count)

        # Load All CSVs
        db_csvs=[]

        # Iterate Over DB
        for folder in os.listdir(self.conf.invariant_domain_output_dir):
            if folder != "db":
                continue
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
                if len(new_seq) >= window_size:
                    db_csvs.append(new_seq)

        # Print Result
        print("No of DB CSVs :",len(db_csvs))

        # Load All CSVs
        type_csvs = []

        # Iterate Over DB
        for folder in os.listdir(self.conf.invariant_domain_output_dir):
            if folder != type:
                continue
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
                if len(new_seq) >= window_size:
                    type_csvs.append(new_seq)

        # Print Result
        print("No of",type,"CSVs :", len(type_csvs))



        # Generate Required Folder
        folder_path = osp.join(self.conf.pair_db_loc,type)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)


        # Select 2 similar random pairs
        similar_pairs=0

        while similar_pairs<pairs_count/2:

            # Generate Random and Check Condition
            from_db = self.select_one_random(db_csvs,window_size,adjust=True)
            from_type = self.select_one_random(type_csvs, window_size, adjust=True)


            dist,_ = fastdtw(from_db[:, [3, 4]], from_type[:, [3, 4]])

            # dist,_ = fastdtw(from_db, from_type, dist=euclidean)
            # print(dist)
            if dist<=self.conf.distance_threshold_filter:
                print(dist)
                # Make Folders
                loc = osp.join(folder_path,"S-"+str(similar_pairs))
                os.makedirs(loc)

                # plt.scatter(from_db[:, 3],from_db[:, 4],color="red")
                # plt.scatter(from_type[:, 1], from_type[:, 2], color="blue")
                # plt.show()

                # Save
                # np.savetxt(loc + "\\db.csv", from_db[:, [3, 4]], delimiter=",", fmt='%s')
                # np.savetxt(loc + "\\type.csv", from_type[:, [1, 2]], delimiter=",", fmt='%s')

                x = from_type[:, 1]
                y = from_type[:, 2]
                plt.plot(x, y, linewidth=10, linestyle='-')
                plt.axis('off')
                plt.savefig(loc + "\\type.png", dpi=40)
                plt.clf()

                x = from_db[:, 1]
                y = from_db[:, 2]
                plt.plot(x, y, linewidth=10, linestyle='-')
                plt.axis('off')
                plt.savefig(loc + "\\db.png", dpi=40)
                plt.clf()

                # Increment
                if similar_pairs%10==0:
                    print("Saved Simlar Pair :",similar_pairs)
                similar_pairs += 1

        # Select 2 disimilar random pairs
        disimilar_pairs = 0

        while disimilar_pairs < pairs_count / 2:

            # Generate Random and Check Condition
            from_db = self.select_one_random(db_csvs, window_size, adjust=True)
            from_type = self.select_one_random(type_csvs, window_size, adjust=True)
            # dist,_ = fastdtw(from_db, from_type, dist=euclidean)#self.frechet_distance(from_db[:, [3, 4]], from_type[:, [1, 2]])
            # dist = self.frechet_distance(from_db[:, [3, 4]], from_type[:, [1, 2]])
            dist,_ = fastdtw(from_db[:, [3, 4]], from_type[:, [3, 4]])
            if dist > self.conf.distance_threshold_filter+20000:

                # Make Folders
                loc = osp.join(folder_path, "D-" + str(disimilar_pairs))
                os.makedirs(loc)

                # Save
                # np.savetxt(loc + "\\db.csv", from_db[:, [3, 4]], delimiter=",", fmt='%s')
                # np.savetxt(loc + "\\type.csv", from_type[:, [1, 2]], delimiter=",", fmt='%s')

                x = from_type[:, 1]
                y = from_type[:, 2]
                plt.plot(x, y, linewidth=10, linestyle='-')
                plt.axis('off')
                plt.savefig(loc + "\\type.png", dpi=40)
                plt.clf()

                x = from_db[:, 1]
                y = from_db[:, 2]
                plt.plot(x, y, linewidth=10, linestyle='-')
                plt.axis('off')
                plt.savefig(loc + "\\db.png", dpi=40)
                plt.clf()


                # Increment
                if disimilar_pairs % 10 == 0:
                    print("Saved Different Pair :", disimilar_pairs)
                disimilar_pairs += 1
