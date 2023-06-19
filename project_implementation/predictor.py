import copy
import os
import random
import math
import numpy as np
import uuid
import h5py
import csv
from Configurations.Config import Config
from scipy.spatial.distance import cdist
from compile_dataset.compiler import Compiler
from traj_visualizer.traj_visualizer import TrajVisualizer
from domain_mapper.convert_domain import DomainConverter
from DBManager.DBManager import DBManager
from DBManager.eval import Evaluator
from scipy.spatial.distance import euclidean
from os import path as osp
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors



class Predictor:

    def __init__(self,new_data_name,building_name,freq = 200,no_of_sec_per_split = 45):
        self.out_dir = "C:\\Users\\musab\\OneDrive\\Desktop\\projects\\fyp\\fyp-retrival-based-localization\\project_implementation\\outputs"
        self.new_data_root_dir = f"{self.out_dir}\\predictions\\{new_data_name}"
        self.building_root_dir = f"{self.out_dir}\\{building_name}"
        self.new_data_config = None
        self.building_data_config = None
        self.freq = freq
        self.no_of_sec_per_split = no_of_sec_per_split

    def generate_config(self):

        building_config = Config(root_dir="")
        building_config.load_config_from_json(f"{self.building_root_dir}\\config.json")

        new_data_config  = Config(root_dir=self.new_data_root_dir)
        new_data_config.x_lim = building_config.x_lim
        new_data_config.y_lim = building_config.y_lim
        new_data_config.hdf5datadir = {}
        new_data_config.freq = self.freq
        new_data_config.sample_rate = self.freq
        new_data_config.no_of_sec_per_split = self.no_of_sec_per_split
        new_data_config.segment_length = building_config.segment_length
        new_data_config.window_size = building_config.window_size
        new_data_config.step_size = building_config.step_size
        new_data_config.no_of_candidates = building_config.no_of_candidates
        new_data_config.ronin_checkpoint = building_config.ronin_checkpoint
        new_data_config.image_db_loc_kdtree = building_config.image_db_loc_kdtree
        new_data_config.kdtree_features_loc = building_config.kdtree_features_loc
        new_data_config.kdtree_tags_loc = building_config.kdtree_tags_loc
        new_data_config.merge_threshold = building_config.merge_threshold

        new_data_config.save_config_to_json()

        self.new_data_config=new_data_config
        self.building_data_config=building_config

    def getRoNINTrajectory(self):
        config = self.new_data_config
        data_compiler = Compiler(config)
        data_compiler.compile()

    def visualizeTrajectory(self):
        config = self.new_data_config
        csvmeta = [["OType", "Target Folder", "Source Folder", "Source Start Index", "Source End Index"]]
        np.savetxt(config.train_test_val_meta_file, np.array(csvmeta), delimiter=",", fmt='%s')
        traj_visualizer = TrajVisualizer(config)
        traj_visualizer.drawRoNINTraj()

    def makeTimeInvariant(self):
        config = self.new_data_config
        domain_convertor = DomainConverter(config)
        domain_convertor.make_time_invariant()

    def generateImageDB(self):
        config = self.new_data_config
        generate_imagedb = DBManager(config)
        generate_imagedb.generateImageDB()

    def findMatchings(self):

        new_data_config = self.new_data_config
        building_data_config=self.building_data_config

        evaluator = Evaluator()
        layers,expected_locs = evaluator.evaluate(new_data_config,building_data_config)
        return layers,expected_locs

    def getDTW(self,nplist1, nplist2):
        distance, path = fastdtw(nplist1, nplist2, dist=euclidean)
        return distance

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

    def calculate_ate(self,expected_locs, predicted_real_loc):
        # Calculate Euclidean distances between corresponding points
        distances = np.linalg.norm(expected_locs - predicted_real_loc, axis=1)

        # Calculate ATE
        ate = np.mean(distances)

        return ate
    def combineLocs(self, layer_1, layer_2):

        window_size = self.building_data_config.window_size
        step_size = self.building_data_config.step_size
        common_points_count = window_size - step_size
        locs = []

        min_dist = float('inf')
        selected_node_prev = None
        selected_node_post = None

        for node_prev in layer_1:
            for node_post in layer_2:

                for i in (1,-1):
                    for j in (1,-1):

                        node_prev_updated = node_prev[::i]
                        node_post_updated = node_post[::j]

                        node_middle_1 = node_prev_updated[step_size:, [0, 1]]
                        node_middle_2 = node_post_updated[:common_points_count, [0, 1]]

                        # dist = math.sqrt((node_prev_updated[-1, 0] - node_post_updated[0, 0]) ** 2 + (
                        #             node_prev_updated[-1, 1] - node_post_updated[0, 1]) ** 2)
                        dist = self.getDTW(node_middle_1,node_middle_2)
                        if dist<min_dist:
                            min_dist=dist
                            selected_node_prev=node_prev_updated
                            selected_node_post=node_post_updated

        return min_dist, selected_node_prev, selected_node_post, selected_node_post[-1]

    def euclidean(self, row_x_y_1, row_x_y_2):
        dist = math.sqrt((row_x_y_1[0] - row_x_y_2[0]) ** 2 + (row_x_y_1[1] - row_x_y_2[1]) ** 2)
        return dist

    def filterMatchings(self, layers, expected_locs):

        filtering_path = self.new_data_config.root_dir + "\\filterings"
        if os.path.exists(filtering_path):
            shutil.rmtree(filtering_path)
        os.mkdir(filtering_path)

        for i in range(len(layers)):

            initial_loc = expected_locs[i][0]
            expected_end_loc = expected_locs[i][-1]
            possible_paths = layers[i]

            predicted_end_loc = None
            min_dist = float('inf')
            best_path = None

            for path in possible_paths:
                # print(path[0])
                # print(path[-1])
                # print(initial_loc)
                dist_1 = self.euclidean(initial_loc,path[0])
                dist_2 = self.euclidean(initial_loc, path[-1])
                # print(dist_1)
                # print(dist_2)

                if dist_1<=min_dist:
                    min_dist=dist_1
                    best_path=path
                    predicted_end_loc = path[-1]

                if dist_2<=min_dist:
                    min_dist=dist_2
                    best_path=path
                    predicted_end_loc = path[0]

            err = self.euclidean(predicted_end_loc,expected_end_loc)
            name = str(uuid.uuid4())

            plt.xlim([0, self.building_data_config.x_lim])
            plt.ylim([0, self.building_data_config.y_lim])

            plt.scatter(best_path[:, 0], best_path[:, 1], color='blue', s=1)
            plt.scatter(expected_locs[i][:, 0], expected_locs[i][:, 1], color='red', s=1)

            plt.scatter(initial_loc[0], initial_loc[1], color='yellow', s=20)
            plt.scatter(expected_end_loc[0], expected_end_loc[1], color='black', s=20)
            plt.scatter(predicted_end_loc[0], predicted_end_loc[1], color='green', s=20)

            with open(self.new_data_config.to_err_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([name, err])

            plt.savefig(filtering_path + "\\" + name)
            plt.clf()


samples = 10
root_dir = "C:\\Users\\musab\\OneDrive\\Desktop\\projects\\fyp\\fyp-retrival-based-localization\\project_implementation\\outputs"
data_dir = f"{root_dir}\\building_unib\\nilocdata-subset\\unib\\unseen"
prediction_dir = f"{root_dir}\\predictions"
files = os.listdir(data_dir)
preferred_files = {
            'loc': 'computed/aligned_pos',
            'gyro': "synced/gyro",
            'acce': "synced/acce",
            'game_rv': "synced/game_rv"
        }

for sample in range(samples):

    try:
        plt.rcParams.update(plt.rcParamsDefault)
        plt.close('all')

        # Get Random File
        file = random.choice(files)

        # Generate Directory
        outname = uuid.uuid4()
        freq = 200
        secs = 150
        csv_dir = f"{prediction_dir}\\{outname}\\1. csv_data\\db\\{uuid.uuid4()}"
        os.makedirs(csv_dir)

        # Open File
        with h5py.File(f"{data_dir}\\{file}", 'r') as f:

            # Sub Windows
            sub_window =  None

            # Get Fields
            for filename, loc in preferred_files.items():
                np_data = f.get(loc)

                # Generate Random Sub Window
                if sub_window is None:
                    length = len(np_data)
                    no_of_dpoints = freq*secs
                    original_list = list(range(length))
                    start_index = random.randint(0, length - no_of_dpoints)
                    sub_window = original_list[start_index:start_index + no_of_dpoints]

                if np_data != None:
                    np_data = np_data[sub_window]
                    time = f.get("synced/time")[sub_window]
                    np_data = np.c_[time, np_data]
                else:
                    continue
                np.savetxt(osp.join(csv_dir, filename + ".txt"), np_data, delimiter=" ")

        predictor = Predictor(outname,"building_unib",freq,secs)
        predictor.generate_config()
        predictor.getRoNINTrajectory()
        predictor.visualizeTrajectory()
        predictor.makeTimeInvariant()
        predictor.generateImageDB()
        layers,expected_locs = predictor.findMatchings()
        predictor.filterMatchings(layers,expected_locs)

    except Exception as ex:
        print(ex)
