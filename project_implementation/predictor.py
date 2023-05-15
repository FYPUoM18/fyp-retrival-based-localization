import copy
import os
import random
import numpy as np
import uuid
import h5py
from Configurations.Config import Config
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
        self.out_dir = "C:\\Users\\mashk\\MyFiles\\Semester 8\\FYP\\code\\project_implementation\\outputs"
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
    def combineLocs(self, layer_1, layer_2):

        window_size = self.building_data_config.window_size
        step_size = self.building_data_config.step_size
        common_points_count = window_size - step_size
        locs = []
        for node_prev in layer_1:
            for node_post in layer_2:

                # Calc Result
                node_start = node_prev[:step_size,[0,1]]
                node_end = node_post[common_points_count:,[0,1]]
                node_middle_1 = node_prev[step_size:,[0,1]]
                node_middle_2 = node_post[:common_points_count,[0,1]]
                middle_avg = (node_middle_1 + node_middle_2) / 2
                result = np.concatenate((node_start, middle_avg, node_end), axis=0)

                # Interpolate
                x = result[:, 0]
                y = result[:, 1]

                distances = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
                distances = np.insert(distances, 0, 0)
                total_length = distances[-1]
                num_points = int(total_length / self.building_data_config.segment_length) + 1
                f = interp1d(distances, x, kind='cubic')
                distances_new = np.linspace(0, total_length, num_points)
                x_new = f(distances_new)
                y_new = interp1d(distances, y, kind='cubic')(distances_new)

                segments = np.column_stack((x_new, y_new))

                # Check If Walkable
                if len(segments)<=self.building_data_config.merge_threshold:
                    locs.append(segments)



        return locs

    def filterMatchings(self, layers, expected_locs):

        path = self.new_data_config.root_dir + "\\filterings"
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        if len(layers)==1:
            print("Only One Layer Is Available. Can't Do Filterings ... ")

        else:
            plots = 2 if len(layers)-1<2 else len(layers)-1
            fig, ax = plt.subplots(1,plots )
            current_layer = 0
            next_layer = 1
            while next_layer<len(layers):
                print(f"Layer 1 : {current_layer} , Layer 2 : {next_layer}")
                ax[current_layer].scatter(expected_locs[current_layer][:, 0], expected_locs[current_layer][:, 1], color="red", s=0.1)
                ax[current_layer].scatter(expected_locs[next_layer][:, 0], expected_locs[next_layer][:, 1], color="red", s=0.1)
                ax[current_layer].set_xlim([0, self.building_data_config.x_lim])
                ax[current_layer].set_ylim([0, self.building_data_config.y_lim])

                combined_filtered_locs = self.combineLocs(layers[current_layer],layers[next_layer])
                colors = [mcolors.to_rgb(np.random.rand(3)) for _ in range(len(combined_filtered_locs))]
                for loc in combined_filtered_locs:
                    ax[current_layer].scatter(loc[:, 0], loc[:, 1], color=random.choice(colors), s=1)



                current_layer+=1
                next_layer+=1

            fig.savefig(path + "\\" + str(uuid.uuid4()))
            plt.clf()


samples = 10
root_dir = "C:\\Users\\mashk\\MyFiles\\Semester 8\\FYP\\code\\project_implementation\\outputs"
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
                    no_of_dpoints = 90 * 200
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

        predictor = Predictor(outname,"building_unib",200,90)
        predictor.generate_config()
        predictor.getRoNINTrajectory()
        predictor.visualizeTrajectory()
        predictor.makeTimeInvariant()
        predictor.generateImageDB()
        layers,expected_locs = predictor.findMatchings()
        predictor.filterMatchings(layers,expected_locs)

    except:
        pass
