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
from os import path as osp
import matplotlib.pyplot as plt



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
        evaluator.evaluate(new_data_config,building_data_config)


samples = 5
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
    predictor.findMatchings()
