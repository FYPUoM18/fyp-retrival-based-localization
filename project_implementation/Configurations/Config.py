import json


class Config:
    def __init__(self,root_dir):
        self.steps = [6,7]

        # Building Params
        self.x_lim = 60
        self.y_lim = 150

        # Root Dir
        self.root_dir = root_dir

        # Step1: Generate CSV From HDF5
        self.hdf5datadir = {
            "db": {
                "loc": f"{self.root_dir}\\nilocdata-subset\\unib\\seen",
                "isseen": True,
                "countperset": -1,
                "indb": True
            },
            "train": {
                "loc": f"{self.root_dir}\\nilocdata-subset\\unib\\seen",
                "isseen": True,
                "countperset": 2,
                "indb": False
            },
            "test": {
                "loc": f"{self.root_dir}\\nilocdata-subset\\unib\\unseen",
                "isseen": False,
                "countperset": 2,
                "indb": False
            },
            "val": {
                "loc": f"{self.root_dir}\\nilocdata-subset\\unib\\unseen",
                "isseen": False,
                "countperset": 2,
                "indb": False

            }

        }
        self.csv_out_dir = f"{self.root_dir}\\1. csv_data"
        self.preferred_files = {
            'loc': 'computed/aligned_pos',
            'gyro': "synced/gyro",
            'acce': "synced/acce",
            'game_rv': "synced/game_rv"
        }
        # Step2: Generate Train/Test/Val From HDF5
        self.freq = 200  # Dpoints Per Sec
        self.no_of_sec_per_split = 60 #45  # Windows Size
        self.train_test_val_meta_file = f"{self.root_dir}\\train_test_val_meta.csv"

        # Step3: Get RoNIN Trajectory
        self.sample_rate = self.freq
        self.ronin_checkpoint = f"{self.root_dir}\\ronin_resnet\\checkpoint_gsn_latest.pt"
        self.processed_hdf5_out_dir = f"{self.root_dir}\\2. processed_data"

        # Step4 : Draw Trajectory
        self.traj_drawing_out_dir = f"{self.root_dir}\\3. traj_visualized"

        # Step5 : Make Time Invariant
        self.invariant_domain_output_dir = f"{self.root_dir}\\4. invariant"
        self.segment_length = 0.1

        # Step6 : Image DB Generate
        self.image_db_loc = f"{self.root_dir}\\5. imageDB"
        self.image_db_meta_file = f"{self.root_dir}\\image_db_meta_file.csv"
        self.window_size = 500 #400
        self.step_size = 125 #150

        # Step7 : Build KDTree
        self.image_db_loc_kdtree = f"{self.root_dir}\\5. imageDB\\db"
        self.kdtree_features_loc = f"{self.root_dir}\\kdtree_features.pickle"
        self.kdtree_tags_loc = f"{self.root_dir}\\kdtree_tags.pickle"

        # Step8 : Eval Tune
        self.train_invariant_dir = f"{self.root_dir}\\4. invariant\\train"
        self.test_invariant_dir = f"{self.root_dir}\\4. invariant\\test"
        self.val_invariant_dir = f"{self.root_dir}\\4. invariant\\val"
        self.to_eval_dir = f"{self.root_dir}\\5. imageDB\\db"
        self.to_ates_csv = "mean.csv"
        self.to_err_csv = "err.csv"
        self.to_k_ates_csv = "k_ates.csv"
        self.no_of_candidates = 5

        # Filtering
        self.merge_threshold = 1200

    def save_config_to_json(self):
        with open(f"{self.root_dir}\\config.json", 'w') as f:
            json.dump(self.__dict__, f)

    def load_config_from_json(self, directory):
        with open(f"{directory}", 'r') as f:
            data = json.load(f)
        self.__dict__.update(data)
