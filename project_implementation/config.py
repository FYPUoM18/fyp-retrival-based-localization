# Steps
steps = [7]

# Root Dir
root_dir = "C:\\Users\\mashk\\MyFiles\\Semester 8\\FYP\\code"

# Step1: Generate CSV From HDF5
hdf5datadir = {
    "db": {
        "loc": f"{root_dir}\\nilocdata-subset\\unib\\seen",
        "isseen": True,
        "countperset": -1,
        "indb": True
    },
    "train": {
        "loc": f"{root_dir}\\nilocdata-subset\\unib\\seen",
        "isseen": True,
        "countperset": 16,
        "indb": False
    },
    "test": {
        "loc": f"{root_dir}\\nilocdata-subset\\unib\\unseen",
        "isseen": False,
        "countperset": 2,
        "indb": False
    },
    "val": {
        "loc": f"{root_dir}\\nilocdata-subset\\unib\\unseen",
        "isseen": False,
        "countperset": 2,
        "indb": False

    }

}
csv_out_dir = f"{root_dir}\\project_implementation\\outputs\\1. csv_data"
preferred_files = {
    'loc': 'computed/aligned_pos',
    'gyro': "synced/gyro",
    'acce': "synced/acce",
    'game_rv': "synced/game_rv",
    'ronin': "computed/ronin"
}

# Step2: Generate Train/Test/Val From HDF5
freq = 200  # Dpoints Per Sec
no_of_sec_per_split = 45  # No fo seconds considered for a split
train_test_val_meta_file = f"{root_dir}\\project_implementation\\outputs\\train_test_val_meta.csv"

# Step3: Get RoNIN Trajectory
ronin_checkpoint = f"{root_dir}\\project_implementation\\ronin_resnet\\checkpoint_gsn_latest.pt"
processed_hdf5_out_dir = f"{root_dir}\\project_implementation\\outputs\\2. processed_data"

# Step4 : Draw Trajectory
traj_drawing_out_dir = f"{root_dir}\\project_implementation\\outputs\\3. traj_visualized"

# Step5 : Make Time Invariant
invariant_domain_output_dir = f"{root_dir}\\project_implementation\\outputs\\4. invariant"
segment_length = 0.1

# Step6 : Image DB Generate
image_db_loc = f"{root_dir}\\project_implementation\\outputs\\5. imageDB"
image_db_meta_file = f"{root_dir}\\project_implementation\\outputs\\image_db_meta_file.csv"
window_size = 220  # Same Length Curve For All Images : Length = 220 x 0.1 = 22
step_size = 50

# Step7 : Build KDTree
image_db_loc_kdtree = f"{root_dir}\\project_implementation\\outputs\\5. imageDB\\db"
kdtree_features_loc = f"{root_dir}\\project_implementation\\outputs\\kdtree_features.pickle"
kdtree_tags_loc = f"{root_dir}\\project_implementation\\outputs\\kdtree_tags.pickle"

# Eval
DTWThreshold_Expected_Real_LOC_Predicted__REAL_LOC_SEQS = 100
window_size = 120
