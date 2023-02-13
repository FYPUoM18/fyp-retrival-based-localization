# Steps
steps = [3]

# Step1: Generate CSV From HDF5
hdf5datadir = {
    "train": "C:\\Users\\mashk\\MyFiles\\Semester 7\\FYP\\code\\nilocdata-subset\\unib\\train",
    "val": "C:\\Users\\mashk\\MyFiles\\Semester 7\\FYP\\code\\nilocdata-subset\\unib\\val",
    "test": "C:\\Users\\mashk\\MyFiles\\Semester 7\\FYP\\code\\nilocdata-subset\\unib\\test",
}
csv_out_dir = "C:\\Users\\mashk\\MyFiles\\Semester 7\\FYP\\code\\project_implementation\\outputs\\csv_data"
preferred_files = {
    'loc':'computed/aligned_pos',
    'gyro': "synced/gyro",
    'acce': "synced/acce",
    'game_rv': "synced/game_rv"
}

# Step2: Get RoNIN Trajectory
csvdatadir = "C:\\Users\\mashk\\MyFiles\\Semester 7\\FYP\\code\\project_implementation\\outputs\\csv_data"
ronin_checkpoint = "C:\\Users\\mashk\\MyFiles\\Semester 7\\FYP\\code\\project_implementation\\ronin_resnet\\checkpoint_gsn_latest.pt"
processed_hdf5_out_dir = "C:\\Users\\mashk\\MyFiles\\Semester 7\\FYP\\code\\project_implementation\\outputs\\processed_data"

# Step3 : Draw Trajectory
window_w_h=100
traj_drawing_out_dir="C:\\Users\\mashk\\MyFiles\\Semester 7\\FYP\\code\\project_implementation\\outputs\\traj_visualized"