# Steps
steps = [5]

# Root Dir
root_dir="C:\\Users\\mashk\\MyFiles\\Semester 8\\FYP\\code"

# Step1: Generate CSV From HDF5
hdf5datadir = {
    "db": f"{root_dir}\\nilocdata-subset\\unib",
}
csv_out_dir = f"{root_dir}\\project_implementation\\outputs\\csv_data"
preferred_files = {
    'loc':'computed/aligned_pos',
    'gyro': "synced/gyro",
    'acce': "synced/acce",
    'game_rv': "synced/game_rv"
}

# Step2: Generate Train/Test/Val From HDF5
no_of_sets_per_hdf5 = [16,2,2] # Train Test Val
freq=200 # Dpoints Per Sec
no_of_sec_per_split=30 # No fo seconds considered for a split
meta_file=f"{root_dir}\\project_implementation\\outputs\\train_test_meta.csv"

# Step3: Get RoNIN Trajectory
csvdatadir = f"{root_dir}\\project_implementation\\outputs\\csv_data"
ronin_checkpoint = f"{root_dir}\\project_implementation\\ronin_resnet\\checkpoint_gsn_latest.pt"
processed_hdf5_out_dir = f"{root_dir}\\project_implementation\\outputs\\processed_data"

# Step4 : Draw Trajectory
traj_drawing_out_dir=f"{root_dir}\\project_implementation\\outputs\\traj_visualized"

# Step5 : Make Time & Rotation Invariant
invariant_domain_output_dir = f"{root_dir}\\project_implementation\\outputs\\invariant"
segment_length=0.1
smooth_by_rounding=1

# Eval
DTWThreshold_Expected_Real_LOC_Predicted__REAL_LOC_SEQS=100