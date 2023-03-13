# Steps
steps = [4]

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

# Step2: Get RoNIN Trajectory
csvdatadir = f"{root_dir}\\project_implementation\\outputs\\csv_data"
ronin_checkpoint = f"{root_dir}\\project_implementation\\ronin_resnet\\checkpoint_gsn_latest.pt"
processed_hdf5_out_dir = f"{root_dir}\\project_implementation\\outputs\\processed_data"

# Step3 : Draw Trajectory
traj_drawing_out_dir=f"{root_dir}\\project_implementation\\outputs\\traj_visualized"

# Step4 : Make Time & Rotation Invariant
invariant_domain_output_dir = f"{root_dir}\\project_implementation\\outputs\\invariant"
segment_length=1
smooth_by_rounding=1