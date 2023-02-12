# Steps
steps = [2]

# Step1: Generate CSV From HDF5
hdf5traindatadir = "C:\\Users\\mashk\\MyFiles\\Semester 7\\FYP\\code\\nilocdata-subset\\unib\\train"
csv_out_dir = "C:\\Users\\mashk\\MyFiles\\Semester 7\\FYP\\code\\project_implementation\\outputs\\csv_data"
preferred_files = {
    'gyro': "synced/gyro", 
    #'gyro_uncalib': "synced/gyro_uncalib", 
    #'gyro_bias': "synced/gyro_bias",
    'acce': "synced/acce", 
    #'magnet': "synced/magnet", 
    'game_rv': "synced/game_rv", 
    #'linacce': "synced/linacce", 
    #'gravity': "synced/gravity", 
    #'step': "synced/step",
    #'rv': "synced/rv",
    #'pressure': "synced/pressure"
}

# Step2: Get RoNIN Trajectory
csvdatadir = "C:\\Users\\mashk\\MyFiles\\Semester 7\\FYP\\code\\project_implementation\\outputs\\csv_data"
ronin_checkpoint = "C:\\Users\\mashk\\MyFiles\\Semester 7\\FYP\\code\\project_implementation\\ronin_resnet\\checkpoint_gsn_latest.pt"
out_dir = "C:\\Users\\mashk\\MyFiles\\Semester 7\\FYP\\code\\project_implementation\\outputs\\processed_data"
