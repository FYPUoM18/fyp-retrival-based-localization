#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Preprocessing Configs
hdf5datadir = "C:\\Users\\mashk\\MyFiles\\Semester 7\\FYP\\code\\nilocdata-subset\\unib"  # Directory contains hdf5 files
processed_hdf5_outputdir = "output\\processed_data.hdf5" # Processed HDF5 Output Directory
locrounding = 0 # No of decimals to round the locations
timerounding = 0 # No of decimals to round the time
freq_threshold = 80 # Drop All Data Below This Threshold, Clip All Data Above Threshold
no_of_attributes = 6 # No Of Fields


# Train/Test/Val Batch Generator Configs
batchsize = 72
no_of_total_batches = 100
train_ratio = 0.80
val_ratio = 0.15
test_ratio = 0.05
train_hdf5_outputdir = "output\\train_data.hdf5"
test_hdf5_outputdir = "output\\test_data.hdf5"
val_hdf5_outputdir = "output\\val_data.hdf5"

# Contrastive Model
input_shape = (no_of_total_batches,freq_threshold,no_of_attributes)
no_of_epochs = 8
training_graph_output = "output\\training_graph.pdf"
model_state_save_path = "output\\model_state.pt"
