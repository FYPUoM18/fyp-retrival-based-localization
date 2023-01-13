#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# main
steps=[6]

# Preprocessing Configs
hdf5datadir = "C:\\Users\\mashk\\MyFiles\\Semester 7\\FYP\\code\\nilocdata-subset\\unib"  # Directory contains hdf5 files
processed_hdf5_outputdir = "output\\processed_data.hdf5" # Processed HDF5 Output Directory
locrounding = 0 # No of decimals to round the locations
timerounding = 0 # No of decimals to round the time
freq_threshold = 80 # Drop All Data Below This Threshold, Clip All Data Above Threshold
no_of_attributes = 6 # No Of Fields


# Train/Test/Val Pair Batch Generator Configs
batchsize = 72
no_of_total_batches = 100
train_pair_ratio = 0.80
val_pair_ratio = 0.15
test_pair_ratio = 0.05
train_pair_hdf5_outputdir = "output\\train_pair_data.hdf5"
test_pair_hdf5_outputdir = "output\\test_pair_data.hdf5"
val_pair_hdf5_outputdir = "output\\val_pair_data.hdf5"

# Contrastive Model
input_shape = (no_of_total_batches,freq_threshold,no_of_attributes)
no_of_epochs = 8
contrastive_training_graph_output = "output\\contrastive_model_state.pdf"
contrastive_model_state_save_path = "output\\contrastive_model_state.pt"

# Tagged Train Test Batch Splitter
train_tagged_ratio = 0.80
train_tagged_dat_outputdir = "output\\train_tagged.dat"
test_tagged_dat_outputdir = "output\\test_tagged.dat"

# Embedding Space Generator
nn_model_save_path = "output\\motion-to-motion-mapper.model"
loc_tags_save_path = "output\\loc-tags.txt"

# Embedding Space Validator
n_neighbors_to_test=10