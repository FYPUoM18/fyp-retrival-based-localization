#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import pandas as pd
import random

class PairBatchGenerator:

    def __init__(self,conf) -> None:
        self.conf = conf
        print("Configurations Loaded ")
    
    def _load_processed_hdf5(self):
        return h5py.File(self.conf.processed_hdf5_outputdir, 'r')

    def _load_a_batch(self,processed_hdf5_file):
        batch = [] # List Of (Seq1,Seq2,Issimilarloc)
        locs = list(processed_hdf5_file.keys())

        # Take Similars
        for i in range(self.conf.batchsize//2):
            loc_1 = random.choice(locs)
            times_1 = list(processed_hdf5_file.get(loc_1).keys())
            time_1 = random.choice(times_1)
            seq_1 = processed_hdf5_file.get(loc_1).get(time_1)
            
            while True:
                loc_2 = random.choice(locs)
                
                if loc_1==loc_2:
                    times_2 = list(processed_hdf5_file.get(loc_2).keys())
                    time_2 = random.choice(times_2)
                    seq_2 = processed_hdf5_file.get(loc_2).get(time_2)
                    batch.append((seq_1,seq_2,True))
                    break
        
        # Take DisSimilars
        for i in range(self.conf.batchsize//2,self.conf.batchsize):
            loc_1 = random.choice(locs)
            times_1 = list(processed_hdf5_file.get(loc_1).keys())
            time_1 = random.choice(times_1)
            seq_1 = processed_hdf5_file.get(loc_1).get(time_1)
            
            while True:
                loc_2 = random.choice(locs)
                if loc_1!=loc_2:
                    times_2 = list(processed_hdf5_file.get(loc_2).keys())
                    time_2 = random.choice(times_2)
                    seq_2 = processed_hdf5_file.get(loc_2).get(time_2)
                    batch.append((seq_1,seq_2,False))
                    break
        
        random.shuffle(batch)

        return batch

    def _generate_batches(self,processed_hdf5_file):
        train = []
        test = []
        val = []

        # Train Batches
        for i in range(int(self.conf.no_of_total_batches*self.conf.train_ratio)):
            batch = self._load_a_batch(processed_hdf5_file)
            train.append(batch)
        
        # Test Batches
        for j in range(int(self.conf.no_of_total_batches*self.conf.test_ratio)):
            batch = self._load_a_batch(processed_hdf5_file)
            test.append(batch)

        # Val Batches
        for k in range(int(self.conf.no_of_total_batches*self.conf.val_ratio)):
            batch = self._load_a_batch(processed_hdf5_file)
            val.append(batch)
        
        print("No Of Train Batches :",len(train))
        print("No Of Test Batches :",len(test))
        print("No Of Val Batches :",len(val))
        print("Batch Size :",len(train[0]))

        return train,test,val 

    def _save_batches(self,batches,output_loc):
        with h5py.File(output_loc, 'w') as f:
            for i in range(len(batches)):
                collection = f.create_group("batch_"+str(i))
                for j in range(len(batches[i])):
                    ds = batches[i][j]
                    seq_1 = np.array(ds[0][:])
                    seq_2 = np.array(ds[1][:])
                    tag = "similar_" if ds[2] else "dissimilar_"
                    sub_colection = collection.create_group(tag+str(j))
                    sub_colection.create_dataset("1",data=seq_1)
                    sub_colection.create_dataset("2",data=seq_2)
        print("Saved Batches To",output_loc)

    
    def start(self):
        processed_hdf5_file = self._load_processed_hdf5()
        train,test,val = self._generate_batches(processed_hdf5_file)
        self._save_batches(train,self.conf.train_hdf5_outputdir)
        self._save_batches(test,self.conf.test_hdf5_outputdir)
        self._save_batches(val,self.conf.val_hdf5_outputdir)
       