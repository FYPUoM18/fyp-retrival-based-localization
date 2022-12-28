#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import pandas as pd


class PreProcessor:

    def __init__(self, conf):
        """
        Initialize and load the paramters required for preprocessing
        :param conf: python configuration file
        """
        self.conf = conf
        print("Configurations Loaded")

    def __listhdf5(self):
        """
        List Hdf5 files in a the configured directory
        :return: return a list of hdf5 files in the configured directory
        """
        hdf5list = []
        for file in os.listdir(self.conf.hdf5datadir):
            if file.endswith(".hdf5"):
                hdf5list.append(file)
        print("No of HDF5 files found :", len(hdf5list),"\n")
        return hdf5list

    def __select_and_combine(self, hdf5list):
        """
        Select required columns and combine all them together into a np array
        :param hdf5list: name list of hdf5 files in the configured directory
        :return: pandas dataframe
        """
        # Final np array
        alldata = np.empty(shape=(0, 9))

        # Iterate Over HDF5 Files
        for hdf5file in hdf5list:

            # Read Data
            with h5py.File(os.path.join(self.conf.hdf5datadir, hdf5file), 'r') as f:

                loc = np.copy(f['computed/aligned_pos'])
                time = np.copy(f['synced/time'])
                acce = np.copy(f['synced/acce'])
                gyro = np.copy(f['synced/gyro'])

            # Check Data
            if not (len(loc) == len(time) == len(acce) == len(gyro)):
                print("Skipping", hdf5file)
                continue

            # Reshape Data
            time = time.reshape(len(time), 1)

            # Rounding
            loc = np.round(loc, decimals=self.conf.locrounding)
            time = np.round(time, decimals=self.conf.timerounding)

            # Concatenate filtered data
            filtered = np.concatenate((loc, time, acce, gyro), axis=1)

            # Add to all data
            alldata = np.concatenate((alldata, filtered), axis=0)

            print("Added", len(filtered), "data from", hdf5file)

        # Check Shape
        print("\nShape of Final Combined Data", np.shape(alldata),"\n")
        return pd.DataFrame(alldata, columns=["loc_1","loc_2","time","acce_1","acce_2","acce_3","gyro_1","gyro_2","gyro_3"])


    def __groupdata(self, combined_data):
        """
        Get a combined np array, group/sort data and return it
        :param combined_data: combined pandas dataframe [Loc(2),time,acce(3),gyro(3)]
        :return: grouped/sorted DataFrame
        """
        grouped_data = combined_data.groupby(["loc_1","loc_2","time"])
        return grouped_data

    def start(self):
        """
        start preprocessing and output hdf5 file
        """
        # Read File Names
        hdf5list = self.__listhdf5()

        # Select and Combine
        combined_data = self.__select_and_combine(hdf5list)

        # Group Data By Loc
        grouped_data = self.__groupdata(combined_data)

        # Saving
        # Open New HDF5
        
        with h5py.File(self.conf.processed_hdf5_outputdir, 'w') as f:

            # Print no of groups
            grp_len = len(grouped_data)
            print("No of Groups : ",grp_len)

            # Iterate Over Groups
            i=0
            no_of_locs = 0
            for group_name, df in grouped_data:
                # Create Names
                hdf5_collection_id = "loc_"+str(df["loc_1"].iloc[0])+"_"+str(df["loc_2"].iloc[0])
                hdf5_dataset_id = "time_" + str(df["time"].iloc[0])

                # Get Existing Collections
                current_collections = [col[0] for col in list(f.items())]

                # If Exists Get That Collection
                if current_collections.count(hdf5_collection_id) > 0:
                    collection = f.get(hdf5_collection_id)

                # Else Create New Collection
                else:
                    collection = f.create_group(hdf5_collection_id)
                    no_of_locs=no_of_locs+1

                # Add Data
                collection.create_dataset(hdf5_dataset_id, data=df[["acce_1","acce_2","acce_3","gyro_1","gyro_2","gyro_3"]].to_numpy())
                print("Added : ",str(i+1)+"/"+str(grp_len))
                i=i+1

            print("\nNo of Unique Locations :",no_of_locs)

        