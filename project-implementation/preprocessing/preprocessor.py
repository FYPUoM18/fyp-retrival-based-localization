#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np


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
        print("No of HDF5 files found :", len(hdf5list))
        return hdf5list

    def __select_and_combine(self, hdf5list):
        """
        Select required columns and combine all them together into a np array
        :param hdf5list: name list of hdf5 files in the configured directory
        :return: np array
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
        print("Shape of Final Combined Data", np.shape(alldata))
        return alldata

    def __groupdata(self, combined_data):
        """
        Get a combined np array, group/sort data and return it
        :param combined_data: combined np array
        :return: grouped/sorted np array
        """
        
        
        return combined_data

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
