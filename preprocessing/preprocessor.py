#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import config
import os
import h5py

class Preporcessor:
    
    """
    Initialize and load the paramters required for preprocessing
    
    @param conf: python configuration file
    @return: None 
    
    """
    def __init__(self,conf):
        
        self.conf = conf
        print("Configurations Loaded")
        
        
        
        
    """
    start preprocessing
    
    @return: grouped files
    
    """
        
    def start(self):
        
        hdf5list = self.__listhdf5()
        print("No of HDF5 files found :",len(hdf5list))
        
        for hdf5file in hdf5list:
        with h5py.File(os.path.join(self.conf.hdf5datadir, hdf5file), 'r') as f:
            
        
    
    """
    List Hdf5 files in a the configured directory
    
    @return: return a list of hdf5 files in the configured directory
    
    """
    def __listhdf5(self):
        
        hdf5list = []        
        for file in os.listdir(self.conf.hdf5datadir):
            if file.endswith(".hdf5"):
                hdf5list.append(file)
        return hdf5list
        

if __name__ == "__main__":
    preprocessor = Preporcessor(config)
    preprocessor.start()
        
        
    