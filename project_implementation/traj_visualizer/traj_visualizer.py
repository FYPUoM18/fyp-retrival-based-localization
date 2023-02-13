import h5py
import argparse
import json
import os
import shutil
import sys
import matplotlib.pyplot as plt
from os import path as osp

class TrajVisualizer:

    def __init__(self,conf) -> None:
        self.conf=conf

    def drawRoNINTraj(self):    

        list_of_hdf5s=[]
        folder_types=[osp.split(path)[1] for path in os.listdir(self.conf.processed_hdf5_out_dir)]
        for folder in folder_types:
            for file in os.listdir(osp.join(self.conf.processed_hdf5_out_dir,folder)):
                file_loc=osp.join(self.conf.processed_hdf5_out_dir,folder,file,"data.hdf5")
                save_loc=osp.join(self.conf.traj_drawing_out_dir,folder,file)
                list_of_hdf5s.append((folder,file_loc,save_loc))
        

        for hdf5 in list_of_hdf5s:
            with h5py.File(hdf5[1], 'r') as f:
                traj_arr=f.get("computed/ronin")[:]
                
                print(traj_arr)
                plt.scatter(x=traj_arr[:,0], y=traj_arr[:,1])
                plt.show()