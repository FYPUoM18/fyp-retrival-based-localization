import h5py
import argparse
import json
import os
import uuid
import shutil
import sys
import matplotlib.pyplot as plt
import collections
import numpy as np
from os import path as osp


class TrajVisualizer:

    def __init__(self,conf) -> None:
        self.conf=conf

    def get_all_hdf5_list(self):
        list_of_hdf5s=[]
        folder_types=[osp.split(path)[1] for path in os.listdir(self.conf.processed_hdf5_out_dir)]
        for folder in folder_types:
            for file in os.listdir(osp.join(self.conf.processed_hdf5_out_dir,folder)):
                file_loc=osp.join(self.conf.processed_hdf5_out_dir,folder,file,"data.hdf5")
                list_of_hdf5s.append((folder,file_loc))
        return list_of_hdf5s
    
    def get_time_slices(self,hdf5):

        time_slices=[]
        
        with h5py.File(hdf5[1], 'r') as f:
            # Get Sub Times
            times=list(f.get("synced/time")[:])
            marker_start=times[0]
            marker_end=marker_start+self.conf.time_window_size
                            
            while True:
                current_slice=[]
                for i in range(len(times)):
                    if times[i]>=marker_start and times[i]<=marker_end:
                        current_slice.append(i)

                time_slices.append(current_slice)
                marker_start+=self.conf.time_stride
                marker_end+=self.conf.time_stride
                if marker_end>times[-1]:
                    break
        return time_slices
        

    def drawRoNINTraj(self):    

        if os.path.exists(self.conf.traj_drawing_out_dir):
                shutil.rmtree(self.conf.traj_drawing_out_dir)

        list_of_hdf5s=self.get_all_hdf5_list()

        for hdf5 in list_of_hdf5s:
            with h5py.File(hdf5[1], 'r') as f:

                # Get Time Slices
                time_slices=self.get_time_slices(hdf5)

                # Get Mean Locs For Those Time Slices
                if f.get("synced/loc")!=None:
                    locs_for_slices=[np.take(f.get("synced/loc")[:], time_slice, axis=0) for time_slice in time_slices]
                    mean_locs_for_slices= [np.rint(np.median(loc_slice,axis=0)) for loc_slice in locs_for_slices]   
                
                # Get Trajectory For Slices
                trajs_for_slices=[np.take(f.get("computed/ronin")[:], time_slice, axis=0) for time_slice in time_slices]
                
                # Centerize Traj
                centered_trajs_for_slices = [ traj - traj[0,:] for traj in trajs_for_slices ]

                
                for i in range(len(centered_trajs_for_slices)):
                    
                    # Name Directory
                    if f.get("synced/loc")!=None:
                        folder_name="loc_"+str(int(mean_locs_for_slices[i][0]))+"_"+str(int(mean_locs_for_slices[i][1]))
                    else:
                        folder_name="loc_no_no"
                    save_path=osp.join(self.conf.traj_drawing_out_dir,hdf5[0])
                    img_path=osp.join(self.conf.traj_drawing_out_dir,hdf5[0],folder_name+"-"+str(uuid.uuid4())+".png")

                    # Create Directory
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    x_one_side_range=self.conf.x_one_side_range
                    y_one_side_range=self.conf.y_one_side_range

                    # Plot and Save
                    x=centered_trajs_for_slices[i][:,0]
                    y=centered_trajs_for_slices[i][:,1]
                    
                    assert(min(x)>=-x_one_side_range and max(x)<=x_one_side_range)
                    assert(min(y)>=-y_one_side_range and max(y)<=y_one_side_range)
                    
                    plt.scatter(x=x, y=y,s=0.01,linewidths=0.05,c="black")
                    plt.axis('off')
                    plt.xlim([-x_one_side_range,x_one_side_range])
                    plt.ylim([-y_one_side_range,y_one_side_range])
                    plt.savefig(img_path,dpi=300)
                    plt.clf()
                    print("Saved",img_path)


        # # Find Min Max For The Specific Building
        # loc_0=[]
        # loc_1=[]
        # for hdf5 in list_of_hdf5s:
        #     with h5py.File(hdf5[1], 'r') as f:
        #         # Get Traj
        #         traj_arr=f.get("computed/ronin")[:]
        #         loc_0.extend(list(traj_arr[:,0]))
        #         loc_1.extend(list(traj_arr[:,1]))
            

        # for hdf5 in list_of_hdf5s:
        #     with h5py.File(hdf5[1], 'r') as f:
        #         traj_arr=f.get("computed/ronin")[:]
        #         time = f.get("synced/time")[:]
        #         plt.scatter(x=traj_arr[:,0], y=traj_arr[:,1],s=0.1,c="black")
        #         plt.axis("off")
        #         plt.xlim([min(loc_0),max(loc_0)])
        #         plt.ylim([min(loc_1),max(loc_1)])
        #         os.makedirs(save_loc)
        #         plt.savefig(osp.join(save_loc,"traj.png"))