import os
import shutil
import sys
import h5py
import numpy as np
import scipy
from os import path as osp


class CSV_Generator:

    def __init__(self,conf) -> None:
        self.conf=conf
    
    def generate(self):
        np.set_printoptions(suppress=True)

        for key, meta in self.conf.hdf5datadir.items():

            if not meta["indb"]:
                continue

            data_list=[osp.split(path)[1] for path in os.listdir(meta["loc"])]
            csv_out_dir=self.conf.csv_out_dir

            for data in data_list:
                data_path=osp.join(meta["loc"],data)
                data_out_dir = osp.join(csv_out_dir,key,str(data).split(".")[0])

                if os.path.exists(data_out_dir):
                    shutil.rmtree(data_out_dir)
                os.makedirs(data_out_dir)

                print(data_path)
                print(data_out_dir)
                
                with h5py.File(data_path, 'r') as f:
                    for filename,loc in self.conf.preferred_files.items():
                        np_data = f.get(loc)
                        if np_data!=None:
                            np_data=np_data[:]
                            time = f.get("synced/time")[:]
                            np_data=np.c_[time,np_data]
                        else:
                            continue
                        np.savetxt(osp.join(data_out_dir,filename+".txt"),np_data,delimiter=" ")
        