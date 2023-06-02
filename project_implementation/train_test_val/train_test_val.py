import random
import shutil
import numpy as np
import h5py
import os
import uuid
from os import path as osp


class TrainTestValSplitter:

    def __init__(self,conf) -> None:
        self.conf=conf
    
    def split(self):

        no_of_dpoints = self.conf.no_of_sec_per_split * self.conf.freq

        #StoreMetaData
        csvmeta=[["OType", "Target Folder", "Source Folder", "Source Start Index", "Source End Index"]]      # OType, Target Folder, Source Folder, Source Start Index, Source End Index

        # Select a folder
        for key, meta in self.conf.hdf5datadir.items():

            if meta["indb"]:
                continue

            data_list = [osp.split(path)[1] for path in os.listdir(meta["loc"])]
            csv_out_dir = self.conf.csv_out_dir

            # Select a file
            for data in data_list:
                data_out_dir = None
                try:
                    data_path = osp.join(meta["loc"], data)
                    with h5py.File(data_path, 'r') as f:

                        # Select a type
                        for otype in [key]*meta["countperset"]:
                            target_name=str(uuid.uuid4())
                            data_out_dir = osp.join(csv_out_dir, otype,target_name )

                            if os.path.exists(data_out_dir):
                                shutil.rmtree(data_out_dir)
                            os.makedirs(data_out_dir)

                            row_indexes=None

                            for filename, loc in self.conf.preferred_files.items():
                                np_data = f.get(loc)

                                # Generate Random Sub Sequence
                                if row_indexes is None:
                                    length=len(np_data)
                                    original_list = list(range(length))
                                    start_index = random.randint(0,length - no_of_dpoints)
                                    row_indexes = original_list[start_index:start_index + no_of_dpoints]


                                if np_data != None:
                                    np_data = np_data[row_indexes]
                                    time = f.get("synced/time")[row_indexes]
                                    np_data = np.c_[time, np_data]
                                else:
                                    continue

                                np.savetxt(osp.join(data_out_dir, filename + ".txt"), np_data, delimiter=" ")

                            csvmeta.append([otype, target_name, data, row_indexes[0], row_indexes[-1]])

                        print(key,":",data,"Generated")
                except Exception as e:
                    print(e)
                    shutil.rmtree(data_out_dir)
        np.savetxt(self.conf.train_test_val_meta_file,np.array(csvmeta),delimiter=",",fmt='%s')