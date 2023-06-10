import os
import shutil
import sys
import h5py
import numpy as np
import scipy
from os import path as osp
import matplotlib.pyplot as plt
import pandas as pd
import csv
import random

class FingerPrintManager:

    def __init__(self,conf):
        self.conf=conf

    def build(self):
        np.set_printoptions(suppress=True)
        fingerprints=[]

        try:
            os.remove(f"{self.conf.root_dir}\\fingerprint.csv")
            print("File removed successfully.")
        except OSError as e:
            print(f"Error occurred while removing the file: {e}")


        for key, meta in self.conf.hdf5datadir.items():

            if not meta["indb"]:
                continue

            data_list = [osp.split(path)[1] for path in os.listdir(meta["loc"])]
            # csv_out_dir = self.conf.csv_out_dir

            for data in data_list:
                data_path = osp.join(meta["loc"], data)
                # data_out_dir = osp.join(csv_out_dir, key, str(data).split(".")[0])

                # if os.path.exists(data_out_dir):
                #     shutil.rmtree(data_out_dir)
                # os.makedirs(data_out_dir)

                # print(data_path)
                # print(data_out_dir)

                with h5py.File(data_path, 'r') as f:
                    print("Processing",data_path)

                    scans = pd.DataFrame(f.get("raw/w, 0ifi_scans"))
                    values = pd.DataFrame(f.get("raw/wifi_values"))
                    address = pd.DataFrame(f.get("raw/wifi_address"))

                    time = pd.DataFrame(f.get("synced/time"))
                    loc = pd.DataFrame(f.get("computed/aligned_pos"))


                    for i in range(1,len(scans)):

                        selected_scan =  values.loc[values[0] == i]
                        highest_signal_idx = selected_scan[2].idxmax()

                        target_time = values.loc[highest_signal_idx,1]
                        bssid = address.loc[highest_signal_idx,0].decode()

                        time_diff = (time[0] - target_time).abs()
                        nearest_time_index = time_diff.idxmin()
                        nearest_time_diff = time_diff.loc[nearest_time_index]

                        if nearest_time_diff<=0.5:
                            wifi_loc = loc.loc[nearest_time_index]
                            fingerprints.append([":".join(bssid.split(":")[:-1]),wifi_loc[0],wifi_loc[1]])

        with open(f"{self.conf.root_dir}\\fingerprint.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(fingerprints)


        bssid_set=set([row[0] for row in fingerprints])
        print(bssid_set)
        print("LENGTH",len(bssid_set))

        print("FingerPrint CSV file created successfully.")


        # DRAWING

        labels = list(set(row[0] for row in fingerprints))
        ucolors = [plt.cm.tab20(i) for i in np.linspace(0, 1, len(labels))]
        color_map = {label: ucolors[i] for i, label in enumerate(labels)}

        print("Started Plotting")

        for row in random.sample(fingerprints, k=1000):
            label, x, y = row
            color = color_map[label]
            plt.scatter(x, y, color=color,s=1)

        print("Finished Plotting")
        # Show the plot
        plt.show()