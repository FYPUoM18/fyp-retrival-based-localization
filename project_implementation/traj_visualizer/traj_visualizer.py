import os
import shutil
import csv
import h5py
import matplotlib.pyplot as plt
import numpy as np

from os import path as osp



class TrajVisualizer:

    def __init__(self, conf) -> None:
        self.conf = conf

    def get_all_hdf5_list(self):
        list_of_hdf5s = []
        folder_types = [osp.split(path)[1] for path in os.listdir(self.conf.processed_hdf5_out_dir)]
        for folder in folder_types:
            for file in os.listdir(osp.join(self.conf.processed_hdf5_out_dir, folder)):
                file_loc = osp.join(self.conf.processed_hdf5_out_dir, folder, file, "data.hdf5")
                list_of_hdf5s.append((folder,file,file_loc))
        return list_of_hdf5s

    def drawTrajAll(self, hdf5):

        with h5py.File(hdf5[2], 'r') as f:

            # Define and Create Directory/Path
            save_path = osp.join(self.conf.traj_drawing_out_dir, hdf5[0], hdf5[1])
            ground_img_path = osp.join(self.conf.traj_drawing_out_dir, hdf5[0], hdf5[1], "ground_full_traj.png")
            ronin_img_path = osp.join(self.conf.traj_drawing_out_dir, hdf5[0], hdf5[1], "ronin_full_traj.png")
            csv_path = osp.join(self.conf.traj_drawing_out_dir, hdf5[0], hdf5[1], "full_traj.csv")

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Get Trajectory
            ronin = f.get("computed/ronin")[:]
            real = f.get("synced/loc")[:]


            # Plot and Save
            x = ronin[:, 0]
            y = ronin[:, 1]
            real_x = real[:,0] - real[0,0]
            real_y = real[:,1] - real[0,1]

            plt.scatter(x=x, y=y, s=0.01, linewidths=0.01, c="blue", label="RoNIN")
            plt.axis('off')
            plt.savefig(ronin_img_path, dpi=100)
            plt.clf()
            print("Saved", ronin_img_path)

            plt.scatter(x=real_x, y=real_y, s=0.01, linewidths=0.01, c="red",label="Ground Truth")
            plt.axis('off')
            plt.savefig(ground_img_path, dpi=100)
            plt.clf()
            print("Saved", ground_img_path)

            # Create CSV
            if f.get("synced/loc") != None:
                real_locs = f.get("synced/loc")[:]
                csv_data = np.concatenate((ronin,real_locs), axis=1)
            else:
                csv_data = np.array(ronin)

            np.savetxt(csv_path, csv_data, delimiter=",",fmt='%.16f',comments='',header=','.join(["traj_ronin_x","traj_ronin_y","traj_real_x","traj_real_y"]))


    def drawRoNINTraj(self):
        if os.path.exists(self.conf.traj_drawing_out_dir):
            shutil.rmtree(self.conf.traj_drawing_out_dir)

        list_of_hdf5s = self.get_all_hdf5_list()

        # Draw DB Traj
        with open(self.conf.train_test_val_meta_file, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for i, (otype1,target,source,start,end) in enumerate(reader):
                for otype2,file,loc in list_of_hdf5s:
                    if source.split(".")[0]==file:
                        with h5py.File(loc, 'r') as f:
                            ronin = f.get("computed/ronin")[range(int(start),int(end))]

                            savepath = osp.join(self.conf.traj_drawing_out_dir, otype1, target)
                            if not os.path.exists(savepath):
                                os.makedirs(savepath)

                            plt.scatter(x=ronin[:,0], y=ronin[:,1], s=0.01, linewidths=0.01, c="blue", label="RoNIN")
                            # plt.axis('off')
                            plt.savefig(osp.join(savepath,"traj_in_db.png"), dpi=100)
                            plt.clf()
                            print("Added Traj In DB For :",target,otype2,file)


        for hdf5 in list_of_hdf5s:
            self.drawTrajAll(hdf5)
