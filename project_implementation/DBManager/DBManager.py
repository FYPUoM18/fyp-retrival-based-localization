import os
import uuid
from os import path as osp

import numpy as np
from matplotlib import pyplot as plt


class DBManager:

    def __init__(self, conf):
        self.conf = conf

    def generateImageDB(self):

        fail_list = []
        csv_data = [["Image_name", "invariant folder", "invariant dataset", "start", "end"]]

        # Initiate Window Parameters
        window_size = self.conf.window_size
        step_size = self.conf.step_size
        print("Window Size", window_size)
        print("Step Size", step_size)

        # Iterate Over DB
        for folder in os.listdir(self.conf.invariant_domain_output_dir):
            folder_root = osp.join(self.conf.invariant_domain_output_dir, folder)
            for dataset in os.listdir(folder_root):
                dataset_root = osp.join(folder_root, dataset)
                csv_path = os.path.join(dataset_root, "invariant_traj.csv")

                # Load Sequence
                new_seq = np.genfromtxt(csv_path, delimiter=',')[1:, :]

                # Check Whether Minimum Length Found
                if len(new_seq) < window_size:
                    fail_list.append(csv_path)
                    print("Failed DB :", folder, dataset)
                    continue

                else:
                    print("Processing DB :", folder, dataset)

                # Building Sliding Window Parameter
                max_index = len(new_seq) - 1
                current_start_index = 0
                current_end_index = current_start_index + window_size - 1

                while current_end_index <= max_index:

                    # Initialize
                    image_name = str(uuid.uuid4())
                    sub_db_loc = osp.join(self.conf.image_db_loc, folder)
                    image_loc = osp.join(sub_db_loc, image_name + ".png")
                    if not os.path.exists(sub_db_loc):
                        os.makedirs(sub_db_loc)

                    # Maintain Meta Data
                    csv_data.append([image_name, folder, dataset, current_start_index, current_end_index])
                    # "Image_name","folder","dataset","start","end"

                    # Get A Sliding Window
                    window = new_seq[current_start_index:current_end_index + 1, :]
                    current_start_index += step_size
                    current_end_index += step_size

                    # Generate and Save Image
                    x = window[:, 1]
                    y = window[:, 2]

                    plt.plot(x, y, linestyle='-')
                    plt.axis('off')
                    plt.savefig(image_loc, dpi=40)
                    plt.clf()


        print("No of failures :",len(fail_list))
        np.savetxt(self.conf.image_db_meta_file,csv_data,delimiter=",",fmt='%s')
        print("Save DB Info")

    def buildKDTree(self):
        # Loop Only Images IN DB Use Config to More psecific
        # Extract feature using VGG16
        # Save Features In KDTree
        # Save Image Names In A List
        # Save List
        # Save KDTree

        pass