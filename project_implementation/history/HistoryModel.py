import os
import random

import numpy as np
import matplotlib.pyplot as plt
import shutil
import pickle
import csv
from os import path as osp
from PIL import Image
from DBManager.DBManager import DBManager
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


class HistoryModel:
    def __init__(self,conf):
        self.conf=conf

    def list_contains_np_array(self,lst, target_arr):
        """
        Checks whether a Python list contains a NumPy array.

        Parameters:
        lst (list): A list of NumPy arrays to search through.
        target_arr (numpy.ndarray): The NumPy array to search for in the list.

        Returns:
        bool: True if the target array is present in the list, False otherwise.
        """
        for arr in lst:
            if np.array_equal(arr, target_arr):
                return True
        return False

    def is_overlaps(self, P, Q):

        distance = self.frechet_distance(P,Q)
        return distance<self.conf.filter_distance_threshold

    def frechet_distance(self,P, Q):
        """
        Compute the Fréchet distance between two trajectories P and Q.
        """
        n = len(P)
        m = len(Q)
        if n != m:
            raise ValueError("Trajectories must have the same length")
        D = cdist(P, Q)
        L = np.zeros((n, m))
        L[0, 0] = D[0, 0]
        for i in range(1, n):
            L[i, 0] = max(L[i - 1, 0], D[i, 0])
        for j in range(1, m):
            L[0, j] = max(L[0, j - 1], D[0, j])
        for i in range(1, n):
            for j in range(1, m):
                L[i, j] = max(min(L[i - 1, j], L[i - 1, j - 1], L[i, j - 1]), D[i, j])
        return L[n - 1, m - 1]

    def fetchRealLocs(self,image_name_in_db):
        data_row = None
        with open(self.conf.db_meta_csv, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            column_index = header.index('Image_name')

            for row in reader:
                if row[column_index] == image_name_in_db.split(".")[0]:
                    data_row = row
                    break
        data_loc = osp.join(self.conf.invariant_dir, data_row[1], data_row[2], "invariant_traj.csv")
        start = int(data_row[3])
        end = int(data_row[4])
        data = np.genfromtxt(data_loc, delimiter=',')[start + 1:end + 2, [3, 4]]
        return data
    def PLTToPIL(self,fig):

        # Convert the fig to a numpy array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(h, w, 3)

        # Create a new PIL image using the numpy array
        dpi = 40
        pil_image = Image.fromarray(buf).resize((int(w * dpi / 72), int(h * dpi / 72)))

        return pil_image

    def generateImages(self,filepath,data_out_dir):

        # Initiate Window Parameters
        window_size = self.conf.window_size
        step_size = self.conf.step_size
        print("File Path",filepath)
        print("Window Size", window_size)
        print("Step Size", step_size)

        # Load Sequence
        csv_path = os.path.join(filepath, "invariant_traj.csv")
        new_seq = np.genfromtxt(csv_path, delimiter=',')[1:, :]

        # Check Whether Minimum Length Found
        if len(new_seq) < window_size:
            print("Failed Insufficient Length !!!")
            exit()
        else:
            print("Started Processing ")

        # Building Sliding Window Parameter
        max_index = len(new_seq) - 1
        current_start_index = 0
        current_end_index = current_start_index + window_size - 1

        # List to Store PIL Images
        locs=[]
        i=0
        while current_end_index <= max_index:
            i+=1
            # Get A Sliding Window
            window = new_seq[current_start_index:current_end_index + 1, :]
            current_start_index += step_size
            current_end_index += step_size

            # Generate and Save Image
            x = window[:, 1]
            y = window[:, 2]

            plt.plot(x, y, linestyle='-')
            plt.axis('off')
            # fig = plt.gcf()
            # img = self.PLTToPIL(fig)
            # pil_imgs.append(img)
            img_loc=osp.join(data_out_dir,str(i))
            locs.append(img_loc)
            plt.savefig(img_loc, dpi=40)
            plt.clf()

        return [Image.open(img+".png") for img in locs]
    def process(self):

        # Loading KDTree
        tree = None
        tags = None
        with open(self.conf.kdtree_features_loc, "rb") as f:
            tree = pickle.load(f)
            print("Loaded KDTree")
        with open(self.conf.kdtree_tags_loc, "rb") as f:
            tags = pickle.load(f)
            print("Loaded Tags")

        # List Directory
        files=os.listdir(self.conf.to_eval_dir)
        random.shuffle(files)
        print("No of Files Found :",len(files))
        count=0
        for file in files:
            count+=1
            # Setup Directories
            data_out_dir = osp.join(self.conf.history_output_loc,file)
            if os.path.exists(data_out_dir):
                shutil.rmtree(data_out_dir)
            os.makedirs(data_out_dir)

            # Generate Images
            fullPath=osp.join(self.conf.to_eval_dir,file)
            pil_imgs = self.generateImages(fullPath,data_out_dir)
            print("Generated {length} Images".format(length=len(pil_imgs)))

            # Get Real Location
            csv_path = os.path.join(fullPath, "invariant_traj.csv")
            real_loc = np.genfromtxt(csv_path, delimiter=',')[1:, [3,4]]

            # Extract Features
            print("Extracting Features  . . .")
            dbmanager=DBManager(self.conf)
            features = dbmanager.extract_features_all(pil_imgs)
            print("Features Shape :",features.shape)

            # Find Best Matching Locations
            print("Finding Best Matching Locations . . .")
            subTraj=0
            traj_layers=[]
            for feature in features:
                subTraj+=1
                print("-- For Sub Trajectory:",subTraj)

                layer=[]
                img_dist, ind = tree.query(feature, k=self.conf.no_of_candidates)

                if isinstance(ind,np.int64):
                    ind=[ind]

                for i in range(len(ind)):
                    best_image_name = tags[ind[i]]
                    predicted_real_loc = self.fetchRealLocs(best_image_name)
                    layer.append(predicted_real_loc)
                    #plt.scatter(x=predicted_real_loc[:, 0], y=predicted_real_loc[:, 1],s=0.01, linestyle='-')
                traj_layers.append(layer)

            X=[]

            for layer in traj_layers:
                for node in layer:
                    for coordinate in node:
                        X.append(coordinate)
                Y=np.array(X)
                X = dbscan_cluster(Y,real_loc)
            continue


            # # Filter Layers Using History
            # Filter Layers Using History
            no_of_layers = len(traj_layers)
            lines = set()
            passed = set()
            for i in range(no_of_layers - 1):
                print("Processing Layer {one} and {two}".format(one=i, two=i + 1))
                layer1 = traj_layers[i]
                layer2 = traj_layers[i + 1]
                layer1_no = i
                layer2_no = i + 1

                for traj1no, traj1 in enumerate(layer1):
                    if i > 0 and not any(line[-1] == (layer1_no, traj1no) for line in lines):
                        continue
                    for traj2no, traj2 in enumerate(layer2):
                        if self.is_overlaps(traj1, traj2):
                            passed.add(((layer1_no, traj1no), (layer2_no, traj2no)))

                if i == 0:
                    lines = passed.copy()
                    passed = set()
                else:
                    lines = {merged for line in lines for passes in passed
                             if line[-1] == passes[0] and (
                                 merged := line + passes[1:] if passes[1:] not in line else line)}
                    passed = set()

                plt.axis('off')
                plt.xlim((0, self.conf.x_lim))
                plt.ylim((0, self.conf.y_lim))
                print("No of Lines :", len(lines))
                # print(lines)

                plt.scatter(x=real_loc[:, 0], y=real_loc[:, 1], color="red", s=0.1)

                for color, line in zip(np.random.rand(len(lines), 3), lines):
                    for node in line:
                        plt.scatter(x=traj_layers[node[0]][node[1]][:, 0], y=traj_layers[node[0]][node[1]][:, 1],
                                    s=0.01, color=color)
                plt.savefig(osp.join(data_out_dir, "traj-" + str(i) + ".png"))
                # plt.show()
                plt.clf()

            # no_of_layers=len(traj_layers)
            # lines=[]
            # passed=[]
            # for i in range(no_of_layers-1):
            #     print("Processing Layer {one} and {two}".format(one=i,two=i+1))
            #     layer1=traj_layers[i]
            #     layer2=traj_layers[i+1]
            #     layer1_no=i
            #     layer2_no=i+1
            #
            #     for traj1no in range(len(layer1)):
            #         if i>0:
            #             inprevlayer=False
            #             for line in lines:
            #                 if line[-1]==(layer1_no,traj1no):
            #                     inprevlayer=True
            #                     break
            #             if not inprevlayer:
            #                 continue
            #         for traj2no in range(len(layer2)):
            #             traj1=layer1[traj1no]
            #             traj2=layer2[traj2no]
            #             if self.is_overlaps(traj1,traj2):
            #                 passed.append(((layer1_no,traj1no),(layer2_no,traj2no)))
            #     if i==0:
            #         lines=passed.copy()
            #         passed=[]
            #     else:
            #         templine=[]
            #         for line in lines:
            #             endnode=line[-1]
            #             for passes in passed:
            #                 startnode=passes[0]
            #                 if endnode==startnode:
            #                     merged = line
            #                     for element in passes:
            #                         if element not in line:
            #                             merged += (element,)
            #                     templine.append(merged)
            #         passed=[]
            #         lines=templine
            #
            #
            #
            #     plt.axis('off')
            #     plt.xlim((0,60))
            #     plt.ylim((0, 150))
            #     print("No of Lines :",len(lines))
            #     # print(lines)
            #
            #     plt.scatter(x=real_loc[:, 0], y=real_loc[:, 1], color="red", s=0.1)
            #
            #     for line_no in range(len(lines)):
            #         line = lines[line_no]
            #         color = np.random.rand(3)
            #
            #         for node in line:
            #             plt.scatter(x=traj_layers[node[0]][node[1]][:, 0], y=traj_layers[node[0]][node[1]][:, 1],s=0.01,color=color)
            #     plt.savefig(osp.join(data_out_dir,"traj-"+str(i)+".png"))
            #     #plt.show()
            #     plt.clf()

            #print("Start Plotting")
            # plt.axis('off')
            # for traj in passed:
            #     plt.scatter(x=traj_layers[node[0]][node[1]][:, 0], y=traj_layers[node[0]][node[1]][:, 1],s=0.01, linestyle='-')
            # plt.show()
            #break

from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

def dbscan_cluster(X,real_loc=None):
    # Instantiate DBSCAN with the given epsilon and min_samples
    dbscan = DBSCAN(eps=0.01, min_samples=5)

    # Fit the model to the data
    dbscan.fit(X)

    # Retrieve the cluster labels for each point
    labels = dbscan.labels_

    # Get the number of points in each cluster
    _, counts = np.unique(labels, return_counts=True)

    # Get the cluster label with the largest number of members
    max_label = np.argmax(counts)

    # Keep only the points that belong to the largest cluster
    X = X[labels == max_label]

    # Visualize the clustering results using a scatter plot
    plt.scatter(x=real_loc[:, 0], y=real_loc[:, 1], color="red", s=0.3)
    plt.scatter(X[:, 0], X[:, 1], cmap='viridis',s=0.1)
    plt.title('DBSCAN Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    # Return the cluster labels for each point
    return list(X)
