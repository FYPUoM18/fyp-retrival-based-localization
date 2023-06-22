import copy
import os
import random
import math
import numpy as np
import uuid
import h5py
import csv
from Configurations.Config import Config
from scipy.spatial.distance import cdist
from compile_dataset.compiler import Compiler
from traj_visualizer.traj_visualizer import TrajVisualizer
from domain_mapper.convert_domain import DomainConverter
from DBManager.DBManager import DBManager
from DBManager.eval import Evaluator
from scipy.spatial.distance import euclidean
from typing_extensions import Annotated
from os import path as osp
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File,Form
from pydantic import BaseModel
import os
import uuid
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import uvicorn

root_dir_of_output = "C:\\Users\\mashk\\MyFiles\\Semester 8\\FYP\\code\\project_implementation\\outputs"

class Predictor:

    def __init__(self,new_data_name,building_name,device_id,session_id,trainedBuilding_name="building_unib",freq = 200,no_of_sec_per_split = 45):
        self.out_dir = root_dir_of_output
        self.new_data_root_dir = f"{self.out_dir}\\mobile_pred\\{building_name}_{device_id}_{session_id}"
        self.building_root_dir = f"{self.out_dir}\\{trainedBuilding_name}"
        self.new_data_config = None
        self.building_data_config = None
        self.freq = freq
        self.no_of_sec_per_split = no_of_sec_per_split

    def generate_config(self):

        building_config = Config(root_dir="")
        building_config.load_config_from_json(f"{self.building_root_dir}\\config.json")

        new_data_config  = Config(root_dir=self.new_data_root_dir)
        new_data_config.x_lim = building_config.x_lim
        new_data_config.y_lim = building_config.y_lim
        new_data_config.hdf5datadir = {}
        new_data_config.freq = self.freq
        new_data_config.sample_rate = self.freq
        new_data_config.no_of_sec_per_split = self.no_of_sec_per_split
        new_data_config.segment_length = building_config.segment_length
        new_data_config.window_size = building_config.window_size
        new_data_config.step_size = building_config.step_size
        new_data_config.no_of_candidates = building_config.no_of_candidates
        new_data_config.ronin_checkpoint = building_config.ronin_checkpoint
        new_data_config.image_db_loc_kdtree = building_config.image_db_loc_kdtree
        new_data_config.kdtree_features_loc = building_config.kdtree_features_loc
        new_data_config.kdtree_tags_loc = building_config.kdtree_tags_loc
        new_data_config.merge_threshold = building_config.merge_threshold

        new_data_config.save_config_to_json()

        self.new_data_config=new_data_config
        self.building_data_config=building_config

    def getRoNINTrajectory(self):
        config = self.new_data_config
        data_compiler = Compiler(config)
        data_compiler.compile()

    def visualizeTrajectory(self):
        config = self.new_data_config
        csvmeta = [["OType", "Target Folder", "Source Folder", "Source Start Index", "Source End Index"]]
        np.savetxt(config.train_test_val_meta_file, np.array(csvmeta), delimiter=",", fmt='%s')
        traj_visualizer = TrajVisualizer(config)
        print("visualizing trajectory")
        traj_visualizer.drawRoNINTraj()

    def makeTimeInvariant(self):
        config = self.new_data_config
        domain_convertor = DomainConverter(config)
        domain_convertor.make_time_invariant()

    def generateImageDB(self):
        config = self.new_data_config
        generate_imagedb = DBManager(config)
        generate_imagedb.generateImageDB()

    def findMatchings(self):
        print("start finding matching")
        new_data_config = self.new_data_config
        building_data_config=self.building_data_config
        evaluator = Evaluator()
        layers,expected_locs = evaluator.evaluate(new_data_config,building_data_config)
        print("matchings predicted")
        return layers,expected_locs

    def getDTW(self,nplist1, nplist2):
        distance, path = fastdtw(nplist1, nplist2, dist=euclidean)
        return distance

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

    def calculate_ate(self,expected_locs, predicted_real_loc):
        # Calculate Euclidean distances between corresponding points
        distances = np.linalg.norm(expected_locs - predicted_real_loc, axis=1)

        # Calculate ATE
        ate = np.mean(distances)

        return ate
    def combineLocs(self, layer_1, layer_2):

        window_size = self.building_data_config.window_size
        step_size = self.building_data_config.step_size
        common_points_count = window_size - step_size
        locs = []

        min_dist = float('inf')
        selected_node_prev = None
        selected_node_post = None

        for node_prev in layer_1:
            for node_post in layer_2:

                for i in (1,-1):
                    for j in (1,-1):

                        node_prev_updated = node_prev[::i]
                        node_post_updated = node_post[::j]

                        node_middle_1 = node_prev_updated[step_size:, [0, 1]]
                        node_middle_2 = node_post_updated[:common_points_count, [0, 1]]

                        # dist = math.sqrt((node_prev_updated[-1, 0] - node_post_updated[0, 0]) ** 2 + (
                        #             node_prev_updated[-1, 1] - node_post_updated[0, 1]) ** 2)
                        dist = self.getDTW(node_middle_1,node_middle_2)
                        if dist<min_dist:
                            min_dist=dist
                            selected_node_prev=node_prev_updated
                            selected_node_post=node_post_updated

        return min_dist, selected_node_prev, selected_node_post, selected_node_post[-1]

    def euclidean(self, row_x_y_1, row_x_y_2):
        dist = math.sqrt((row_x_y_1[0] - row_x_y_2[0]) ** 2 + (row_x_y_1[1] - row_x_y_2[1]) ** 2)
        return dist

    def filterMatchings(self, layers, expected_locs):

        filtering_path = self.new_data_config.root_dir + "\\filterings"
        if os.path.exists(filtering_path):
            shutil.rmtree(filtering_path)
        os.mkdir(filtering_path)

        for i in range(len(layers)):

            initial_loc = expected_locs[i][0]
            expected_end_loc = expected_locs[i][-1]
            possible_paths = layers[i]

            predicted_end_loc = None
            min_dist = float('inf')
            best_path = None

            for path in possible_paths:
                # print(path[0])
                # print(path[-1])
                # print(initial_loc)
                dist_1 = self.euclidean(initial_loc,path[0])
                dist_2 = self.euclidean(initial_loc, path[-1])
                # print(dist_1)
                # print(dist_2)

                if dist_1<=min_dist:
                    min_dist=dist_1
                    best_path=path
                    predicted_end_loc = path[-1]

                if dist_2<=min_dist:
                    min_dist=dist_2
                    best_path=path
                    predicted_end_loc = path[0]

            err = self.euclidean(predicted_end_loc,expected_end_loc)
            name = str(uuid.uuid4())

            plt.xlim([0, self.building_data_config.x_lim])
            plt.ylim([0, self.building_data_config.y_lim])

            plt.scatter(best_path[:, 0], best_path[:, 1], color='blue', s=1)
            plt.scatter(expected_locs[i][:, 0], expected_locs[i][:, 1], color='red', s=1)

            plt.scatter(initial_loc[0], initial_loc[1], color='yellow', s=20)
            plt.scatter(expected_end_loc[0], expected_end_loc[1], color='black', s=20)
            plt.scatter(predicted_end_loc[0], predicted_end_loc[1], color='green', s=20)

            with open(self.new_data_config.to_err_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([name, err])

            plt.savefig(filtering_path + "\\" + name)
            plt.clf()


# Define the request body model
class PredictionRequest(BaseModel):
    building_name: str
    device_id: str
    session_id: str


# Create the FastAPI application
app = FastAPI()
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
from fastapi import FastAPI, UploadFile, File
from typing import List

app = FastAPI()


@app.post("/upload")
async def upload_files(building_name: Annotated[str, Form()], device_id: Annotated[str, Form()], session_id: Annotated[str, Form()], files: List[UploadFile] = File(...)):
    # Create the main directory if it doesn't exist
    print(building_name,device_id,session_id,files)
    main_directory = f"{root_dir_of_output}\\mobile_pred"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

    # Create the subdirectory
    subdirectory = f"{building_name}_{device_id}_{session_id}"
    subdirectory_path = os.path.join(main_directory, subdirectory)
    if not os.path.exists(subdirectory_path):
        os.makedirs(subdirectory_path)

    # Create the csv_data folder inside the subdirectory
    csv_folder_path = os.path.join(subdirectory_path, "1. csv_data","db","1")
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)

    # Save the files in the csv_data folder
    for file in files:
        file_path = os.path.join(csv_folder_path, file.filename)
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)

    # Return a response
    return {"message": "Files uploaded successfully"}


@app.post("/predict")
def predict_locations(request: PredictionRequest):
    try:
        building_name = request.building_name
        device_id = request.device_id
        session_id = request.session_id

        root_dir = root_dir_of_output
        prediction_dir = f"{root_dir}\\mobile_pred\\{building_name}_{device_id}_{session_id}"
        # Need to be repalced with session ID

        outname = os.listdir(prediction_dir)[0]
        print(outname)
        freq = 200
        predictor = Predictor(outname,building_name,device_id,session_id,building_name,freq)
        predictor.generate_config()
        print("Config generated")
        predictor.getRoNINTrajectory()
        print("RoNIN trajectory generated")
        predictor.visualizeTrajectory()
        print("Trajectory visualized")
        predictor.makeTimeInvariant()
        print("Time invariance applied")
        predictor.generateImageDB()
        print("Image DB generated")
        layers,expected_locs = predictor.findMatchings()
        print("Matchings found")
        predictor.filterMatchings(layers,expected_locs)
        print("Matchings filtered")

  

        return JSONResponse({"message": "Prediction completed successfully."})

    except Exception as ex:
        return JSONResponse({"message": str(ex)}, status_code=500)

@app.post("/predicton_images")
async def get_image( request_body: PredictionRequest):
    # Access the building_name, device_id, and session_id from the request_body
    building_name = request_body.building_name
    device_id = request_body.device_id
    session_id = request_body.session_id
    root_dir = root_dir_of_output

   # Retrieve the list of image files from the specified folder
    folder_path = f"{root_dir}\\mobile_pred\\{building_name}_{device_id}_{session_id}\\predictions"  # Replace "images" with the path to your folder
    image_files = os.listdir(folder_path)
    image_files.sort()  # Sort the list of files

    # Check if any image files exist
    if image_files:
        # Retrieve the last image file from the list
        first_image = image_files[-1]

        # Construct the full path to the first image file
        image_path = os.path.join(folder_path, first_image)

          # Read the image file as binary data
        with open(image_path, "rb") as file:
            image_data = file.read()

        # Convert the binary image data to Base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Return the first image file as a Base64-encoded response with the appropriate media type
        return JSONResponse({"image": image_base64}, media_type="application/json")
    else:
        # Raise an HTTPException with a 404 status code if no image files are found
        raise HTTPException(status_code=404, detail="No image files found")
    
@app.post("/filtered_images")
async def get_filtered_image( request_body: PredictionRequest):
    # Access the building_name, device_id, and session_id from the request_body
    building_name = request_body.building_name
    device_id = request_body.device_id
    session_id = request_body.session_id
    root_dir = root_dir_of_output

   # Retrieve the list of image files from the specified folder
    folder_path = f"{root_dir}\\mobile_pred\\{building_name}_{device_id}_{session_id}\\filterings"  # Replace "images" with the path to your folder
    image_files = os.listdir(folder_path)
    image_files.sort()  # Sort the list of files

    # Check if any image files exist
    if image_files:
        # Retrieve the last image file from the list
        first_image = image_files[-1]

        # Construct the full path to the first image file
        image_path = os.path.join(folder_path, first_image)

          # Read the image file as binary data
        with open(image_path, "rb") as file:
            image_data = file.read()

        # Convert the binary image data to Base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Return the first image file as a Base64-encoded response with the appropriate media type
        return JSONResponse({"image": image_base64}, media_type="application/json")
    else:
        # Raise an HTTPException with a 404 status code if no image files are found
        raise HTTPException(status_code=404, detail="No image files found")         


@app.get("/app_options")
def get_folders():
    root_dir = root_dir_of_output
    folder_path = f"{root_dir}\\mobile_pred"
    if not os.path.isdir(folder_path):
        return {"message": "Invalid folder path"}

    folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    unique_buildings = set()
    unique_devices = set()
    unique_sessions = set()

    for folder in folders:
        # Split the folder name using "_" as the delimiter
        parts = folder.split("_")

        if len(parts) == 3:
            building_name, device_id, session_id = parts

            unique_buildings.add(building_name)
            unique_devices.add(device_id)
            unique_sessions.add(session_id)

    return {
        "folders": folders,
        "unique_buildings": list(unique_buildings),
        "unique_devices": list(unique_devices),
        "unique_sessions": list(unique_sessions)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")