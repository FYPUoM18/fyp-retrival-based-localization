import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


class DomainConverter:

    def __init__(self, conf):
        self.conf = conf

    def make_time_invariant_ronin(self, data, segment_length):
        # Input coordinate array (latitude, longitude)
        # Skip Header
        coords = np.array(data[1:,[0,1]])

        # Separate the latitude and longitude coordinates into separate 1D arrays
        lat_coords = coords[:, 0]
        lon_coords = coords[:, 1]

        # Interpolate the latitude and longitude coordinates
        interp_func_lat = interp1d(np.arange(len(lat_coords)), lat_coords, kind='cubic')
        interp_func_lon = interp1d(np.arange(len(lon_coords)), lon_coords, kind='cubic')

        # Calculate the length of the curve
        curve_length = np.sum(np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1)))

        # Divide the curve into segments of segment_length
        num_segments = int(curve_length / segment_length)
        t = np.linspace(0, len(lat_coords) - 1, num_segments)
        lat = interp_func_lat(t)
        lon = interp_func_lon(t)

        # Combine the latitude and longitude coordinates into a single array
        segments = np.column_stack((lat, lon))

        return num_segments,segments

    def make_time_invariant_ground(self, data, num_segments):
        # Input coordinate array (latitude, longitude)
        # Skip Header
        coords = np.array(data[1:, [2, 3]])

        # Separate the latitude and longitude coordinates into separate 1D arrays
        lat_coords = coords[:, 0]
        lon_coords = coords[:, 1]

        # Interpolate the latitude and longitude coordinates
        interp_func_lat = interp1d(np.arange(len(lat_coords)), lat_coords, kind='cubic')
        interp_func_lon = interp1d(np.arange(len(lon_coords)), lon_coords, kind='cubic')

        # Divide the curve into segments of segment_length
        t = np.linspace(0, len(lat_coords) - 1, num_segments)
        lat = interp_func_lat(t)
        lon = interp_func_lon(t)

        # Combine the latitude and longitude coordinates into a single array
        segments = np.column_stack((lat, lon))

        return segments

    def cal_angular_code(self, ronin):
        codes=[np.NaN]
        p1 = ronin[0]
        p2 = ronin[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        first_angle = np.arctan2(dy, dx) * 180 / np.pi
        prev_angle = first_angle

        for i in range(1, len(ronin) - 1):
            p1 = ronin[i]
            p2 = ronin[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            angle = np.arctan2(dy, dx) * 180 / np.pi
            code = angle - prev_angle
            prev_angle = angle
            codes.append(code/self.conf.smooth_by_rounding)
        codes.append(np.NaN)
        return np.array(codes)

    def convert_domain(self):

        folders=list(os.listdir(self.conf.traj_drawing_out_dir))

        for folder in folders:
            datasets = list(os.listdir(os.path.join(self.conf.traj_drawing_out_dir,folder)))
            for dataset in datasets:

                # Get Required Paths
                csv_path = os.path.join(self.conf.traj_drawing_out_dir,folder,dataset,"full_traj.csv")
                png_path = os.path.join(self.conf.traj_drawing_out_dir, folder, dataset, "full_traj.png")
                data_out_dir = os.path.join(self.conf.invariant_domain_output_dir, folder, dataset)

                # Create Folders
                if os.path.exists(data_out_dir):
                    shutil.rmtree(data_out_dir)
                os.makedirs(data_out_dir)

                # Copy PNG
                shutil.copy(png_path,os.path.join(data_out_dir,"full_traj.png"))

                # Read CSV, Get No Of Segments, Make Time Invariant
                data = np.genfromtxt(csv_path, delimiter=',')
                segment_length = self.conf.segment_length
                num_segments,time_invariant_ronin = self.make_time_invariant_ronin(data, segment_length)
                time_invariant_ground = self.make_time_invariant_ground(data, num_segments)
                segments_ids=[id for id in range(num_segments)]

                # Calc Rotation
                angular_code = self.cal_angular_code(time_invariant_ronin)

                # Combine All
                combined=np.column_stack((segments_ids,time_invariant_ronin,time_invariant_ground,angular_code))[1:-1,:]

                # Plot and Save Traj
                x = combined[:, 1]
                y = combined[:, 2]
                real_x = combined[:, 3] - combined[0, 3]
                real_y = combined[:, 4] - combined[0, 4]
                img_path = os.path.join(data_out_dir,"invariant_traj.png")
                graph_path=os.path.join(data_out_dir,"graph.png")

                plt.scatter(x=x, y=y, s=0.01, linewidths=0.01, c="blue", label="RoNIN")
                plt.scatter(x=real_x, y=real_y, s=0.01, linewidths=0.01, c="red", label="Ground Truth")
                plt.axis('off')
                plt.savefig(img_path, dpi=1000)
                plt.clf()

                # Plot and Save Graph
                seg_ids=combined[:,0]
                angles = combined[:, 5]
                plt.plot(seg_ids, angles)
                plt.savefig(graph_path, dpi=1000)
                plt.clf()

                print("No Of Segments",num_segments)
                print("Saved", img_path)

                # Save CSV
                np.savetxt(os.path.join(data_out_dir,"invariant_traj.csv"), combined, delimiter=",", fmt='%.16f', comments='',
                           header=','.join(["id","i_ronin_x", "i_ronin_y", "i_ground_x", "i_ground_y","angle_to_rotate (RoNIN)"]))
                print("Saved Invariant Domain For",dataset)













