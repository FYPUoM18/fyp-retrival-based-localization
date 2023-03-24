import argparse
import json
import os
import shutil
import sys
from os import path as osp

import h5py
import numpy as np
import scipy
from compile_dataset.math_utils import interpolate_quaternion_linear
from compile_dataset.ronin_resnet_minimal import test_sequence

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))

'''
HDF5 data format
* data.hdf5
     |---raw
     |    |--- imu
     |          |---gyro, gyro_uncalib, acce, magnet, game_rv, gravity, linacce, step. rv, pressure, wifi, gps, 
     magnetic_rv, magnet_uncalib
     |---synced
     |    |---time, gyro, acce, game_rv, 
     |---filtered
     |    |---[Optional] flp, locations
     |---computed
     |----|---ronin_traj (from ronin_resnet)
  The HDF5 file stores all data. "raw" subgroup store all unprocessed data. "synced" subgroup stores synchronized data.
* info.json
  Stores meta information such as reference time.
To read a HDF5 dataset:
import h5py
with h5py.File(<path-to-hdf5-file>) as f:
     gyro = f['synced/gyro']
     acce = f['synced/acce']
     .....
NOTICE: the HDF5 library will not read the data until it's actually used. For example, all data domains in the
        above code are NOT actually read from the disk. This means that if you try to access "gyro" or "acce" 
        etc. after the "with" closure is released, an error will occur. To avoid this issue, use:
               gyro = np.copy(f['synced/gyro'])
        to force reading.
'''


class Compiler:

    def __init__(self, conf) -> None:
        self.conf = conf

    def interpolate_vector_linear(self, input, input_timestamp, output_timestamp):
        """
        This function interpolate n-d vectors (despite the '3d' in the function name) into the output time stamps.
        Args:
            input: Nxd array containing N d-dimensional vectors.
            input_timestamp: N-sized array containing time stamps for each of the input quaternion.
            output_timestamp: M-sized array containing output time stamps.
        Return:
            quat_inter: Mxd array containing M vectors.
        """
        assert input.shape[0] == input_timestamp.shape[0]
        func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
        interpolated = func(output_timestamp)
        return interpolated

    def process_data_source(self, raw_data, output_time, method):
        input_time = raw_data[:, 0]
        if method == 'vector':
            output_data = self.interpolate_vector_linear(
                raw_data[:, 1:], input_time, output_time)
        elif method == 'quaternion':
            assert raw_data.shape[1] == 5
            output_data = interpolate_quaternion_linear(
                raw_data[:, 1:], input_time, output_time)
        else:
            raise ValueError(
                'Interpolation method must be "vector" or "quaternion"')
        return output_data

    def compute_output_time(self, all_sources, sample_rate=200):
        """
        Compute the output reference time from all data sources. The reference time range must be within the time range of
        all data sources.
        :param data_all:
        :param sample_rate:
        :return:
        """
        interval = 1. / sample_rate
        min_t = max([data[0, 0] for data in all_sources.values()]) + interval
        max_t = min([data[-1, 0] for data in all_sources.values()]) - interval
        return np.arange(min_t, max_t, interval)

    def get_ronin_resnet_traj(self, data_path, ronin_checkpoint):

        kwargs = {'step_size': 10,
                  'window_size': 200,
                  'model_path': ronin_checkpoint,
                  'feature_sigma': 0.00001}
        traj = test_sequence(data_path, **kwargs)
        return traj

    def load_wifi_dataset(self, path):
        """
        Process WiFi records
        :param path: path to file
        :return: df_num : [scan_no, last_timestamp, level, frequency, freq0, freq1, channelwidth] - for each record
                df_mac_addr : [bssid, ssid]    - for each record
                scans - [scan_no, num_access_points, [If available] recorded timestamp, is_success] - for each scan
        """
        df_mac_addr, df_num = [], []
        scans = []
        scan_no = 0

        with open(path, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                values = line.split()
                if len(values) == 3:
                    if values[2] == 'f':
                        continue
                    scan_no += 1
                    scans.append([scan_no, int(values[0]), int(
                        values[1]), 1 if values[2] == 's' else 0])
                elif len(values) == 1:
                    scan_no += 1
                    scans.append([scan_no, int(values[0]), 0, 1])
                else:
                    df_mac_addr.append(
                        [values[1], " " if len(values) < 8 else values[7]])
                    df_num.append(
                        [scan_no, int(values[0]), int(values[2]), int(values[3]), int(values[4]), int(values[5]),
                         int(values[6])])
        df_mac_addr = np.array(df_mac_addr, dtype=object)
        df_num = np.array(df_num, dtype=np.int)
        return df_num, df_mac_addr, np.asarray(scans)

    def load_map_data(self, path):
        """
        Load manualy clicked locations from file and merge data across files and divide by floorplan.
        :param path: paths to maps folder
        :return: data : [timestamp, image_coordinates(x, y), longitude, lattitude] - for each valid record
                count : [start index (in data), num of records in data] for each tour (consecutive points in same
                floorplan)
                building: array of strings of floorplan name for each tour
        """
        mapfiles = os.listdir(path)
        mapfiles.sort()
        lines = []
        for mapfile in mapfiles:
            if not mapfile.endswith('.txt'):
                continue
            with open(osp.join(path, mapfile), 'r') as f:
                for line in f:
                    if line[0] == '#' or not line.strip():
                        continue
                    if line.strip() == 'CANCEL SESSION':
                        break
                    if line.split()[-1] == "DELETE":
                        continue
                    lines.append(line)

        data, count, building = [], [], []
        b, c, l = False, 0, 0
        for line in lines:
            line = line.split()
            if b == line[0]:
                l += 1
                c += 1
                data.append([int(line[1]), float(line[2]), float(
                    line[3]), float(line[4]), float(line[5])])
            else:
                print(b, c)
                if c > 0:
                    building.append(b)
                    count[-1][-1] = c
                data.append([int(line[1]), float(line[2]), float(
                    line[3]), float(line[4]), float(line[5])])
                count.append([l, 0])
                b = line[0]
                c = 1
                l += 1
        print(b, c)
        if c > 0:
            building.append(b)
            count[-1][-1] = c

        return np.asarray(data), np.asarray(count), building

    # def filter_data(self, data, output_time, reference_time=None):
    #     # return data with in the output time range
    #     if data is None or len(data) < 1:
    #         return None

    #     if reference_time is not None:
    #         data[:, 0] = (data[:, 0] - reference_time) /divider
    #     data = data[np.logical_and(
    #         data[:, 0] >= output_time[0], data[:, 0] <= output_time[-1])]
    #     return data

    def compile_unannotated_sequence(self, root_dir, data_list, ronin_checkpoint, out_dir, isnano, is_loc_available):
        """
        Compile unannotated(or imu_only) sequence directly from raw files.
        """
        source_vector = {'gyro', 'acce','ronin'}
        source_quaternion = {'game_rv'}
        fail_list = []
        divider = 1000000000 if isnano else 1
        _raw_data_sources = ['gyro', 'acce', 'game_rv','ronin']
        _optional_data_sources = []
        if is_loc_available:
            source_vector.add('loc')
            _raw_data_sources.append('loc')
        source_all = source_vector.union(source_quaternion)

        for data in data_list:
            try:
                data_path = osp.join(root_dir, data)
                out_path = osp.join(out_dir, data)
                if osp.isdir(out_path):
                    shutil.rmtree(out_path)
                    os.makedirs(out_path)

                else:
                    os.makedirs(out_path)

                # mapdata, mapcount, mapbuilding = self.load_map_data(osp.join(root_dir, data, 'map'))

                print('Compiling {} to {}'.format(data_path, out_path))

                all_sources = {}
                # We assume that all IMU/magnetic/pressure files must exist.
                try:
                    reference_time = np.loadtxt(
                        osp.join(data_path, 'reference_time.txt')).tolist()
                except:
                    reference_time = 0
                for source in source_all:
                    try:
                        source_path = osp.join(root_dir, data, source + '.txt')
                        source_data = np.genfromtxt(source_path)
                        source_data[:, 0] = (source_data[:, 0] - reference_time) / divider
                        all_sources[source] = source_data
                    except OSError:
                        print('Can not find file for source {}. Please check the dataset.'.format(
                            source_path))
                        continue
                if osp.exists(osp.join(root_dir, data, 'pose.txt')):
                    pose = np.genfromtxt(osp.join(root_dir, data, 'pose.txt'))
                    pose[:, 0] = (pose[:, 0] - reference_time) / divider
                    all_sources['tango_pos'] = pose[:, :4]
                    all_sources['tango_ori'] = pose[:, [0, -4, -3, -2, -1]]
                    source_vector.add('tango_pos')
                    source_quaternion.add('tango_ori')

                output_time = self.compute_output_time(all_sources)
                processed_sources = {}
                for source in all_sources.keys():
                    if source in source_vector:
                        processed_sources[source] = self.process_data_source(
                            all_sources[source], output_time, 'vector')
                    else:
                        print(output_time)
                        processed_sources[source] = self.process_data_source(
                            all_sources[source][:, [0, 4, 1, 2, 3]], output_time, 'quaternion')

                meta_info = {'type': 'unannotated',
                             'length': output_time[-1] - output_time[0],
                             'imu_reference_time': reference_time}
                json.dump(meta_info, open(
                    osp.join(out_path, 'info.json'), 'w'))
                with h5py.File(osp.join(out_path, 'data.hdf5'), 'x') as f:
                    # f.create_group('raw/imu')
                    # for source in _raw_data_sources:
                    #     print(source)
                    #     f.create_dataset('raw/imu/' + source,
                    #                      data=np.genfromtxt(osp.join(data_path, source + '.txt')))
                    # for source in _optional_data_sources:
                    #     if osp.isfile(osp.join(data_path, source + '.txt')):
                    #         if source == 'wifi':
                    #             data_num, data_string, scans = self.load_wifi_dataset(
                    #                 osp.join(data_path, source + '.txt'))
                    #             f.create_dataset(
                    #                 'raw/imu/wifi_values', data=data_num)
                    #             f.create_dataset('raw/imu/wifi_address', data=data_string,
                    #                              dtype=h5py.special_dtype(vlen=str))
                    #             f.create_dataset(
                    #                 'raw/imu/wifi_scans', data=scans)
                    #         else:
                    #             try:
                    #                 f.create_dataset('raw/imu/' + source,
                    #                                  data=np.genfromtxt(osp.join(data_path, source + '.txt')))
                    #             except IOError:
                    #                 pass

                    f.create_group('synced')
                    f.create_dataset('synced/time', data=output_time)
                    for source in source_all:
                        if source == 'ronin':
                            continue
                        else:
                            f.create_dataset(
                                'synced/' + source, data=processed_sources[source])

                    # f.create_group('filtered')
                    # flp = self.filter_data(np.genfromtxt(osp.join(data_path, 'FLP.txt')), output_time, reference_time)
                    # if flp is not None:
                    #     f.create_dataset('filtered/flp', data=flp)
                if 'tango_pos' in source_vector:
                    f.create_group('pose')
                    f.create_dataset('pose/tango_pos',
                                     data=processed_sources['tango_pos'])
                    f.create_dataset('pose/tango_ori',
                                     data=processed_sources['tango_ori'])

                print('Calculate ronin trajectory')
                # ronin_traj = self.get_ronin_resnet_traj(
                #     out_path, ronin_checkpoint)
                with h5py.File(osp.join(out_path, 'data.hdf5'), 'a') as f:
                    f.create_group('computed')
                    f.create_dataset('computed/ronin', data=processed_sources["ronin"])

            except (OSError, FileNotFoundError, TypeError) as e:
                print(e)
                fail_list.append(data)

        print('Fail list:')
        [print(data) for data in fail_list]

    def compile(self):

        csv_dir = self.conf.csv_out_dir
        folder_list = [osp.split(path)[1] for path in os.listdir(csv_dir)]

        for folder in folder_list:
            root_dir = osp.join(csv_dir, folder)
            data_list = [osp.split(path)[1] for path in os.listdir(root_dir)]
            out_dir = osp.join(self.conf.processed_hdf5_out_dir, folder)
            ronin_checkpoint = self.conf.ronin_checkpoint

            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir)

            is_nano = False
            is_loc_available = True
            if folder == "mobile":
                is_nano = True
                is_loc_available = False

            self.compile_unannotated_sequence(
                root_dir, data_list, ronin_checkpoint, out_dir, is_nano, is_loc_available)
