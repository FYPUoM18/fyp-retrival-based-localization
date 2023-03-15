

class tran_test_val:

    def __init__(self,conf) -> None:
        self.conf=conf
    
    def generate(self):
        import h5py
import numpy as np

# Open the HDF5 file
with h5py.File('example.hdf5', 'r') as f:
    # Get the time dataset
    time = f['synced/time'][:]

    # Set the duration to 20 seconds
    duration = 20.0

    # Calculate the number of samples in the duration
    num_samples = int(duration * 100) # Assumes a 100Hz sampling rate

    # Iterate over a fixed number of samples
    for i in range(10):
        # Select a random starting time index
        start_idx = np.random.randint(0, len(time) - num_samples)

        # Get the gyro data for the current range
        gyro = f['synced/gyro'][start_idx:start_idx+num_samples]

        # Get the accelerometer data for the current range
        acce = f['synced/acce'][start_idx:start_idx+num_samples]

        # Construct the output file name
        file_name = f'output_{start_idx:05d}.txt'

        # Save the data to a text file
        np.savetxt(file_name, np.c_[gyro, acce], delimiter=' ')
