import numpy as np
import h5py

class tran_test_val:

    def __init__(self,conf) -> None:
        self.conf=conf
    
    def generate(self):
        import h5py

    with h5py.File('example.hdf5', 'r') as f:
       
        time = f['synced/time'][:]

        duration = 20.0

        num_samples = int(duration * 100) # Assumes a 100Hz sampling rate

        # Iterate over a fixed number of samples
        for i in range(10):
            
            start_idx = np.random.randint(0, len(time) - num_samples)

            gyro = f['synced/gyro'][start_idx:start_idx+num_samples]
            acce = f['synced/acce'][start_idx:start_idx+num_samples]
            game_rv = f['synced/game_rv'][start_idx:start_idx+num_samples]

            file_name = f'output_{start_idx:05d}.txt'

            np.savetxt(file_name, np.c_[gyro, acce, game_rv], delimiter=' ')
