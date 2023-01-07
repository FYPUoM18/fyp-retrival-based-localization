import h5py
import random
import pickle

class Train_Test_Splitter():

    def __init__(self,conf) -> None:
        self.conf = conf
        print("Configurations Loaded ")
    
    def _load_processed_hdf5(self):
        print("Loading Preporcessed Data . . .")
        return h5py.File(self.conf.processed_hdf5_outputdir, 'r')

    def split_and_save(self):

        print("Splitter Started")
        hdf5file = self._load_processed_hdf5()

        print("Tagging PreProcessed Data . . .")
        locs = list(hdf5file.keys())

        print("No of Locations :",len(locs))
        tagged_data =[]

        for loc in locs:
            times = list(hdf5file.get(loc).keys())
            for time in times:
                tag = loc
                data = hdf5file.get(loc).get(time)[:]
                one_set = (tag,data)
                tagged_data.append(one_set)
        
        
        print("Randomizing")
        random.shuffle(tagged_data)
        total_len = len(tagged_data)
        train_len = int(total_len*self.conf.train_tagged_ratio) #1
        test_len = total_len-train_len #0
        selection = [1 for i in range(train_len)]+[0 for j in range(test_len)]
        random.shuffle(selection)

        print("Splitting . . .")
        train=[]
        test=[]

        for k in range(len(selection)):
            if selection[k]==1:
                train.append(tagged_data[k])
            else:
                test.append(tagged_data[k])
        print("Total No Of Data Points :",total_len)
        print("No Of Train Data Points :",len(train))
        print("No Of Test Data Points :",len(test))

        with open(self.conf.train_tagged_dat_outputdir,"wb") as f:
            pickle.dump(train,f)
            print("Saved Train Tagged Data")
        
        with open(self.conf.test_tagged_dat_outputdir,"wb") as f:
            pickle.dump(test,f)
            print("Saved Test Tagged Data")



        

        

