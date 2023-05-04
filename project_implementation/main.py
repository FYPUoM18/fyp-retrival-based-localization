import config
from DBManager.DBManager import DBManager
from DBManager.eval import Evaluator
from compile_dataset.compiler import Compiler
from csv_converter.csv_generator import CSV_Generator
from domain_mapper.convert_domain import DomainConverter
from train_test_val.train_test_val import TrainTestValSplitter
from traj_visualizer.traj_visualizer import TrajVisualizer
from history.HistoryModel import HistoryModel
from history.DataStorer import DataStorer
from history.LSTM import LSTM

steps = config.steps

if __name__ == "__main__":

    print("============================================\nUpdate Following Configs\n------------------------\n1. Run\n\t- steps : steps to run\n\n2. Dataset to Train\n\t- root dir : root directory\n\t- hdf5datadir : db/train/test/val/mobile directory\n\n3. Generate Train Test Val Data\n\t- freq : capturing frequency of data [Hz][*]\n\t- no_of_sec_per_split : window size for random cropping for train/test/val [s][*]\n\n4. Generate RoNIN Trajectories\n\t- ronin_checkpoint : where to find pretrained RoNIN ResNET Model\n\n5. Make Time Invariant\n\t- segment_length : segment length for resampling after interpolation [m][*]\n\n6. ImageDB Generate\n\t- window_size : no of segments per curve caputured for single image [*]\n\t- step_size : no of segments to skip for next image [*]\n\n[m] : Measured in Meters\n[Hz] : Measured in Hz\n[s] : Measured in Seconds\n[*] : Tune/Change According to Dataset\n============================================\n\n")

    if 1 in steps:
        csv_generator = CSV_Generator(config)
        csv_generator.generate()

    if 2 in steps:
        splitter = TrainTestValSplitter(config)
        splitter.split()

    if 3 in steps:
        data_compiler = Compiler(config)
        data_compiler.compile()

    if 4 in steps:
        traj_visualizer = TrajVisualizer(config)
        traj_visualizer.drawRoNINTraj()

    if 5 in steps:
        domain_convertor = DomainConverter(config)
        domain_convertor.make_time_invariant()

    if 6 in steps:
        generate_imagedb = DBManager(config)
        generate_imagedb.generateImageDB()

    if 7 in steps:
        generate_imagedb = DBManager(config)
        generate_imagedb.buildKDTree()

    if 8 in steps:
        evaluator = Evaluator(config)
        evaluator.evaluate()
    # if 8 in steps:
    #     storer=DataStorer(config)
    #     #storer.process(config.train_invariant_dir,"train")
    #     storer.process(config.test_invariant_dir, "test")
    #     storer.process(config.val_invariant_dir, "val")

    if 9 in steps:
        # LSTM_MODEL=LSTM(config)
        # LSTM_MODEL.train()
        historyModel=HistoryModel(config)
        historyModel.process()

    # if 10 in steps:
    #     LSTM_MODEL=LSTM(config)
    #     LSTM_MODEL.visualize()