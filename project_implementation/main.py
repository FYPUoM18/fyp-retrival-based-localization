import config
from DBManager.DBManager import DBManager
from compile_dataset.compiler import Compiler
from csv_converter.csv_generator import CSV_Generator
from domain_mapper.convert_domain import DomainConverter
from train_test_val.train_test_val import TrainTestValSplitter
from traj_visualizer.traj_visualizer import TrajVisualizer

steps = config.steps

if __name__ == "__main__":

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
