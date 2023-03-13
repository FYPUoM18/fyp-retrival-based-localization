
import config
from compile_dataset.compiler import Compiler
from csv_converter.csv_generator import CSV_Generator
from domain_mapper.convert_domain import DomainConverter
from traj_visualizer.traj_visualizer import TrajVisualizer


steps=config.steps

if __name__ == "__main__":
    
    if 1 in steps:
        csv_generator=CSV_Generator(config)
        csv_generator.generate()

    if 2 in steps:
        data_compiler=Compiler(config)
        data_compiler.compile()

    if 3 in steps:
        traj_visualizer=TrajVisualizer(config)
        traj_visualizer.drawRoNINTraj()

    if 4 in steps:
        domain_convertor=DomainConverter(config)
        domain_convertor.convert_domain()
