
import config
from compile_dataset.compiler import Compiler
from csv_converter.csv_generator import CSV_Generator
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

    # TODO: Put Sample Mobile With None Loc and ReGenerate all [PROGRESSING], 
    #       Build Diagram [], 
    #       Method To Draw Tajactories,
    #       Draw Trajectories For Original RoNIN/Our RoNIN, Cut Data To Make Similar Time Check Whether Similar Values