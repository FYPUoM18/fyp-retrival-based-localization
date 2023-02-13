
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

    # TODO:     #Build Diagram [], 
    #           #VGG FEature Extractor
    #           #Use Feature Extractor and build offline DB
    #           #Runtime :For Each Image In Test/Val: Use Feature Exaractor-> Extract Features -> Find Top 10 matchings -> Note Rank/None
