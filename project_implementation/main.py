
import config
from compile_dataset.compiler import Compiler
from csv_converter.csv_generator import CSV_Generator
from feature_extractor.feature_extractor import FeatureExtractor
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

    # if 4 in steps:
    #     feature_extractor=FeatureExtractor(config)
    #     feature_extractor.generate_feature_db()
    #
    # if 5 in steps:
    #     feature_extractor=FeatureExtractor(config)
    #     feature_nps,feature_locs=feature_extractor.load_features_db()
    #     feature_extractor.find_best_matchings("loc_52_107",
    #     "C:\\Users\\mashk\\MyFiles\\Semester 7\\FYP\code\\project_implementation\outputs\\traj_visualized\\test\\"+
    #         "loc_52_107-d6be7ae7-0065-49f7-a7cb-bd5dd047b0a9.png",
    #     feature_nps,feature_locs,100)


    # TODO:     #Build Diagram [], 
    # One Shot Learning Try
    # Make Sure RoNIN Works Correct, If So Problem Almost Solved
    # Single Image With All Rankings TO Visualize Whther VGG works correct
    # Median Not Going To Work, Create CSV [Image ID],[Start X,End X], [Start Y, End Y]
    # Check New Images Mean X,Y within Its Start X,Y