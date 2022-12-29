import config
from preprocessing.preprocessor import PreProcessor
from batch_generator.batchgenerator import BatchGenerator

steps=[2]

if __name__ == "__main__":

    if 1 in steps:
        print("\n### STEP 1 ###")
        print("PREPROCESSING")
        print("##############")
        preprocessor = PreProcessor(config)
        preprocessor.start()
    

    if 2 in steps:
        print("\n### STEP 2 ###")
        print("BATCH GENERATION")
        print("##############")
        batchgenerator = BatchGenerator(config)
        batchgenerator.start()