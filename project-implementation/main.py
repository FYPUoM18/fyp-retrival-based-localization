import config
from preprocessing.preprocessor import PreProcessor

steps=[2]

if __name__ == "__main__":

    if 1 in steps:
        preprocessor = PreProcessor(config)
        preprocessor.start()
    

    if 2 in steps:
        pass