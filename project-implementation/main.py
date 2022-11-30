import config
from preprocessing.preprocessor import PreProcessor

if __name__ == "__main__":
    preprocessor = PreProcessor(config)
    preprocessor.start()

