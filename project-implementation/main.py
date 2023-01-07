import config
from preprocessing.preprocessor import PreProcessor
from pair_batch_generator.pairbatchgenerator import PairBatchGenerator
from contrastive_model.contrastivetrainer import ContrastiveTrainer
from batch_splitter.train_test_splitter import Train_Test_Splitter

steps=config.steps

if __name__ == "__main__":

    if 1 in steps:
        print("\n### STEP 1 ###")
        print("PREPROCESSING")
        print("##############")
        preprocessor = PreProcessor(config)
        preprocessor.start()
    

    if 2 in steps:
        print("\n### STEP 2 ###")
        print("PAIR BATCH GENERATION")
        print("##############")
        pairbatchgenerator = PairBatchGenerator(config)
        pairbatchgenerator.start()

    if 3 in steps:
        print("\n### STEP 3 ###")
        print("TRAIN CONTRSTIVE MODEL")
        print("################")
        contrastivetrainer = ContrastiveTrainer(config)
        contrastivetrainer.train()

    if 4 in steps:
        print("\n### STEP 4 ###")
        print("TRAIN TEST SPLITTER")
        print("################")
        ttsplitter = Train_Test_Splitter(config)
        ttsplitter.split_and_save()
        