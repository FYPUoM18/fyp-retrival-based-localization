import pickle
import torch
import numpy as np
from contrastive_model.contrastivemodel import ContrastiveModel
from .NNTrainer import NNTrainer
from sklearn.metrics import accuracy_score


class Embedder:

    def __init__(self,conf) -> None:
        self.conf=conf


    def eval(self):

        print("Reading",self.conf.test_tagged_dat_outputdir)
        test_set = None
        with open(self.conf.test_tagged_dat_outputdir,'rb') as f:
            test_set = pickle.load(f)

        print("Initializing Contrastive Model . . .")
        model = ContrastiveModel(self.conf)
        print("Initialized Parameters")
        #print(list(model.parameters()))

        print("Loading Parameters")
        model.load_state_dict(torch.load(self.conf.contrastive_model_state_save_path))
        model.eval()
        print("Loaded Parameters")
        #print(list(model.parameters()))

        print("Create A Batch . . .")
        X = torch.from_numpy(np.array([data[1] for data in test_set])).float()
        y_real = [data[0] for data in test_set]
        print("X :",X.shape)
        print("y :",len(y_real))

        print("Get Embeddings . . .")
        embeddings = model(X)
        embeddings = embeddings.detach().numpy()
        print("Embedded Vectors Shape (Test) :",embeddings.shape)
        print("Labels Shape  (Test) :",len(y_real))

        print("Loading Model ",self.conf.nn_model_save_path)
        nn_loaded = pickle.load(open(self.conf.nn_model_save_path, 'rb'))

        print("Loading Tags ",self.conf.loc_tags_save_path)
        tag_list = None
        with open(self.conf.loc_tags_save_path,'r') as f:
            tag_list = [i.strip() for i in f.readlines()]
        



        print("Predicting ...")
        dist,indices=nn_loaded.query(embeddings,k=self.conf.n_neighbors_to_test)

        print("Mapping Indeces To Tag And Get Y_Pred . . .")
        y_pred=[]
        for one_pred in indices:
            mapped=[]
            for i in one_pred:
                mapped.append(tag_list[i])
            y_pred.append(mapped)

        # print(dist)
        # print(indices)
        # print(y_pred)

        print("Calc Accuracy . . .")
        total=len(y_real)
        correct=0
        wrong = 0
        for i in range(total):
            if y_real[i] in y_pred[i]:
                correct+=1
            else:
                wrong+=1
        print("Total :",total)
        print("Correct :",correct)
        print("Wrong :",wrong)
        print("Acc :",correct/total)

        # print(calc)
        # print(y[:100])
        # print(predicted_labels[:100])
        # acc = accuracy_score(y,predicted_labels)
        # print("Overal Accuracy :",acc)
        

    def embed(self):

        print("Reading",self.conf.train_tagged_dat_outputdir)
        train_set = None
        with open(self.conf.train_tagged_dat_outputdir,'rb') as f:
            train_set = pickle.load(f)
        
        print("Initializing Contrastive Model . . .")
        model = ContrastiveModel(self.conf)
        print("Initialized Parameters")
        #print(list(model.parameters()))

        print("Loading Parameters")
        model.load_state_dict(torch.load(self.conf.contrastive_model_state_save_path))
        model.eval()
        print("Loaded Parameters")
        #print(list(model.parameters()))

        print("Create A Batch . . .")
        X = torch.from_numpy(np.array([data[1] for data in train_set])).float()
        y = [data[0] for data in train_set]
        print("X :",X.shape)
        print("y :",len(y))

        print("Get Embeddings . . .")
        embeddings = model(X)
        embeddings =embeddings.detach().numpy()
        print("Embeddings Vectors Shape :",embeddings.shape)
        print("Embeddings Labels Shape :",len(y))

        print("Train KDTree Embedding Space . . .")
        nn_model = NNTrainer(self.conf)
        trained_nn_model=nn_model.trainNN(embeddings)

        print("NN Model Saving To",self.conf.nn_model_save_path)
        pickle.dump(trained_nn_model, open(self.conf.nn_model_save_path, 'wb'))

        print("Saving Location Tags To",self.conf.loc_tags_save_path)
        with open(self.conf.loc_tags_save_path, 'w') as f:
            for loc_tag in y:
                f.write("%s\n" % loc_tag)
        print('Done')


