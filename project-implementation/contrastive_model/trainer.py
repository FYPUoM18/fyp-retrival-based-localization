import torch
import h5py
import numpy as np
from .embedmodel import EmbedModel
import matplotlib.pyplot as plt
import random

class Trainer:

    def __init__(self,conf) -> None:
        self.conf = conf
    
    def ContastiveLoss(self,x1, x2, label, margin: float = 1.0):
        dist = torch.nn.functional.pairwise_distance(x1, x2)

        loss = (1 - label) * torch.pow(dist, 2) \
            + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
        loss = torch.mean(loss)

        return loss
    
    def _to_dataloader(self,hdf5):
        batch_names = list(hdf5.items())
        random.shuffle(batch_names)
        all_batches = []

        for batch_name in batch_names:
            one_batch_seq_1s = []
            one_batch_seq_2s = []
            one_batch_labels = []
            pairs = list(hdf5.get(batch_name[0]).items())
            random.shuffle(pairs)
            for pair in pairs:
                seq_1 = hdf5.get(batch_name[0]).get(pair[0]).get("1")
                seq_2 = hdf5.get(batch_name[0]).get(pair[0]).get("2")
                issimilar = 0 if "dis" in str(pair) else 1

                np_seq_1 = np.array(seq_1[:])
                np_seq_2 = np.array(seq_2[:])
    
                one_batch_seq_1s.append(np_seq_1)
                one_batch_seq_2s.append(np_seq_2)
                one_batch_labels.append(issimilar)
            
            one_batch_seq_1s = torch.from_numpy(np.array(one_batch_seq_1s)).float()
            one_batch_seq_2s = torch.from_numpy(np.array(one_batch_seq_2s)).float()
            one_batch_labels = torch.from_numpy(np.array(one_batch_labels)).float()

            all_batches.append((one_batch_seq_1s,one_batch_seq_2s,one_batch_labels))
        
        return all_batches

    def train(self):
        # Loading Data
        train_data_hdf5 = h5py.File(self.conf.train_hdf5_outputdir, 'r')
        val_data_hdf5 = h5py.File(self.conf.val_hdf5_outputdir, 'r')
        print("Data Loaded Into Memory")
        
        # Create Data Loaders
        train_data_loader = self._to_dataloader(train_data_hdf5)
        val_data_loader = self._to_dataloader(val_data_hdf5)
        print("Data Loaders Created")

        # Define Network
        model = EmbedModel(self.conf)
        loss_func = self.ContastiveLoss
        opt = torch.optim.Adam(model.parameters(), lr = 0.0005)
        print("Model Loaded")

        # History
        counter = []
        train_history = []
        val_history = []
        iter_no = 0

        # train
        print("Started Training")
        for epoch in range(self.conf.no_of_epochs):
            i = 0
            for seq_1,seq_2,label in train_data_loader:
                
                # Zero The Grads
                opt.zero_grad()
                
                # Get Outputs
                embed_1 = model(seq_1)
                embed_2 = model(seq_2)

                # Calc Loss
                loss = loss_func(embed_1,embed_2,label)
                
                # Calc Gradients
                loss.backward()
                
                # Update Weights
                opt.step()
                
                #Every 10 batch print result
                if i%10 ==0:
                    val_loss = []
                    for seq_1,seq_2,label in val_data_loader:
                        
                        # Get Outputs
                        embed_1 = model(seq_1)
                        embed_2 = model(seq_2)
                
                        # Calc Loss
                        val_loss.append(loss_func(embed_1,embed_2,label).item())
                    
                    val_loss_mean = sum(val_loss) / len(val_loss)
                    print("Epoch No :",epoch,"iter :",i,"Current Loss:",loss.item(),"Validation Loss : ",val_loss_mean)
                    iter_no+=10
                    counter.append(iter_no)
                    train_history.append(loss.item())
                    val_history.append(val_loss_mean)
                i+=1
        
        plt.plot(counter, train_history, color='r', label='Train')
        plt.plot(counter, val_history, color='g', label='Validation')
        plt.ylim(0, 1)
        fig = plt.gcf()
        plt.show()
        fig.savefig(self.conf.training_graph_output)
        print("Saved Graph")
        torch.save(model.state_dict(), self.conf.model_state_save_path)
        print("Saved Model")