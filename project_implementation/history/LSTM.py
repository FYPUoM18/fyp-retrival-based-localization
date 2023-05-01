import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from os import path as osp
import pickle
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the RNN model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=5, out_channels=16, kernel_size=2, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=2, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=2, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=6720, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 6720)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class Dataset:
    def __init__(self,type,conf):

        self.conf=conf
        self.features=None
        self.labels=None

        extracted_data_out_dir = osp.join(self.conf.extracted_data_output_loc, type)

        with open(extracted_data_out_dir + "\\p-features.pickle", "rb") as f:
            self.features =pickle.load(f)
        with open(extracted_data_out_dir + "\\p-lables.pickle", "rb") as f:
            self.labels = pickle.load(f)




    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):

        data = np.array(self.features[index]).astype(np.float32)
        labels = np.array(self.labels[index]).astype(np.float32)

        return data, labels


class LSTM():

    def __init__(self,conf):

        self.conf=conf
        self.input_size = 5 * 50 * 120 * 2
        self.output_size = 2  # Output size of the model
        self.train_batch_size = 1
        self.test_batch_size=1
        self.learning_rate = 0.001
        self.num_epochs = 50

    def train(self):

        #Create model and optimizer
        model = Model().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        train_dataloader = DataLoader(Dataset("train",self.conf), batch_size=self.train_batch_size, shuffle=True,drop_last=True)
        test_dataloader = DataLoader(Dataset("test", self.conf), batch_size=self.test_batch_size, shuffle=True,drop_last=True)

        # Train the model
        for epoch in range(self.num_epochs):
            for i, (data, labels) in enumerate(train_dataloader):
                # data = data.reshape((self.train_batch_size,-1))
                data = data.to(device)
                labels = labels.float().to(device)
                outputs = model(data)
                loss =  criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 89 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, self.num_epochs, i + 1, len(train_dataloader), loss.item()))

            # Evaluate the model on the test set
            with torch.no_grad():
                test_loss = 0.0
                total = 0
                correct = 0

                for i, (data, labels) in enumerate(test_dataloader):
                    data = data.to(device)
                    labels = labels.float().to(device)
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * labels.size(0)


                test_loss /= len(test_dataloader.dataset)

                print('Test set: Average loss: {:.4f}'.format(test_loss))

        # Save the model checkpoint
        torch.save(model.state_dict(), 'model.pth')

    def visualize(self):

        model_params = torch.load('model.pth', map_location=device)
        model = Model().to(device)
        model.load_state_dict(model_params)

        dataloader = DataLoader(Dataset("test", self.conf), batch_size=1, shuffle=True)

        with torch.no_grad():
            for i, (data, labels) in enumerate(dataloader):

                outputs = model(data)
                print(data.shape,labels[:,0],outputs[0])
                plt.axis('off')
                plt.xlim((0, self.conf.x_lim))
                plt.ylim((0, self.conf.y_lim))

                for layer in data[0]:
                    for node in layer:
                        plt.scatter(x=node[:, 0], y=node[:, 1], color="yellow", s=10)

                plt.scatter(x=labels[:,0], y=labels[:,1], color="red", s=10)
                plt.scatter(x=outputs[:, 0], y=outputs[:, 1], color="green", s=10)

                plt.show()

































        # first_features = None
        # second_features = None
        #
        # first_labels = None
        # second_labels = None
        #
        # with open(self.conf.extracted_data_output_loc + "\\train\\features-90.pickle", "rb") as f:
        #     first_features = pickle.load(f)
        #
        # with open(self.conf.extracted_data_output_loc + "\\train\\features.pickle", "rb") as f:
        #     second_features = pickle.load(f)
        #
        # with open(self.conf.extracted_data_output_loc + "\\train\\labels-90.pickle", "rb") as f:
        #     first_labels = pickle.load(f)
        #
        # with open(self.conf.extracted_data_output_loc + "\\train\\lables.pickle", "rb") as f:
        #     second_labels = pickle.load(f)
        #
        # print(len(first_features))
        # print(len(second_features))
        # print(len(first_labels))
        # print(len(second_labels))
        #
        # all_features = []
        # all_features.extend(first_features)
        # all_features.extend(second_features)
        #
        # all_labels = []
        # all_labels.extend(first_labels)
        # all_labels.extend(second_labels)
        #
        # print(len(all_features))
        # print(len(all_labels))
        #
        # with open(self.conf.extracted_data_output_loc + "\\train\\all-features.pickle", "wb") as f:
        #     pickle.dump(all_features, f)
        #
        # with open(self.conf.extracted_data_output_loc + "\\train\\all-labels.pickle", "wb") as f:
        #     pickle.dump(all_labels, f)
