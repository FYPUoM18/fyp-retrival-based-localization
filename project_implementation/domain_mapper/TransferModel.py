import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import inception_v3
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


class ContrastiveInceptionV3(nn.Module):
    def __init__(self, embedding_size=1000):
        super(ContrastiveInceptionV3, self).__init__()
        self.inception_v3 = inception_v3(pretrained=True)
        self.inception_v3.fc = nn.Identity()  # Remove the final classification layer
        self.embedding_layer = nn.Linear(2048, embedding_size)  # Add a linear embedding layer

    def forward(self, x):
        x = self.inception_v3(x)
        x = x.logits
        x = self.embedding_layer(x)
        return x


class MNISTContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, label = self.mnist_dataset[idx]
        return img, label, idx


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels, indices):
        pairwise_distances = torch.cdist(embeddings, embeddings, p=2)
        same_class_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        negative_mask = torch.logical_not(same_class_mask)
        positive_distances = pairwise_distances[same_class_mask]
        negative_distances = pairwise_distances[negative_mask]
        loss = torch.mean(torch.maximum(torch.tensor(0.0), self.margin - positive_distances)) + torch.mean(
            torch.maximum(torch.tensor(0.0), negative_distances))
        return loss


class Trainer:
    def __init__(self, model, train_loader, val_loader, lr=0.001, margin=2.0, device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.margin = margin
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = ContrastiveLoss(margin=self.margin)

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss = 0.0
            self.model.train()
            count=0
            for x, y, _ in self.train_loader:
                count+=1
                x = x.to(self.device)
                y = y.to(self.device)
                x = x.repeat(1, 3, 1, 1)
                embeddings = self.model(x)
                loss = self.criterion(embeddings, y, None)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * x.shape[0]
                print("Step:",train_loss,count,len(self.train_loader))
            train_loss /= len(self.train_loader.dataset)
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss}")


    def test(self):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for x, y, _ in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                embeddings = self.model(x)
                pairwise_distances = torch.cdist(embeddings, embeddings, p=2)
                same_class_mask = torch.eq(y.unsqueeze(0), y.unsqueeze(1))
                different_class_mask = torch.logical_not(same_class_mask)
                same_class_distances = pairwise_distances[same_class_mask]
                different_class_distances = pairwise_distances[different_class_mask]
                distances = torch.cat((same_class_distances, different_class_distances))
                targets = torch.cat((torch.ones_like(same_class_distances), torch.zeros_like(different_class_distances)))
                predictions = torch.ge(distances, self.margin).float()
                correct += (predictions == targets).sum().item()
                total += targets.numel()
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

class Model:

    def __init__(self,conf):
        self.conf=conf
    def train_and_test_contrastive_inception_v3(self):
        # Define transform
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load MNIST dataset
        mnist_train = MNIST(".", train=True, transform=transform, download=True)
        mnist_val = MNIST(".", train=False, transform=transform, download=True)

        # Create contrastive dataset
        train_dataset = MNISTContrastiveDataset(mnist_train)
        val_dataset = MNISTContrastiveDataset(mnist_val)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Create model
        model = ContrastiveInceptionV3()

        # Create trainer
        trainer = Trainer(model, train_loader, val_loader, lr=0.001, margin=2.0)

        # Train model
        trainer.train(epochs=10)

        # Test model
        trainer.test()

        # Save model
        trainer.save(self.conf.model_path)
