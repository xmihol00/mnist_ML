import sys
import numpy as np
import torch
import idx2numpy
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class TrainingDataset():
    def __init__(self):
        self.training_data = torch.from_numpy(np.expand_dims(idx2numpy.convert_from_file("./mnist/train-images.idx3-ubyte") / 255.0, axis=1)).to(torch.float32)
        self.training_labels = torch.from_numpy(np.array(idx2numpy.convert_from_file("./mnist/train-labels.idx1-ubyte")))

    def __getitem__(self, idx):
        return self.training_data[idx], self.training_labels[idx]

    def __len__(self):
        return self.training_labels.shape[0]

class TestingDataset():
    def __init__(self):
        self.testing_data = torch.from_numpy(np.expand_dims(idx2numpy.convert_from_file("./mnist/t10k-images.idx3-ubyte") / 255.0, axis=1)).to(torch.float32)
        self.testing_labels = torch.from_numpy(np.array(idx2numpy.convert_from_file("./mnist/t10k-labels.idx1-ubyte")))

    def __getitem__(self, idx):
        return self.testing_data[idx], self.testing_labels[idx]

    def __len__(self):
        return self.testing_labels.shape[0]

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 96, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(96*22*22, 10)
        )
    
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    classifier = MNIST_CNN()
    loss_function = nn.CrossEntropyLoss()

    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)

        for i in range(5):
            for images, labels in DataLoader(TrainingDataset(), 32):
                output = classifier(images)
                loss = loss_function(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Accuracy in epoch {i}: {1 - loss.item()}")

        with open("mnist_CNN.pt", "wb") as file:
            torch.save(classifier.state_dict(), file)
    else:
        with open("mnist_CNN.pt", "rb") as file:
            classifier.load_state_dict(torch.load(file))
        
        for image, label in DataLoader(TestingDataset(), 1):
            output = classifier(image)
            classified = torch.argmax(output).item()
            labeled = label.item()
            if labeled != classified:
                plt.imshow(image[0][0].numpy(), cmap='gray')
                plt.title(f"Image classified as {classified}, but labeled as {labeled}.")
                plt.show()
            
