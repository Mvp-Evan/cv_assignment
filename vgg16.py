# use vgg16 to classify cifar-10 dataset

import torch
import torchvision
import torch.nn as nn
from torch import optim
from torchvision.models import VGG16_Weights
from tqdm.rich import trange


class VGG16():
    def __init__(self, device, batch_size, epochs, plot, train_loader, test_loader, criterion):

        model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(4096, 10)
        self.device = device
        self.model = model.to(self.device)
        self.plot = plot
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = criterion

    def train(self):
        train_losses = []
        train_accuracies = []
        for epoch in trange(self.epochs):
            self.model.train()
            loss = None
            outputs = None
            labels = None
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            train_losses.append(loss.item())
            train_accuracies.append((outputs.argmax(1) == labels).sum().item() / self.batch_size)

        self.plot(train_losses, train_accuracies)

    def test(self):
        """
        test by percentage correct
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy: {100 * correct / total:.2f}%")
