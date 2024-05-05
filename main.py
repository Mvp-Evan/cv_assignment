import torch
import torchvision
from torch import nn
from torchvision import transforms

from vgg16 import VGG16


def plot(losses, accuracies):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.show()


def main():
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    DEVICE = torch.device("mps")
    BATCH_SIZE = 512
    EPOCHS = 100

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    vgg16 = VGG16(DEVICE, BATCH_SIZE, EPOCHS, plot, train_loader, test_loader, criterion)
    vgg16.train()
    vgg16.test()


if __name__ == '__main__':
    main()
