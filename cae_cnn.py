import torch
import torchvision
from torch import nn
from tqdm.rich import trange

from config import CNNConf, DatasetConf, TrainConf

transform = DatasetConf.Transforms

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

DEVICE = TrainConf.Device
BATCH_SIZE = TrainConf.BatchSize
EPOCHS = TrainConf.Epochs


class CAE(nn.Module):
    """
    Convolutional AutoEncoder
    """
    def __init__(self):
        super(CAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CNN(nn.Module):
    def __init__(self, CAE, output_dim):
        super(CNN, self).__init__()

        self.CAE = CAE

        self.cnn_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(8 * 16 * 16, 8 * 16 * 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8 * 16 * 16, output_dim),
            nn.Softmax(dim=1)
        )

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        if CNNConf.UseAutoEncoder:
            x = self.CAE(x)
        y = self.cnn_layer(x)
        return y


def one_hot_encode(y, num_classes):
    return torch.eye(num_classes)[y]


# accuracy function
def accuracy_fn(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=1)
    y_true = torch.argmax(y_true, dim=1)
    return torch.mean((y_pred == y_true).float())


# plot loss and accuracy
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


# load data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# model
model = CNN(CAE(), 10).to(DEVICE)

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=TrainConf.LearningRate)

# training
train_losses = []
train_accuracies = []
for epoch in trange(EPOCHS):
    model.train()
    loss = None
    outputs = None
    labels = None
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = one_hot_encode(labels, 10).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    train_losses.append(loss.item())
    train_accuracies.append(accuracy_fn(outputs, labels).item())
    print(f"Loss: {loss.item()}, Accuracy: {train_accuracies[-1]}")

plot(train_losses, train_accuracies)

# testing
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = one_hot_encode(labels, 10).to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

    print(f"Accuracy: {100 * correct / total}")

