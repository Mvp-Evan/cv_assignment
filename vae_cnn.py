import torch
import torchvision
from torch import nn
from tqdm.rich import trange

from config import CNNConf, TrainConf, DatasetConf

transform = DatasetConf.Transforms

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

DEVICE = TrainConf.Device
BATCH_SIZE = TrainConf.BatchSize
EPOCHS = TrainConf.Epochs


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_layer_1 = nn.Linear(input_dim, hidden_dim)
        self.FC_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        h_1 = self.LeakyReLU(self.FC_layer_1(x))
        # R^{image_dim} \ni x -> = LeakyReLU(A_1(x)) = h_1 \in R^{hidden_dim}
        h_2 = self.LeakyReLU(self.FC_layer_2(h_1))
        # R^{hidden_dim} \ni h_1 -> LeakyReLU(A_2(h_1)) = h_2 \in R^{hidden_dim}
        mean = self.FC_mean(h_2)
        log_var = self.FC_var(h_2)
        # R^{hidden_dim} \ni h_2 -> (A_31(h_2),A_32(h_2)) = (mean,log_var) \in R^{hidden_dim} x R^{hidden_dim}

        # encoder produces mean and log of variance i.e., parameters of a Gaussian distribution "q"

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()

        self.FC_dec_layer_1 = nn.Linear(latent_dim, hidden_dim)
        self.FC_dec_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, z):
        dec_h_1 = self.LeakyReLU(self.FC_dec_layer_1(z))
        # R^{latent_dim} \ni z -> ReLU(B1(z)) = dec_h_1 \in R^{hidden_dim}

        dec_h_2 = self.LeakyReLU(self.FC_dec_layer_2(dec_h_1))
        # R^{hidden_dim} \ni dec_h_1 -> ReLU(B2(dec_h_1)) = dec_h_2 \in R^{hidden_dim}

        x_hat = torch.sigmoid(self.FC_output(dec_h_2))
        # R^{hidden_dim} \ni dec_h_2 -> Sigmoid(B3(dec_h_2)) = x_hat \in R^{output_dim}

        return x_hat


class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()

        self.Encoder = Encoder
        self.Decoder = Decoder

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        y = self.__reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.Decoder(y)

        return x_hat

    def __reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)
        # sampling epsilon ~ N(0,I_{latent-dimension x latent-dimension})
        y = mean + var*epsilon
        # The so-called "reparameterization trick"
        return y


class CNN(nn.Module):
    def __init__(self, VAE, output_dim):
        super(CNN, self).__init__()

        self.VAE = VAE

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
            x = self.VAE(x)
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
model = CNN(VAE(Encoder(32, 256, 20), Decoder(20, 256, 32)), 10).to(DEVICE)

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

