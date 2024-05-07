import torch
import torchvision
from torchvision import transforms


class DatasetConf:
    Type = ['CIFAR-10', 'STL-10'][0]
    InputDim = 32 if Type == 'CIFAR-10' else 96
    Root = './data'
    AllInGrey = False
    Transforms = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    TrainDataset = [
        torchvision.datasets.CIFAR10(root=Root, train=True, download=True, transform=Transforms),
        torchvision.datasets.STL10(root=Root, split='train', download=True, transform=Transforms)
    ][0 if Type == 'CIFAR-10' else 1]
    TestDataset = [
        torchvision.datasets.CIFAR10(root=Root, train=False, download=True, transform=Transforms),
        torchvision.datasets.STL10(root=Root, split='test', download=True, transform=Transforms)
    ][0 if Type == 'CIFAR-10' else 1]


class TrainConf:
    Epochs = 100
    BatchSize = 512
    LearningRate = 0.001
    Device = torch.device("mps")
    UseCluster = False


class CNNConf:
    UseAutoEncoder = True
